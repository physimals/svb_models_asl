"""
Inference forward models for ASL data with dispersion
"""

import tensorflow as tf

import tensorflow_probability as tfp
import numpy as np

from ssvb.model import Model, ModelOption

from svb_models_asl import __version__
from svb_models_asl.aslrest import AslRestModel

class AslRestDisp(AslRestModel):
    """
    Implementation of the ASL resting state model with explicit convolution of AIF
    and residue function, to enable incorporation of dispersion
    """
    
    OPTIONS = AslRestModel.OPTIONS + [
        ModelOption("conv_dt", "Time interval for numerical convolution", units="s", type=float, default=0.1),
        ModelOption("conv_type", "Convolution type ('gamma' only supprted type presently)", type=str, default="gamma"),
        ModelOption("infer_disp_params", "Whether to infer parameters of the dispersion", type=bool, default=True),
    ]

    def __init__(self, data_model, **options):
        AslRestModel.__init__(self, data_model, **options)
        if self.infer_disp_params:
            self.params.append(
                self.attach_param("s", dist="LogNormal", mean=7.4, var=2.0, **options)
            )
            self.params.append(
                self.attach_param("sp", dist="LogNormal", mean=0.74, var=2.0, **options)
            )

        # t values for numerical evaluation of convolution [NT]
        self.conv_tmax = max(max(self.tis), 5.0)
        self.conv_nt = 1 + int(self.conv_tmax / self.conv_dt)
        self.conv_t = np.linspace(0.0, self.conv_tmax, self.conv_nt)

    def __str__(self):
        return "ASL resting state model with gamma dispersion: %s" % __version__

    def tissue_signal(self, t, ftiss, delt, t1, extra_params):
        """
        ASL kinetic model for tissue with dispersion implemented by 
        convolution of AIF and residue function
        """
        ftiss = self.log_tf(ftiss, name="ftiss", shape=True, force=False)
        delt = self.log_tf(delt, name="delt", shape=True, force=False)

        # Kinetic curve is numerical convolution of AIF and residue on discrete timepoints
        conv_t = tf.constant(self.conv_t, dtype=tf.float32)
        aif = self.log_tf(self.aif_gammadisp(conv_t, delt, extra_params), force=False, shape=True, name="aif") # [W, S, NT]
        resid = self.log_tf(self.resid_wellmix(conv_t, t1), force=False, shape=True, name="resid") # [NT]
        kinetic_curve = self.log_tf(self.conv_tf(aif, resid, self.conv_dt), force=False, shape=True, name="conv") # [W, S, NT]

        # Sample the kinetic curve at the TIs (using linear interpolation here)
        signal =  self.log_tf(tfp.math.batch_interp_regular_1d_grid(t, 0, self.conv_tmax, kinetic_curve, axis=-1), force=False, shape=True, name="sig") # [W, S, B]
        return ftiss*signal

    def art_signal(self, t, fblood, deltblood, extra_params):
        return fblood * self.aif_gammadisp(t, deltblood, extra_params)

    def aif_gammadisp(self, t, delt, extra_params):
        """
        pCASL/PASL AIF with Gamma dispersion

        :param t: Time points to evaluate AIF curve at [NT]. Note that this
                  is a sequence of evenly spaced points up to a maximum which is
                  based on the maximum TI - it does NOT depend on the node/voxel 
                  and is NOT the same as the timepoints passed in to evaluate 
        :param delt: ATT values [W, S, 1] or [W, 1]

        :return: kcblood [W, S, NT] or [W, NT]
        """
        # Dispersion parameters - note range check as per Fabber
        if self.infer_disp_params:
            s = extra_params[0]
            sp = extra_params[1]
            sp = tf.clip_by_value(sp, -1e12, 10)
        else:
            # Prior defaults from Fabber
            s = 7.4
            sp = 0.74
    
        pre_bolus = self.log_tf(tf.less(t, delt, name="aif_pre_bolus"), shape=True)
        post_bolus = self.log_tf(tf.greater(t, tf.add(delt, self.tau), name="aif_pre_bolus"), shape=True)
        during_bolus = tf.logical_and(tf.logical_not(pre_bolus), tf.logical_not(post_bolus))

        # Calculate AIF
        if self.casl:
            kcblood_nondisp = self.log_tf(2 * tf.exp(-delt / self.t1b), name="kcblood_nodisp", force=False, shape=True)
        else:
            kcblood_nondisp = 2 * tf.exp(-t / self.t1b)

        # This part applies the Gamma function dispersion. Note the need to clip the time argument to >=0 otherwise
        # we get NaN in the gradient which does not disappear even though it does not affect the output
        k = 1 + sp
        gamma1 = self.log_tf(tf.math.igammac(k, s * tf.clip_by_value(t - delt, 0, 1e6)), name="gamma1", force=False, shape=True)
        gamma2 = self.log_tf(tf.math.igammac(k, s * tf.clip_by_value(t - delt - self.tau, 0, 1e6)), name="gamma2", force=False)
        kcblood = tf.zeros(tf.shape(during_bolus), dtype=tf.float32)
        kcblood = tf.where(during_bolus, kcblood_nondisp * (1 - gamma1), kcblood)
        kcblood = tf.where(post_bolus, kcblood_nondisp * (gamma2 - gamma2), kcblood)

        return self.log_tf(kcblood, force=False, shape=True, name="kcblood")

    def aif_nodisp(self, t, delt, extra_params):
        """
        Non-dispersed AIF

        See aif_gammadisp for parameters. This should give the same output as the aslrest model
        (not identical because of numerical convolution), and exists only for testing purposes
        """
        pre_bolus = self.log_tf(tf.less(t, delt, name="aif_pre_bolus"), shape=True)
        post_bolus = self.log_tf(tf.greater(t, tf.add(delt, self.tau), name="aif_post_bolus"), shape=True)
        during_bolus = tf.logical_and(tf.logical_not(pre_bolus), tf.logical_not(post_bolus))

        if self.casl:
            kcblood_nondisp = tf.repeat(2 * tf.exp(-delt / self.t1b), tf.shape(t)[0], axis=-1)
        else:
            kcblood_nondisp = 2 * tf.exp(-t / self.t1b)
        
        kcblood = tf.zeros(tf.shape(during_bolus), dtype=tf.float32)
        kcblood = tf.where(during_bolus, kcblood_nondisp, kcblood)

        return kcblood

    def resid_wellmix(self, t, t1):
        """
        Residue function for well mixed single compartment

        Buxton (1998) model
        
        :param t: Time values to evaluate at [NT]
        :param t1: Tissue T1 values (scalar or [W, S, 1])

        :return: residue function [NT] or [W, S, NT]
        """
        # FIXME does variable t1 work here?
        t1_app = 1 / (1 / t1 + self.fcalib / self.pc)
        return(tf.math.exp(-t / t1_app))

    def conv_tf(self, data, kernel, dt):
        """
        Convolution implemented in Tensorflow
        
        To make it work we need to reverse the kernel as Tensorflow's
        'convolution' is actually cross correlation. Secondly we need
        to pad the kernel with zeros to make sure we get enough output
        values. Plus a bit of reshaping of the data and kernel tensors
        just because.
        
        :param data: Data [W, NT] or [W, S, NT]
        :param kernel: Convolution kernel [NT]
        :param dt: Time interval between samples

        :return: Convolution of data with kernel [W, NT] or [W, S, NT]
        """ 
        kernel = kernel[::-1] # [NT]
        kernel = tf.pad(kernel, [[0, tf.shape(kernel)[0] - 1]]) # [NTP]
        data_shape = tf.shape(data)
        data = tf.expand_dims(tf.reshape(data, [-1, data_shape[-1]]), -1) # [W, NT, 1] or [W*S, NT, 1]
        kernel = tf.reshape(kernel, [kernel.shape[0], 1, 1]) # [NTP, 1, 1]
        result_tf = tf.squeeze(tf.nn.conv1d(data, kernel, 1, 'SAME')) * dt

        return tf.reshape(result_tf, data_shape)
