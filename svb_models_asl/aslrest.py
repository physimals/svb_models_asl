"""Inference forward models for ASL data"""

import copy 

import tensorflow as tf
import numpy as np
from scipy import ndimage

from svb.model import Model, ModelOption
from svb.utils import ValueList, NP_DTYPE, TF_DTYPE

from svb_models_asl import __version__

TIME_SCALE = 1e1

class AslRestModel(Model):
    """
    ASL resting state model

    FIXME integrate with oxasl AslImage class?
    """


    OPTIONS = [

        # ASL parameters 
        ModelOption("tau", "Bolus duration", units="s", clargs=("--tau", "--bolus"), type=float, default=TIME_SCALE * 1.8),
        ModelOption("casl", "Data is CASL/pCASL", type=bool, default=False),
        ModelOption("tis", "Inversion times", units="s", type=ValueList(float)),
        ModelOption("plds", "Post-labelling delays (for CASL instead of TIs)", units="s", type=ValueList(float)),
        ModelOption("repeats", "Number of repeats - single value or one per TI/PLD", units="s", type=ValueList(int), default=[1]),
        ModelOption("slicedt", "Increase in TI/PLD per slice", units="s", type=float, default=0),

        # GM tissue properties 
        ModelOption("t1", "Tissue T1 value", units="s", type=float, default=TIME_SCALE * 1.3),
        ModelOption("att", "Bolus arrival time", units="s", clargs=("--bat",), type=float, default=TIME_SCALE * 1.3),
        ModelOption("attsd", "Bolus arrival time prior std.dev.", units="s", clargs=("--batsd",), type=float, default=None),
        ModelOption("fcalib", "Perfusion value to use in estimation of effective T1", type=float, default=0.01),
        ModelOption("pc", "Blood/tissue partition coefficient. If only inferring on one tissue, default is 0.9; if inferring on both GM/WM default is 0.98/0.8 respectively. See --pcwm", type=float, default=None),

        # WM tissue properties 
        ModelOption("incwm", "Include WM parameters", default=False),
        ModelOption("fwm", "WM perfusion", type=float, default=0),
        ModelOption("attwm", "WM arterial transit time", clargs=("--batwm",), type=float, default=TIME_SCALE * 1.6),
        ModelOption("t1wm", "WM T1 value", units="s", type=float, default=TIME_SCALE * 1.1),
        ModelOption("pcwm", "WM parition coefficient. See --pc", type=float, default=0.8),
        ModelOption("fcalibwm", "WM perfusion value to use in estimation of effective T1", type=float, default=0.003),

        # Blood / arterial properties 
        ModelOption("t1b", "Blood T1 value", units="s", type=float, default=TIME_SCALE * 1.65),
        ModelOption("artt", "Arterial bolus arrival time", units="s", clargs=("--batart",), type=float, default=None),
        ModelOption("arttsd", "Arterial bolus arrival time prior std.dev.", units="s", clargs=("--batartsd",), type=float, default=None),

        # Inference options 
        ModelOption("inferatt", "Infer ATT (default on for multi-time imaging)", type=bool, default=None),
        ModelOption("artonly", "Only infer arterial component not tissue", type=bool),
        ModelOption("inferart", "Infer arterial component", type=bool),
        ModelOption("infert1", "Infer T1 value", type=bool),
        ModelOption("att_init", "Initialization method for ATT (max=max signal - bolus duration)", default=""),
        ModelOption("pvcorr", "Perform PVEc (shortcut for incwm, inferwm)", default=False),
        ModelOption("inferwm", "Infer WM parameters", default=False),

    ]

    def __init__(self, data_model, **options):
        Model.__init__(self, data_model, **options)
        if self.plds is not None:
            self.plds = tf.constant(self.plds, dtype=TF_DTYPE)
            self.tis = self.plds + self.tau

        if self.tis is None and self.plds is None:
            raise ValueError("Either TIs or PLDs must be given")

        # Only infer ATT with multi-time data 
        if self.inferatt is None: 
            self.inferatt = (len(self.tis) > 1)

        if (not self.att_init) and self.inferatt: 
            self.att_init = 'max'

        if self.attsd is None:
            self.attsd = (TIME_SCALE * 0.3) if len(self.tis) > 1 else (TIME_SCALE * 0.1)
        if self.artt is None:
            self.artt = self.att - (TIME_SCALE * 0.4)
        if self.arttsd is None:
            self.arttsd = self.attsd

        # Repeats are supposed to be a list but can be a single number
        if isinstance(self.repeats, (int, np.integer)):
            self.repeats = [self.repeats]

        # For now we only support fixed repeats
        if len(self.repeats) == 1:
            # FIXME variable repeats
            self.repeats = self.repeats[0]
        elif len(self.repeats) > 1 and \
            any([ r != self.repeats[0] for r in self.repeats ]):
            raise NotImplementedError("Variable repeats for TIs/PLDs")

        if self.pvcorr: 
            self.incwm = True
            self.inferwm = True 

        # If no pc provided, default depends on inclusion of WM or not. 
        # Also depends what mode we're in - PVEc is implied in surface/hybrid
        # mode, so set a 'pure GM' value 
        if self.pc is None: 
            if self.pvcorr: 
                self.pc = 0.98
            else: 
                self.pc = 0.9

        if self.artonly:
            self.inferart = True

        if not self.artonly:
            self.attach_param("ftiss", dist="FoldedNormal", edge_scale=True,
                            mean=1.5, prior_var=1e6, post_var=1e2, 
                            post_init=self._init_flow, **options)

            if self.inferatt: 
                self.attach_param("delttiss", dist="FoldedNormal", edge_scale=False,
                                mean=self.att, var=self.attsd**2,
                                post_init=self._init_delt, **options)

        if self.infert1:
            raise NotImplementedError() 
            self.attach_param("t1", mean=self.t1, var=0.01, edge_scale=False, **options)


        if self.inferart:
            raise NotImplementedError()
            self.leadscale = 0.01
            self.attach_param("fblood", dist="FoldedNormal", edge_scale=True,
                              mean=0.0, prior_var=1e6, post_var=1.5,
                              post_init=self._init_fblood,
                              prior_type="A", **options)

            if self.inferatt:
                self.attach_param("deltblood", dist="FoldedNormal", edge_scale=False,
                                mean=self.artt, var=self.arttsd**2,
                                post_init=self._init_delt,
                                **options)

        # Dict used to map anatomical structures to non-inferred
        # tissue properties that are relevant to model evaluation. 
        # Each structure has a 'tissue' attribute that will key 
        # into this dict 
        self.tissue_properties = {
            'GM' : { 'delt': self.att, 'pc': self.pc, 
                    'fcalib':  self.fcalib, 't1': self.t1 }, 
            'WM' : { 'delt': self.attwm, 'pc': self.pcwm, 
                    'fcalib': self.fcalibwm, 't1': self.t1wm }, 
            'mixed' : { 'delt': self.att, 'pc': self.pc, 
                     'fcalib':  self.fcalib, 't1': self.t1 }
        }

    def __str__(self):
        return f"ASL resting state model version {__version__}"

    def evaluate(self, params: dict[str, tf.Tensor], tpts: tf.Tensor) -> tf.Tensor:
        """
        Basic PASL/pCASL kinetic model
        """
        super().evaluate(params, tpts)

        if not self.artonly:
            signal = []

            for struct in self.data_model.structures: 
                slc = struct.slicer
                tp = copy.deepcopy(self.tissue_properties[struct.tissue])

                f = params['ftiss'][slc]
                d = tp.pop('delt')
                if 'delttiss' in params: 
                    d = params['delttiss'][slc]
                elif np.size(d) > 1: 
                    d = d[slc]

                t_1 = tp.pop('t1')
                if 't1' in params: 
                    t_1 = params['t1'][slc]
                elif np.size(t_1) > 1: 
                    t_1 = t_1[slc]

                s = self.tissue_signal(tpts[slc], f, d, t_1, **tp)
                signal.append(s)

            signal = tf.concat(signal, axis=0)
        else:
            signal = tf.zeros(tf.shape(tpts), dtype=TF_DTYPE)

        # if self.inferart:
        #     # FIMXE: is this going to work in surface/hybrid mode?
        #     signal += self.art_signal(tpts, fblood, deltblood)

        return signal

    def tissue_signal(self, t: tf.Tensor, ftiss: tf.Tensor, delt: tf.Tensor,
                      t1: tf.Tensor, pc: float, fcalib: float) -> tf.Tensor:
        """
        PASL/pCASL kinetic model for tissue

        :param t: timepoints for model evaluation 
        :param ftiss: CBF 
        :param delt: ATT
        :param t1: T1 decay constant
        :param pc: partition coefficient 
        :param fcalib: FIXME: some calibration coefficient... 
        """

        # TF is fussy about broadcasting so inputs need to be expanded 
        ndim = len(tf.shape(t))
        if isinstance(ftiss, (tf.Tensor, np.ndarray)):
            while len(tf.shape(ftiss)) < ndim: ftiss = tf.expand_dims(ftiss, -1)
        if isinstance(delt, (tf.Tensor, np.ndarray)):
            while len(tf.shape(delt)) < ndim: delt = tf.expand_dims(delt, -1)
        if isinstance(t1, (tf.Tensor, np.ndarray)): 
            while len(tf.shape(t1)) < ndim: t1 = tf.expand_dims(t1, -1)

        # Boolean masks indicating which voxel-timepoints are during the
        # bolus arrival and which are after
        post_bolus = tf.greater(t, self.tau + delt)
        during_bolus = tf.logical_and(tf.greater(t, delt), tf.logical_not(post_bolus))

        # Rate constants
        t1_app = TIME_SCALE / (1 / (t1 / TIME_SCALE) + (fcalib / pc))

        # Calculate signal
        if self.casl:
            # CASL kinetic model
            factor = 2 * t1_app * tf.exp(-delt / self.t1b) / TIME_SCALE
            during_bolus_signal = factor * (1 - tf.exp(-(t - delt) / t1_app))
            post_bolus_signal = factor * tf.exp(-(t - self.tau - delt) / t1_app) * (1 - tf.exp(-self.tau / t1_app))
        else:
            # PASL kinetic model
            r = 1 / t1_app - 1 / self.t1b
            f = 2 * tf.exp(-t / t1_app)
            factor = f / r
            during_bolus_signal = factor * ((tf.exp(r * t) - tf.exp(r * delt)))
            post_bolus_signal = factor * ((tf.exp(r * (delt + self.tau)) - tf.exp(r * delt)))

        # Build the signal from the during and post bolus components leaving as zero
        # where neither applies (i.e. pre bolus)
        signal = tf.zeros(tf.shape(during_bolus_signal), dtype=TF_DTYPE)
        signal = tf.where(during_bolus, during_bolus_signal, signal)
        signal = tf.where(post_bolus, post_bolus_signal, signal)

        return ftiss * signal

    def art_signal(self, t, fblood, deltblood):
        """
        PASL/pCASL Kinetic model for arterial curve
        
        To avoid problems with the discontinuous gradient at ti=deltblood
        and ti=deltblood+taub, we smooth the transition at these points
        using a Gaussian convolved step function. The sigma value could
        be exposed as a parameter (small value = less smoothing). This is
        similar to the effect of Gaussian dispersion, but can be computed
        without numerical integration
        """
        if self.casl:
            kcblood = 2 * tf.exp(-deltblood / self.t1b)
        else:
            kcblood = 2 * tf.exp(-t / self.t1b)

        # Boolean masks indicating which voxel-timepoints are in the leadin phase
        # and which in the leadout
        leadout = tf.greater(t, tf.add(deltblood, self.tau/2))
        leadin = tf.logical_not(leadout)

        # If deltblood is smaller than the lead in scale, we could 'lose' some
        # of the bolus, so reduce degree of lead in as deltblood -> 0. We
        # don't really need it in this case anyway since there will be no
        # gradient discontinuity
        leadscale = tf.minimum(deltblood, self.leadscale)
        leadin = tf.logical_and(leadin, tf.greater(leadscale, 0))

        # Calculate lead-in and lead-out signals
        leadin_signal = kcblood * 0.5 * (1 + tf.math.erf((t - deltblood) / leadscale))
        leadout_signal = kcblood * 0.5 * (1 + tf.math.erf(-(t - deltblood - self.tau) / self.leadscale))

        # Form final signal from combination of lead in and lead out signals
        signal = tf.zeros(tf.shape(leadin_signal))
        signal = tf.where(leadin, leadin_signal, signal)
        signal = tf.where(leadout, leadout_signal, signal)

        return fblood*signal

    def tpts(self) -> tf.Tensor:
        """
        Generate dense tensor of per-node timepoint values 

        :return tensor of size [W,T]
        """
        if self.data_model.n_tpts != len(self.tis) * self.repeats:
            raise ValueError("ASL model configured with %i time points, but data has %i" 
                             % (len(self.tis)*self.repeats, self.data_model.n_tpts))

        # FIXME assuming grouped by TIs/PLDs
        # Generate timings volume using the slicedt value
        t = np.zeros(list(self.data_model.shape) + [self.data_model.n_tpts], dtype=NP_DTYPE)
        for z in range(self.data_model.shape[2]):
            t[:, :, z, :] = np.array(sum([[ti + z*self.slicedt] * self.repeats for ti in self.tis], []))

        # Discard voxels not in the mask 
        t = t[self.data_model.vol_mask > 0]

        # Time points derived from volumetric data need to be transformed
        # into node space. Potential for a sneaky bug here so ensure the 
        # range of transformed values is consistent with the voxelwise input 
        tn = self.data_model.voxels_to_nodes(t, edge_scale=False)
        if not (np.allclose(np.min(t), np.min(tn), atol=1e-2)
            and np.allclose(np.max(t), np.max(tn), atol=1e-2)): 
                raise ValueError("Node-wise model tpts contains values "
                                 "outside the range of voxel-wise tpts")

        return tn 

    @Model.parameter_initialiser
    def _init_flow(self, data: tf.Tensor) -> tf.Tensor:
        """
        Initial value for the flow parameter

        :param data: voxel-wise data of shape [V,T]

        :return tensor of size [W]
        """

        # Non PVEc CBF init is just the maximum value over time 
        f_vox = tf.math.maximum(tf.reduce_mean(data, -1), 1)

        # PVEc CBF init makes some educated guesses
        # In voxels that contain pure WM, the init is just 1.5 times f_vox 
        # In voxels with some GM, we scale up for missing PV 
        fwm = 1.5 * (1 - self.data_model.pvgm) * f_vox
        fgm = (1 - self.data_model.pvwm) * f_vox / tf.maximum(self.data_model.pvgm, 0.5)

        # Project both onto nodes 
        fwm_n = self.data_model.voxels_to_nodes(fwm, edge_scale=False)
        fgm_n = self.data_model.voxels_to_nodes(fgm, edge_scale=True)

        s = self.data_model.structures[0]
        fgm_n[s.slicer].numpy().mean() 

        # For each structure, select init value either from the GM or WM based on tissue type 
        f_init = [ fgm_n[s.slicer] if s.tissue in ['GM', 'mixed'] else fwm_n[s.slicer] 
                   for s in self.data_model.structures ]

        f_init = tf.concat(f_init, axis=0)
        return f_init, None

    @Model.parameter_initialiser
    def _init_fblood(self, data: tf.Tensor) -> tf.Tensor:
        """
        Initial value for the fblood parameter

        :param data: voxel-wise data of shape [V,T]

        :return tensor of size [W]
        """
        raise NotImplementedError()
        return tf.math.maximum(tf.reduce_max(data, axis=1), 0.1), None

    @Model.parameter_initialiser
    def _init_delt(self, data: tf.Tensor) -> tf.Tensor:
        """
        Initial value for the delttiss parameter

        :param data: voxel-wise data of shape [V,T]

        :return tensor of size [W]
        """
        if self.att_init == "max":

            t_vox = self.data_model.nodes_to_voxels(self.tpts(), edge_scale=False)
            max_idx = tf.math.argmax(data, axis=1)
            time_max = tf.gather(t_vox, max_idx, batch_dims=1)

            t_gm = time_max - self.tau
            t_wm = time_max - self.tau + (TIME_SCALE * 0.3)

            t_gm = (((1 - self.data_model.pvgm) * self.att) 
                      + (self.data_model.pvgm * t_gm))
            t_wm = (((1 - self.data_model.pvwm) * self.attwm) 
                      + (self.data_model.pvwm * t_wm))

            t_gm = self.data_model.as_volume(t_gm)
            t_wm = self.data_model.as_volume(t_wm)

            t_gm = ndimage.percentile_filter(t_gm, percentile=70, size=(3,3,1))
            t_wm = ndimage.percentile_filter(t_wm, percentile=80, size=(3,3,1))

            t_gm = t_gm[self.data_model.vol_mask]
            t_wm = t_wm[self.data_model.vol_mask]

            # Project both onto nodes 
            tgm_n = self.data_model.voxels_to_nodes(t_gm, edge_scale=False)
            twm_n = self.data_model.voxels_to_nodes(t_wm, edge_scale=False)

            # For each structure, select init value either from the GM or WM based on tissue type 
            t_init = [ tgm_n[s.slicer] if s.tissue in ['GM', 'mixed'] else twm_n[s.slicer] 
                       for s in self.data_model.structures ]

            att_init = tf.concat(t_init, axis=0)
            return att_init, self.attsd ** 2

        else: 
            t_init = []
            for struct in self.data_model.structures: 
                t = tf.ones(struct.n_nodes, dtype=TF_DTYPE)
                t *= self.tissue_properties[struct.tissue]['delt']
                t_init.append(t)
            return tf.concat(t_init, axis=0), self.attsd ** 2