"""Inference forward models for ASL data"""

import warnings

import tensorflow as tf
import numpy as np

from svb.model import Model, ModelOption
from svb.utils import ValueList, NP_DTYPE, TF_DTYPE
from svb.parameter import get_parameter

from svb_models_asl import __version__

class AslRestModel(Model):
    """
    ASL resting state model

    FIXME integrate with oxasl AslImage class?
    """

    OPTIONS = [

        # ASL parameters 
        ModelOption("tau", "Bolus duration", units="s", clargs=("--tau", "--bolus"), type=float, default=1.8),
        ModelOption("casl", "Data is CASL/pCASL", type=bool, default=False),
        ModelOption("tis", "Inversion times", units="s", type=ValueList(float)),
        ModelOption("plds", "Post-labelling delays (for CASL instead of TIs)", units="s", type=ValueList(float)),
        ModelOption("repeats", "Number of repeats - single value or one per TI/PLD", units="s", type=ValueList(int), default=[1]),
        ModelOption("slicedt", "Increase in TI/PLD per slice", units="s", type=float, default=0),

        # GM tissue properties 
        ModelOption("t1", "Tissue T1 value", units="s", type=float, default=1.3),
        ModelOption("att", "Bolus arrival time", units="s", clargs=("--bat",), type=float, default=1.3),
        ModelOption("attsd", "Bolus arrival time prior std.dev.", units="s", clargs=("--batsd",), type=float, default=None),
        ModelOption("fcalib", "Perfusion value to use in estimation of effective T1", type=float, default=0.01),
        ModelOption("pc", "Blood/tissue partition coefficient. If only inferring on one tissue, default is 0.9; if inferring on both GM/WM default is 0.98/0.8 respectively. See --pcwm", type=float, default=None),

        # WM tissue properties 
        ModelOption("incwm", "Include WM parameters", default=False),
        ModelOption("fwm", "WM perfusion", type=float, default=0),
        ModelOption("attwm", "WM arterial transit time", clargs=("--batwm",), type=float, default=1.6),
        ModelOption("t1wm", "WM T1 value", units="s", type=float, default=1.1),
        ModelOption("pcwm", "WM parition coefficient. See --pc", type=float, default=0.8),
        ModelOption("fcalibwm", "WM perfusion value to use in estimation of effective T1", type=float, default=0.003),

        # Blood / arterial properties 
        ModelOption("t1b", "Blood T1 value", units="s", type=float, default=1.65),
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

        # PVE options 
        ModelOption("pvgm", "GM partial volume", type=float, default=1.0),
        ModelOption("pvwm", "WM partial volume", type=float, default=0.0),

    ]

    def __init__(self, data_model, **options):
        Model.__init__(self, data_model, **options)
        if self.plds is not None:
            self.plds = np.array(self.plds, dtype=NP_DTYPE)
            self.tis = self.plds + self.tau

        if self.tis is None and self.plds is None:
            raise ValueError("Either TIs or PLDs must be given")

        # Only infer ATT with multi-time data 
        if self.inferatt is None: 
            self.inferatt = (len(self.tis) > 1)
        else: 
            if not isinstance(self.inferatt, bool): 
                raise ValueError("inferatt argument must be bool")

        if (not self.att_init) and self.inferatt: 
            self.att_init = 'max'

        if self.attsd is None:
            self.attsd = 1.5 if len(self.tis) > 1 else 0.1
        if self.artt is None:
            self.artt = self.att - 0.3
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

            # Ensure PVs match data size, whether provided as an array, 
            # single scalar, or path to file. 
            try: 
                self.pvgm = data_model._get_data(self.pvgm)[1].flatten()
                self.pvwm = data_model._get_data(self.pvwm)[1].flatten()
                if self.pvgm.size == self.data_model.mask_flattened.size: 
                    self.pvgm = self.pvgm[self.data_model.mask_flattened]
                    self.pvwm = self.pvwm[self.data_model.mask_flattened]
            except:
                if not isinstance(self.pvgm, (int,float,np.integer,np.floating)): 
                    raise ValueError("Could not interpret PV estimates")
                self.pvgm = np.asanyarray(self.pvgm)
                self.pvwm = np.asanyarray(self.pvwm)
                if self.pvgm.size == 1: 
                    warnings.warn("Single single scalar value for pvgm provided,"
                                  " this will be broadcast across all voxels.")

        if self.incwm and (np.array(self.pvgm + self.pvwm) > 1).any():
            raise ValueError("At least one GM and WM PV sum to > 1")

        # In surface/hybrid mode, PVE are accounted for in the projection matrix,
        # so we hardcode both tissue PVs to unity here. 
        if not self.data_model.is_volumetric:
            self.pvgm = 1.0 
            self.pvwm = 1.0 

        # If no pc provided, default depends on inclusion of WM or not. 
        # Also depends what mode we're in - PVEc is implied in surface/hybrid
        # mode, so set a 'pure GM' value 
        if self.pc is None: 
            if self.incwm or (not self.data_model.is_volumetric): 
                self.pc = 0.98
            else: 
                self.pc = 0.9

        if self.artonly:
            self.inferart = True

        # This is used for casting up various values to full-sized vectors, 
        # which ensures shape compatability in tf 
        ones = np.ones(self.data_model.n_nodes, dtype=NP_DTYPE)

        if not self.artonly:

            # Set up the PC,T1,PV tensors that correspond with the parameter
            # tensors in a node-wise manner. The default case below sets up for 
            # volumetric mode without PVEc, which just passes-through the T1, 
            # PC and GM PV values (NB GM PV defaults to 1 in non-PVEc mode). 
            t1_full = self.t1 * ones
            pc_full = self.pc * ones
            pvgm_full = self.pvgm * ones
            fcalib_full = self.fcalib * ones
            att_full = self.att * ones 
            
            # In surface/hybrid mode, we concatenate all nodes of different tissue types
            # into a single tensor. The data model knows the mapping between node numbers
            # and corresponding tissue type, so we use that to write in the correct tissue
            # properties. 
            if not self.data_model.is_volumetric:
                properties = { 'GM': (self.att, self.t1, self.pc, self.pvgm, self.fcalib), 
                               'WM': (self.attwm, self.t1wm, self.pcwm, self.pvwm, self.fcalibwm) }
                for node_slice, tiss in self.data_model.node_labels: 
                    att, t1, pc, pv, fc = properties[tiss]
                    att_full[node_slice] = att
                    t1_full[node_slice] = t1
                    pc_full[node_slice] = pc
                    pvgm_full[node_slice] = pv
                    fcalib_full[node_slice] = fc

            # Overwrite back onto the self object. 
            self.t1 = t1_full
            self.pc = pc_full
            self.pvgm = pvgm_full
            self.fcalib = fcalib_full
            self.att = att_full

            # NB order is important here - we must initialise these parameters 
            # after we have done all the hybrid-specific initialisation above, 
            # to ensure we are getting the correct initial values for them. 
            self.params = [
                get_parameter("ftiss", dist="NormalDist", 
                            mean=1.5, prior_var=1e6, post_var=1e3, 
                            post_init=self._init_flow, data_model=data_model,
                            **options)
            ]
            if self.inferatt: 
                self.params.append(
                    get_parameter("delttiss", dist="FoldedNormalDist", 
                                mean=self.att, var=self.attsd**2,
                                post_init=self._init_delt, data_model=data_model,
                                **options)
                    )


            if self.inferwm: 

                # Volumetric PVEc mode: we carry 2 complete sets of full-size tensors
                # around, one set for GM (set up above), this set for WM 
                self.t1wm = self.t1wm * ones
                self.pcwm = self.pcwm * ones
                self.pvwm = self.pvwm * ones
                self.fcalibwm = self.fcalibwm * ones
                self.attwm *= ones 

                self.params.append(
                    get_parameter("fwm", dist="NormalDist", 
                            mean=0.5, prior_var=1e6, post_var=10, 
                            post_init=self._init_flow, data_model=data_model,
                            **options)
                )

                if self.inferatt:
                    self.params.append(
                        get_parameter("deltwm", dist="FoldedNormalDist", 
                                mean=self.attwm, var=self.attsd**2,
                                post_init=self._init_delt, data_model=data_model,
                                **options)
                    )


        if self.infert1:
            self.params.append(
                get_parameter("t1", mean=self.t1, var=0.01, **options)
            )

            if self.inferwm:
                self.params.append(
                    get_parameter("t1wm", mean=self.t1wm, var=0.01, **options)
                )

        if self.inferart:
            self.leadscale = 0.01
            self.params.append(
                get_parameter("fblood", dist="NormalDist",
                              mean=0.0, prior_var=1e6, post_var=1.5,
                              post_init=self._init_fblood, data_model=data_model,
                              prior_type="A",
                              **options)
            )
            if self.inferatt:
                self.params.append(
                    get_parameter("deltblood", dist="FoldedNormalDist", 
                                mean=self.artt, var=self.arttsd**2,
                                post_init=self._init_delt, data_model=data_model,
                                **options)
                )

    def evaluate(self, params, tpts):
        """
        Basic PASL/pCASL kinetic model

        :param t: Time values tensor of shape [W, 1, N] or [1, 1, N]
        :param params Sequence of parameter values arrays, one for each parameter.
                      Each array is [W, S, 1] tensor where W is the number of nodes and
                      S the number of samples. This
                      may be supplied as a [P, W, S, 1] tensor where P is the number of
                      parameters.

        :return: [W, S, N] tensor containing model output at the specified time values
                 and for each time value using the specified parameter values
        """

        n_params = len(params) if isinstance(params, list) else params.get_shape().as_list()[0]
        if n_params != len(self.params):
            raise ValueError(f"Model set up to infer {len(self.params)} parameters; "
                "this many parameter arrays must be supplied")

        # Extract parameter tensors
        t = tf.identity(tpts, name="tpts")
        param_idx = 0
        if not self.artonly:
            ftiss = tf.identity(params[param_idx], name="ftiss")
            param_idx += 1

            if self.inferatt:
                delt = tf.identity(params[param_idx], name="delt")
                param_idx += 1
            else: 
                delt = self.att 
        
            if self.inferwm:
                fwm = tf.identity(params[param_idx], name="fwm")
                param_idx += 1
                
                if self.inferatt:
                    deltwm = tf.identity(params[param_idx], name="deltwm")
                    param_idx += 1   
                else: 
                    deltwm = self.attwm 

            else: 
                fwm = self.fwm
                deltwm = self.attwm              
    
        if self.infert1:
            t1 = tf.identity(params[param_idx], name="t1")
            param_idx += 1
            if self.inferwm: 
                t1wm = tf.identity(params[param_idx], name="t1wm")
                param_idx += 1 

        else:
            t1 = self.t1
            t1wm = self.t1wm

        if self.inferart:
            fblood = tf.identity(params[param_idx], name="fblood", shape=True, force=False)
            deltblood = tf.identity(params[param_idx+1], name="deltblood", shape=True, force=False)
            param_idx += 2

        # Extra parameters may be required by subclasses, e.g. dispersion parameters
        # FIXME: will need to separate out WM extra params and GM extra params... 
        extra_params = params[param_idx:]

        # In non-PVEc volumetric mode, all signal is assumed to come from 'GM' (even
        # though we know the real tissue type will actually be mixed).
        # In surface/hybrid mode, nodes for all tissues are contained within the same
        # tensors. In either case, we only need to evaluate the first block to get
        # the tissue signal 
        if not self.artonly:
            signal = tf.identity(self.tissue_signal(t, ftiss, delt, t1, self.pc, 
                        self.fcalib, self.pvgm, extra_params), name="tiss_signal")

            # The only case where we explicitly add a WM contribution is when doing
            # PVEc in volumetric mode (WM has already been accounted for in the above
            # block in surface/hybrid mode)
            if (self.data_model.is_volumetric) and (self.incwm): 
                wmsignal = tf.identity(self.tissue_signal(t, fwm, deltwm, t1wm, self.pcwm, 
                                        self.fcalibwm, self.pvwm, extra_params),
                                        name="wm_tiss_signal")
                signal += wmsignal

        else:
            signal = tf.zeros(tf.shape(t), dtype=tf.float32)

        if self.inferart:
            # FIMXE: is this going to work in surface/hybrid mode?
            signal += tf.identity(self.art_signal(t, fblood, deltblood, extra_params), name="art_signal")

        return tf.identity(signal, name="asl_signal")

    def tissue_signal(self, t, ftiss, delt, t1, pc, fcalib, pv=1.0, extra_params=[]):
        """
        PASL/pCASL kinetic model for tissue
        """

        # if (extra_params != []) and (extra_params.shape[0] > 0): 
        #     raise NotImplementedError("Extra tissue parameters not set up yet")

        # If these variables are np arrays, they may be under-sized compared to t,
        # so expand them up (tensorflow is very fussy about broadcasting)
        ndim = max([len(t.shape), len(ftiss.shape)])
        # FIXME this can probs go due to upgrade to tf2 
        if isinstance(ftiss, np.ndarray): ftiss = self.expand_dims(ftiss, ndim)
        if isinstance(delt, np.ndarray): delt = self.expand_dims(delt, ndim)
        if isinstance(t1, np.ndarray): t1 = self.expand_dims(t1, ndim)
        if isinstance(pc, np.ndarray): pc = self.expand_dims(pc, ndim)
        if isinstance(fcalib, np.ndarray): fcalib = self.expand_dims(fcalib, ndim)
        if isinstance(pv, np.ndarray): pv = self.expand_dims(pv, ndim)

        # Boolean masks indicating which voxel-timepoints are during the
        # bolus arrival and which are after
        # delt = (flag) * delt + (1-flag) * tf.stop_gradient(delt)
        post_bolus = tf.greater(t, self.tau + delt)
        during_bolus = tf.logical_and(tf.greater(t, delt), tf.logical_not(post_bolus))

        # Rate constants
        t1_app = 1 / (1 / t1 + fcalib / pc)

        # Calculate signal
        if self.casl:
            # CASL kinetic model
            factor = 2 * t1_app * tf.exp(-delt / self.t1b)
            during_bolus_signal = factor * (1 - tf.exp(-(t - delt) / t1_app))
            post_bolus_signal = factor * tf.exp(-(t - self.tau - delt) / t1_app) * (1 - tf.exp(-self.tau / t1_app))
        else:
            # PASL kinetic model
            r = 1 / t1_app - 1 / self.t1b
            f = 2 * tf.exp(-t / t1_app)
            factor = f / r
            during_bolus_signal = factor * ((tf.exp(r * t) - tf.exp(r * delt)))
            post_bolus_signal = factor * ((tf.exp(r * (delt + self.tau)) - tf.exp(r * delt)))

        post_bolus_signal = tf.identity(post_bolus_signal, name="post_bolus_signal")
        during_bolus_signal = tf.identity(during_bolus_signal, name="during_bolus_signal")

        # Build the signal from the during and post bolus components leaving as zero
        # where neither applies (i.e. pre bolus)
        signal = tf.zeros(tf.shape(during_bolus_signal), dtype=TF_DTYPE)
        signal = tf.where(during_bolus, during_bolus_signal, signal)
        signal = tf.where(post_bolus, post_bolus_signal, signal)

        return pv * ftiss * signal

    def art_signal(self, t, fblood, deltblood, extra_params):
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

    def tpts(self):
        if self.data_model.n_tpts != len(self.tis) * self.repeats:
            raise ValueError("ASL model configured with %i time points, but data has %i" % (len(self.tis)*self.repeats, self.data_model.n_tpts))

        # FIXME assuming grouped by TIs/PLDs
        # FIXME could define tpts() as a tf.constant to avoid calculating from scratch?
        # Generate timings volume using the slicedt value
        t = np.zeros(list(self.data_model.shape) + [self.data_model.n_tpts], dtype=NP_DTYPE)
        for z in range(self.data_model.shape[2]):
            t[:, :, z, :] = np.array(sum([[ti + z*self.slicedt] * self.repeats for ti in self.tis], []))

        # Discard voxels not in the mask 
        t = t[self.data_model.mask_vol > 0]

        # Time points derived from volumetric data need to be transformed
        # into node space.
        if not self.data_model.is_volumetric:
            t = tf.reshape(t, (-1, 1, self.data_model.n_tpts))
            tn = self.data_model.voxels_to_nodes_ts(t, edge_scale=False)

            # This was a sneaky bug - the projection to nodes was reducing
            # TIs below their original values due to PVE. Sanity check here
            # to ensure this doesn't happen. 
            low = np.min(t) - np.min(tn) > 1e-3
            high = np.max(tn) - np.max(t) > 1e-3
            if low or high: 
                raise ValueError("Node-wise model tpts contains values outside the range of voxel-wise tpts")
            t = tn 

        return tf.reshape(t, (-1, self.data_model.n_tpts))

    def __str__(self):
        return "ASL resting state model: %s" % __version__

    def _init_flow(self, _param, _t, data):
        """
        Initial value for the flow parameter
        """

        # f = tf.math.maximum(data.mean(-1).astype(NP_DTYPE), 0.1)

        if self.data_model.is_volumetric: 
            f = data.mean(-1)
            fgm = f / np.maximum(self.pvgm, 0.5)
            fwm = 1.5 * (1 - self.pvgm) * f
            if not self.pvcorr:
                return f, None
            else: 
                # Intialisation for volumetric PVEc: assume a CBF ratio of 3:1
                if _param.name == 'fwm':
                    return fwm, None
                else: 
                    return fgm, None 

        elif self.data_model.is_hybrid: 
            mask = self.data_model.mask_flattened
            pvgm, pvwm = self.data_model.projector.pvs().reshape(-1,3)[mask,:2].T
            # data = data / np.maximum((pvgm + pvwm), 0.5)[:,None]

            f = data.mean(-1)
            fgm = f / np.maximum(pvgm, 0.5)
            fwm = 1.3 * (1 - pvgm) * f

            fwm_nodes = np.squeeze(self.data_model.voxels_to_nodes(fwm.astype(np.float32), edge_scale=False))
            fgm_nodes = np.squeeze(self.data_model.voxels_to_nodes(fgm.astype(np.float32), edge_scale=False))

            # WARNING: we are assuming all subcortical ROIs are GM here... 
            f_hybrid = tf.concat([
                fgm_nodes[self.data_model.surf_slicer], 
                fwm_nodes[self.data_model.vol_slicer], 
                fgm_nodes[self.data_model.subcortical_slicer]], axis=0)

            return f_hybrid, None 

        else:
            f_surf = tf.squeeze(self.data_model.voxels_to_nodes(f[:,None], edge_scale=False))
            return f_surf, None 



    def _init_fblood(self, _param, _t, data):
        """
        Initial value for the fblood parameter
        """
        return tf.math.maximum(tf.reduce_max(data, axis=1), 0.1), None


    def _init_delt(self, _param, t_node, data):
        """
        Initial value for the delttiss parameter
        """
        if self.att_init == "max":
            t_vox = self.data_model.nodes_to_voxels(t_node, edge_scale=False)
            max_idx = tf.math.argmax(data, axis=1)
            time_max = tf.gather(t_vox, max_idx, batch_dims=1)

            if self.data_model.is_volumetric: 

                if _param.name == 'deltwm': 
                    return (time_max + 0.3 - self.tau, 
                            self.attsd ** 2)
                else: 
                    return (time_max - self.tau, 
                            self.attsd ** 2)

            elif self.data_model.is_pure_surface: 
                raise NotImplementedError() 

            else:         
                time_max = tf.squeeze(self.data_model.voxels_to_nodes(time_max[:,None], edge_scale=False))

                att = tf.concat([
                    time_max[self.data_model.surf_slicer] - self.tau, 
                    time_max[self.data_model.vol_slicer] + 0.3 - self.tau, 
                    time_max[self.data_model.subcortical_slicer] - self.tau], axis=0)

                return att, self.attsd ** 2

        # FIXME the below is out of date 
        elif self.data_model.is_volumetric:
            if _param.name == 'fwm': 
                return self.attwm, self.attsd ** 2
            else: 
                return self.att, self.attsd ** 2
        elif self.data_model.is_hybrid: 
            return self.att, self.attsd ** 2
        else: 
            return _param.prior.dist.ext_mean, _param.prior.dist.ext_var

    def expand_dims(self, array, ndim):
        while array.ndim < ndim: 
            array = array[...,None]
        return array 

    def ensure_fullsize(self, value):
        # Ensure PVs match data size, whether provided as an array
        # single scalar, or path to file. 
        if isinstance(value, tf.Tensor):
            # do the same set of casts here
            raise RuntimeError("cast the tf object")

        if isinstance(value, (float,int)):
            value = np.array(value) 

        if isinstance(value, np.ndarray):
            if value.size == self.data_model.mask_flattened.size:
                value = value[self.data_model.mask_flattened]

        ones = np.ones(self.data_model.n_nodes)
        value = (ones * value).astype(NP_DTYPE)
        return value 
        