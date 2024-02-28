"""Inference forward models for ASL data"""

import nibabel as nib
import numpy as np
import tensorflow as tf
from scipy import stats
from ssvb.model import Model, ModelOption
from ssvb.structure import Cortex, Volumetric
from ssvb.utils import NP_DTYPE, TF_DTYPE, ValueList

from svb_models_asl import __version__


class AslRestModel(Model):
    """
    ASL resting state model
    """

    OPTIONS = [
        # ASL parameters
        ModelOption(
            "taus",
            "Bolus duration",
            units="s",
            clargs=("--taus", "--tau", "--bolus"),
            type=ValueList(float),
            default=[1.8],
        ),
        ModelOption("casl", "Data is CASL/pCASL", type=bool, default=True),
        ModelOption("tis", "Inversion times", units="s", type=ValueList(float)),
        ModelOption(
            "plds",
            "Post-labelling delays (for CASL instead of TIs)",
            units="s",
            type=ValueList(float),
        ),
        ModelOption(
            "repeats",
            "Number of repeats - single value or one per TI/PLD",
            units="s",
            type=ValueList(int),
            default=[1],
        ),
        ModelOption(
            "slicedt",
            "Increase in TI/PLD per slice in Z direction",
            units="s",
            type=float,
            default=0,
        ),
        ModelOption(
            "slicedt_img",
            "3D image giving timing offset at each voxel in acquisition data",
            units="s",
            type=str,
            default=None,
        ),
        # Tissue properties
        ModelOption("t1", "Tissue T1 value", units="s", type=float, default=1.3),
        ModelOption(
            "att",
            "Bolus arrival time",
            units="s",
            clargs=("--bat",),
            type=float,
            default=1.3,
        ),
        ModelOption(
            "attsd",
            "Bolus arrival time prior std.dev.",
            units="s",
            clargs=("--batsd",),
            type=float,
            default=None,
        ),
        ModelOption(
            "fcalib",
            "Perfusion value to use in estimation of effective T1",
            type=float,
            default=0.01,
        ),
        ModelOption(
            "pc", "Blood/tissue partition coefficient.", type=float, default=0.9
        ),
        # Blood properties
        ModelOption("t1b", "Blood T1 value", units="s", type=float, default=1.65),
        # Arterial properties
        ModelOption("infer_artcbf", "Infer arterial CBF", type=bool, default=None),
        ModelOption("infer_artatt", "Infer arterial ATT", type=bool, default=None),
        ModelOption(
            "artatt", "Arterial bolus arrival time", units="s", type=float, default=None
        ),
        ModelOption(
            "artattsd",
            "Arterial bolus arrival time prior std.dev.",
            units="s",
            type=float,
            default=None,
        ),
        # Inference options
        ModelOption(
            "infer_att",
            "Infer ATT",
            type=bool,
            default=None,
        ),
        ModelOption(
            "att_init",
            "Initialization method for ATT (max=max signal - bolus duration)",
        ),
        ModelOption("infer_t1", "Infer T1 value", type=bool, default=None),
    ]

    def __init__(self, structure, **options):
        Model.__init__(self, structure, **options)
        # Default tissue CBF for the case where we are not inferring it.
        # This will almost always be overriden later on
        self.cbf = tf.cast(options.get("cbf", 0.0), TF_DTYPE)
        self.artcbf = tf.cast(options.get("artcbf", 0.0), TF_DTYPE)

        if self.tis is not None and self.plds is not None:
            raise ValueError("Cannot provide both PLDs and TIs")
        if self.tis is None and self.plds is None:
            raise ValueError("Either TIs or PLDs must be given")
        if self.plds is not None:
            num_times = len(self.plds)
        else:
            num_times = len(self.tis)

        # Bolus duration (tau) can be a list or a single number
        try:
            self.taus = [float(self.taus)]
        except (TypeError, ValueError):
            self.taus = list(self.taus)

        if len(self.taus) == 1:
            self.taus = self.taus * num_times
        elif len(self.taus) != num_times:
            raise ValueError("Number of bolus durations must match number of TIs/PLDs")

        if self.plds is not None:
            self.tis = np.array(self.taus) + np.array(self.plds)

        if self.attsd is None:
            self.attsd = 0.75 if len(self.tis) > 1 else 0.1
        if self.artatt is None:
            self.artatt = self.att - 0.4
        if self.artattsd is None:
            self.artattsd = self.attsd

        # Repeats can be a sequence or a single number
        try:
            self.repeats = [int(self.repeats)]
        except (ValueError, TypeError):
            self.repeats = list(self.repeats)
        if len(self.repeats) == 1:
            self.repeats = self.repeats * len(self.tis)
        elif len(self.repeats) != len(self.tis):
            raise ValueError("Number of repeats must match number of TIs/PLDs")

        # Apply repeats to bolus durations
        new_taus = []
        for tau, rpts in zip(self.taus, self.repeats):
            new_taus += [tau] * rpts
        self.taus = new_taus

        if (not self.att_init) and self.is_multidelay:
            self.att_init = "max"

        if self.pc is None:
            self.pc = 0.9

        self.leadscale = 0.01

        # Slice timing offsets
        if self.slicedt and self.slicedt_img is not None:
            raise ValueError("Cannot specify both slicedt and slicedt_img")
        if self.slicedt_img:
            if isinstance(self.slicedt_img, str):
                self.slicedt_img = nib.load(self.slicedt_img).get_fdata()

        name = "cbf"
        if options.get(f"infer_{name}", False):
            defaults = dict(
                post_dist="F",
                mean=1.5,
                post_var=1e3,
                prior_var=1e6,
                post_init=self._init_cbf,
            )
            self.attach_param(name, True, defaults, options)

        name = "att"
        if options.get(f"infer_{name}", False):
            defaults = dict(
                post_dist="F",
                mean=self.att,
                var=self.attsd**2,
                post_init=self._init_att,
            )
            self.attach_param(name, True, defaults, options)

        # name = "t1"
        # defaults = dict(mean=self.t1, var=0.01)
        # self.attach_param("t1", options.get(f"infer_{name}", False), defaults, options)

        # name = "artcbf"
        # defaults = dict(
        #     prior_dist="A",
        #     post_dist="F",
        #     mean=0,
        #     prior_var=1e6,
        #     post_var=0.5,
        #     post_init=self._init_artcbf,
        # )
        # self.attach_param(name, options.get(f"infer_{name}", False), defaults, options)

        # name = "artatt"
        # defaults = dict(
        #     post_dist="F",
        #     mean=self.artatt,
        #     var=self.artattsd**2,
        #     # post_init=self._init_att,
        # )
        # self.attach_param(name, options.get(f"infer_{name}", False), defaults, options)

    def __str__(self):
        return f"ASL resting state tissue model version {__version__}"

    @property
    def is_multidelay(self):
        return len(self.tis) > 1

    @property
    def is_singledelay(self):
        return not self.is_multitime

    @property
    def n_delays(self):
        return len(self.tis)

    # FIXME assumes constant repeats
    @property
    def n_tpts(self):
        return len(self.tis) * self.repeats

    def evaluate(self, params: dict[str, tf.Tensor], tpts: tf.Tensor) -> tf.Tensor:
        """
        PASL/pCASL kinetic model for tissue

        :param t: timepoints for model evaluation as integer volume indexes NOT actual time values
        :param cbf: CBF
        :param att: ATT
        :param t1: T1 decay constant
        :param pc: partition coefficient
        :param fcalib: calibration coefficient
        """

        # Call the base class evaluate method to check shapes
        Model.evaluate(self, params, tpts)

        # Default parameters that we will override with caller's
        # All need to have shape [S,N] (add singleton on front)
        default_params = {
            "cbf": tf.reshape(self.cbf, [1, -1]),
            "att": tf.reshape(self.att, [1, -1]),
            "t1": tf.reshape(self.t1, [1, -1]),
            "artcbf": tf.reshape(self.artcbf, [1, -1]),
            "artatt": tf.reshape(self.artatt, [1, -1]),
        }

        # Override defaults with caller's parameters
        default_params.update(params)

        # Extract time points and corresponding bolus durations
        tpt_indexes = tf.squeeze(tpts)  # [B]
        tpts_full = self.tpts_nodewise(self.structure.data_model)  # [W, N]
        tpts = tf.gather(tpts_full, tpt_indexes, axis=-1)  # [W/1, B]
        taus = tf.expand_dims(tf.gather(self.taus, tpt_indexes), axis=0)  # [1, B]

        # Reshape params and tpts for broadcasting
        default_params, tpts = self.reshape_for_evaluate(default_params, tpts)
        taus = tf.expand_dims(taus, axis=1)

        signal = self.tissue_signal(
            default_params["cbf"], default_params["att"], default_params["t1"], tpts, taus
        )
        if ("artcbf" in params) or ("artatt" in params):
            signal += self.arterial_signal(
                default_params["artcbf"], default_params["artatt"], tpts, taus
            )

        return signal

    def tissue_signal(self, cbf, att, t1, tpts: tf.Tensor, taus) -> tf.Tensor:
        pc = self.pc
        fcalib = self.fcalib

        # Boolean masks indicating which voxel-timepoints are during the
        # bolus arrival and which are after
        post_bolus = tpts > (taus + att)
        during_bolus = (tpts > att) & tf.logical_not(post_bolus)

        # Rate constants
        t1_app = 1 / (1 / t1 + fcalib / pc)

        # Calculate signal
        if self.casl:
            # CASL kinetic model
            factor = (2 * t1_app) * tf.cast(tf.exp(-att / self.t1b), TF_DTYPE)
            during_bolus_signal = factor * (1 - tf.exp(-(tpts - att) / t1_app))
            post_bolus_signal = (
                factor
                * tf.exp(-(tpts - taus - att) / t1_app)
                * (1 - tf.exp(-taus / t1_app))
            )
        else:
            # PASL kinetic model
            r = 1 / t1_app - 1 / self.t1b
            f = 2 * tf.exp(-tpts / t1_app)
            factor = f / r
            during_bolus_signal = factor * ((tf.exp(r * tpts) - tf.exp(r * att)))
            post_bolus_signal = factor * (
                (tf.exp(r * (att + taus)) - tf.exp(r * att))
            )

        # Build the signal from the during and post bolus components leaving as zero
        # where neither applies (i.e. pre bolus)
        # FIXME: small ramp in ATT possibly gives better stability
        # as it gives a continuous gradient when ATT < min PLD
        # signal = tf.linspace(0.0, 1e-2 * att[:, :, 0], tf.shape(tpts)[-1], axis=2)
        signal = tf.zeros(tf.shape(during_bolus_signal), dtype=TF_DTYPE)
        signal = tf.where(during_bolus, during_bolus_signal, signal)
        signal = tf.where(post_bolus, post_bolus_signal, signal)

        return cbf * signal

    def arterial_signal(self, artcbf, artatt, tpts: tf.Tensor, taus) -> tf.Tensor:
        """
        PASL/pCASL Kinetic model for arterial curve

        To avoid problems with the discontinuous gradient at ti=artatt
        and ti=artatt+taub, we smooth the transition at these points
        using a Gaussian convolved step function. The sigma value could
        be exposed as a parameter (small value = less smoothing). This is
        similar to the effect of Gaussian dispersion, but can be computed
        without numerical integration
        """

        if self.casl:
            kcblood = tf.cast(2 * tf.exp(-artatt / self.t1b), TF_DTYPE)
        else:
            kcblood = tf.cast(2 * tf.exp(-tpts / self.t1b), TF_DTYPE)

        # Boolean masks indicating which voxel-timepoints are in the leadin phase
        # and which in the leadout
        leadout = tpts > (artatt + taus / 2)
        leadin = tf.logical_not(leadout)

        # If artatt is smaller than the lead in scale, we could 'lose' some
        # of the bolus, so reduce degree of lead in as artatt -> 0. We
        # don't really need it in this case anyway since there will be no
        # gradient discontinuity
        leadscale = tf.cast(tf.minimum(artatt, self.leadscale), TF_DTYPE)
        leadin = tf.logical_and(leadin, leadscale > 0)

        # Calculate lead-in and lead-out signals
        leadin_signal = kcblood * 0.5 * (1 + tf.math.erf((tpts - artatt) / leadscale))
        leadout_signal = (
            kcblood
            * 0.5
            * (1 + tf.math.erf(-(tpts - artatt - taus) / self.leadscale))
        )

        # Form final signal from combination of lead in and lead out signals
        signal = tf.zeros(tf.shape(leadin_signal), dtype=TF_DTYPE)
        signal = tf.where(leadin, leadin_signal, signal)
        signal = tf.where(leadout, leadout_signal, signal)

        return artcbf * signal

    def tpts_nodewise(self, data_model) -> tf.Tensor:
        """Node-wise timepoints for the model. See :func:`~AslRestModel.tpts_vol`"""

        t = self.tpts_vol(data_model)

        # Time points derived from volumetric data need to be transformed
        # into node space. Potential for a sneaky bug here so ensure the
        # range of transformed values is consistent with the voxelwise input
        ts = self.structure.to_nodes(t)
        if not (
            np.allclose(np.min(t), np.min(ts), atol=1e-2)
            and np.allclose(np.max(t), np.max(ts), atol=1e-2)
        ):
            raise ValueError(
                "Node-wise model tpts contains values "
                "outside the range of voxel-wise tpts"
            )

        return ts

    def tpts_vol(self, data_model) -> tf.Tensor:
        """
        Generate dense tensor of per-node timepoint values

        :return tensor of size [W,T]
        """
        ntpts = data_model.n_tpts
        if ntpts != sum(self.repeats):
            raise ValueError(
                "ASL model configured with %i time points, but data has %i"
                % (sum(self.repeats), ntpts)
            )

        # Note that this assumes data grouped by TIs/PLDs which is required for variable repeats
        base_tpts = np.array(
            sum([[ti] * rpts for ti, rpts in zip(self.tis, self.repeats)], []),
            dtype=NP_DTYPE
        )
        acq_shape = list(data_model.data_vol.shape[:3])

        if self.slicedt_img is None:
            self.slicedt_img = np.zeros(acq_shape, dtype=NP_DTYPE)
            if self.slicedt is not None:
                # Generate timings volume using the slicedt value
                for z in range(acq_shape[2]):
                    self.slicedt_img[:, :, z] = z * self.slicedt
        else:
            shape = list(self.slicedt_img.shape)
            if len(shape) != 3:
                raise ValueError(f"Slice DT image must be 3D - has shape: {shape}")
            if shape != acq_shape:
                raise ValueError(
                    f"Slice DT image shape does not match acquired data: {shape} vs {acq_shape}"
                )

        t = (
            self.slicedt_img[..., None] + base_tpts[None, None, None, :]
        )  # [NX, NY, NZ, NT]

        # Apply mask
        t = t[data_model.vol_mask > 0]

        return t

    def _init_cbf(self, data_model) -> tf.Tensor:
        """
        Initial value for the flow parameter

        :param data: voxel-wise data of shape [V,T]

        :return tensor of size [W]
        """

        data = data_model.data_flattened
        data_filtered = tf.math.reduce_mean(data, axis=-1)

        pv = self.structure.to_voxels(tf.ones(self.structure.n_nodes))
        pv = tf.maximum(pv, 0.5)

        if isinstance(self.structure, Cortex):
            data_filtered = data_filtered / pv
        elif (type(self.structure) is Volumetric) and data_model.mode == "hybrid":
            data_filtered = data_filtered * pv

        f_vox = tf.math.maximum(data_filtered, 0.1)
        f_init = self.structure.to_nodes(f_vox)
        return f_init, f_init / 10

    def _init_att(self, data_model) -> tf.Tensor:
        """
        Initial value for the att parameter

        :param data: voxel-wise data of shape [V,T]

        :return tensor of size [W]
        """

        if not self.is_multidelay:
            return self.att, self.attsd**2

        tpts_vol = self.tpts_vol(data_model)
        max_idx = tf.math.argmax(data_model.data_flattened, axis=1)
        time_max = tf.gather(tpts_vol, max_idx, batch_dims=1)
        att = time_max - np.mean(self.taus)

        pv = self.structure.to_voxels(tf.ones(self.structure.n_nodes))
        pv_thr = np.percentile(pv, 75)
        att_mode = stats.mode(att[pv >= pv_thr], axis=None).mode

        return att_mode, self.attsd**2

    def _init_artcbf(self, data_model) -> tf.Tensor:
        """
        Initial value for the artcbf parameter

        :param data: voxel-wise data of shape [V,T]

        :return tensor of size [W]
        """
        data = data_model.data_flattened
        dmax = tf.reduce_mean(data, axis=1)
        thr = np.percentile(dmax, 90)
        init = tf.where(dmax > thr, 10, 1)
        return init, None
