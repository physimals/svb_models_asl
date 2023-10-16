"""Inference forward models for ASL data"""

import tensorflow as tf
import numpy as np
from scipy import ndimage
import tensorflow_probability as tfp

from ssvb.model import Model, ModelOption
from ssvb.utils import ValueList, NP_DTYPE, TF_DTYPE
from ssvb.structure import Cortex

from svb_models_asl import __version__

TIME_SCALE = tf.cast(1e0, TF_DTYPE)


class AslRestModel(Model):
    """
    ASL resting state model
    """

    OPTIONS = [
        # ASL parameters
        ModelOption(
            "tau",
            "Bolus duration",
            units="s",
            clargs=("--tau", "--bolus"),
            type=float,
            default=1.8 * TIME_SCALE,
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
        # Tissue properties
        ModelOption(
            "t1", "Tissue T1 value", units="s", type=float, default=1.3 * TIME_SCALE
        ),
        ModelOption(
            "att",
            "Bolus arrival time",
            units="s",
            clargs=("--bat",),
            type=float,
            default=1.3 * TIME_SCALE,
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
        ModelOption(
            "t1b", "Blood T1 value", units="s", type=float, default=1.65 * TIME_SCALE
        ),
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
            default="",
        ),
        ModelOption("infer_t1", "Infer T1 value", type=bool, default=None),
    ]

    def __init__(self, structure, **options):
        Model.__init__(self, structure, **options)

        # Default tissue CBF for the case where we are not inferring it.
        # This will almost always be overriden later on
        self.cbf = options.get("cbf", 0.0)
        self.artcbf = options.get("artcbf", 0.0)

        # TIs calculated as PLD + bolus duration
        if self.plds is not None:
            self.plds = tf.constant(self.plds, dtype=TF_DTYPE) * TIME_SCALE
            self.tis = self.plds + self.tau
        else:
            self.tis = tf.constant(self.tis, dtype=TF_DTYPE) * TIME_SCALE
            self.plds = self.tis - self.tau

        if (self.tis is None) and (self.plds is None):
            raise ValueError("Either TIs or PLDs must be given")

        if (not self.att_init) and self.is_multidelay:
            self.att_init = "max"

        if self.attsd is None:
            self.attsd = 0.5 if len(self.tis) > 1 else 0.1
        self.attsd *= TIME_SCALE

        if self.pc is None:
            self.pc = 0.9

        self.leadscale = 0.01
        if self.artatt is None:
            self.artatt = self.att - 0.4
        self.artatt *= TIME_SCALE
        if self.artattsd is None:
            self.artattsd = self.attsd
        self.artattsd *= TIME_SCALE

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

        name = "cbf"
        if getattr(self, f"infer_{name}", options.get(f"infer_{name}", False)):
            defaults = dict(
                post_type="Normal",
                mean=1.5,
                post_var=0.5,
                prior_var=1e6,
                post_init=self._init_cbf,
            )
            self.attach_param(
                name,
                defaults,
                options,
            )

        # This is where we either set up a parameter (inference),
        # or set a scalar value used for model evaluation (not inferece)
        name = "att"
        if getattr(self, f"infer_{name}", options.get(f"infer_{name}", False)):
            defaults = dict(
                post_type="LogNormal",
                mean=self.att,
                var=self.attsd**2,
                post_init=self._init_att,
            )
            self.attach_param(name, defaults, options)

        name = "t1"
        if getattr(self, f"infer_{name}", options.get(f"infer_{name}", False)):
            defaults = dict(mean=self.t1, var=0.01)
            self.attatch_param("t1", defaults, options)

        name = "artcbf"
        if getattr(self, f"infer_{name}", options.get(f"infer_{name}", False)):
            defaults = dict(
                prior_type="ARD",
                post_type="Normal",
                mean=0,
                prior_var=1e6,
                post_var=0.5,
                post_init=self._init_artcbf,
            )
            self.attach_param(name, defaults, options)

        name = "artatt"
        if getattr(self, f"infer_{name}", options.get(f"infer_{name}", False)):
            defaults = dict(
                post_type="Normal",
                mean=self.artatt,
                var=self.artattsd**2,
                post_init=self._init_att,
            )
            self.attach_param(name, defaults, options)

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

    def evaluate(self, params: dict[str, tf.Tensor], tpts: tf.Tensor) -> tf.Tensor:
        """
        PASL/pCASL kinetic model for tissue

        :param t: timepoints for model evaluation
        :param cbf: CBF
        :param att: ATT
        :param t1: T1 decay constant
        :param pc: partition coefficient
        :param fcalib: calibration coefficient
        """

        Model.evaluate(self, params, tpts)
            
        # Default parameters that we will override with caller's
        # All need to have shape [S,N] (add singleton on front)
        eval_params = {
            "cbf": tf.reshape(self.cbf, [1, -1]),
            "att": tf.reshape(self.att, [1, -1]),
            "t1": tf.reshape(self.t1, [1, -1]),
            "artcbf": tf.reshape(self.artcbf, [1, -1]),
            "artatt": tf.reshape(self.artatt, [1, -1]),
        }

        # Merge in the caller's params
        # Add batch dimension on back: [S,N] -> [S,N,1]
        eval_params.update(params)
        eval_params = {k: v[..., None] for k, v in eval_params.items()}

        # Add sample dimension on front: [N,T] -> [1,N,T]
        tpts = tf.expand_dims(tpts, 0)

        signal = self.tissue_signal(
            eval_params["cbf"], eval_params["att"], eval_params["t1"], tpts
        )
        if ("artcbf" in params) or ("artatt" in params): 
            signal += self.arterial_signal(
                eval_params["artcbf"], eval_params["artatt"], tpts
            )

        return signal 

    def tissue_signal(self, cbf, att, t1, tpts: tf.Tensor) -> tf.Tensor:
        pc = self.pc
        fcalib = self.fcalib

        # Boolean masks indicating which voxel-timepoints are during the
        # bolus arrival and which are after
        post_bolus = tpts > (self.tau + att)
        during_bolus = (tpts > att) & tf.logical_not(post_bolus)

        # Rate constants
        t1_app = TIME_SCALE / (1 / (t1 / TIME_SCALE) + (fcalib / pc))
        t1_app = tf.cast(t1_app, TF_DTYPE)

        # Calculate signal
        if self.casl:
            # CASL kinetic model
            factor = (
                (2 * t1_app) * tf.cast(tf.exp(-att / self.t1b), TF_DTYPE) / TIME_SCALE
            )
            during_bolus_signal = factor * (1 - tf.exp(-(tpts - att) / t1_app))
            post_bolus_signal = (
                factor
                * tf.exp(-(tpts - self.tau - att) / t1_app)
                * (1 - tf.exp(-self.tau / t1_app))
            )
        else:
            # PASL kinetic model
            r = 1 / t1_app - 1 / self.t1b
            f = 2 * tf.exp(-tpts / t1_app)
            factor = f / r
            during_bolus_signal = factor * ((tf.exp(r * tpts) - tf.exp(r * att)))
            post_bolus_signal = factor * (
                (tf.exp(r * (att + self.tau)) - tf.exp(r * att))
            )

        # Build the signal from the during and post bolus components leaving as zero
        # where neither applies (i.e. pre bolus)
        signal = tf.zeros(tf.shape(during_bolus_signal), dtype=TF_DTYPE)
        signal = tf.where(during_bolus, during_bolus_signal, signal)
        signal = tf.where(post_bolus, post_bolus_signal, signal)

        return cbf * signal

    def arterial_signal(self, artcbf, artatt, tpts: tf.Tensor) -> tf.Tensor:
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
        leadout = tpts > (artatt + self.tau / 2)
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
            * (1 + tf.math.erf(-(tpts - artatt - self.tau) / self.leadscale))
        )

        # Form final signal from combination of lead in and lead out signals
        signal = tf.zeros(tf.shape(leadin_signal), dtype=TF_DTYPE)
        signal = tf.where(leadin, leadin_signal, signal)
        signal = tf.where(leadout, leadout_signal, signal)

        return artcbf * signal

    # TODO: allow model to accept a tivol directly for fitting in
    # non-ASL space where standard slicedt correction doesn't work
    def tpts_vol(self) -> tf.Tensor:
        """
        Generate dense tensor of per-node timepoint values

        :return tensor of size [W,T]
        """
        if self.data_model.n_tpts != len(self.tis) * self.repeats:
            raise ValueError(
                "ASL model configured with %i time points, but data has %i"
                % (len(self.tis) * self.repeats, self.data_model.n_tpts)
            )

        # FIXME assuming grouped by TIs/PLDs
        tis_repeated = tf.repeat(self.tis, self.repeats)
        t = tf.broadcast_to(
            tis_repeated, [*self.data_model.vol_shape, len(tis_repeated)]
        )

        # Discard voxels not in the mask
        t = t[self.data_model.vol_mask > 0]
        return t

    def _init_cbf(self) -> tf.Tensor:
        """
        Initial value for the flow parameter

        :param data: voxel-wise data of shape [V,T]

        :return tensor of size [W]
        """

        data = self.data_model.data_flattened
        data_mean = tf.math.reduce_mean(data, axis=-1)

        if isinstance(self.structure, Cortex):
            ones = np.ones(self.structure.n_nodes, NP_DTYPE)
            pv = self.structure.to_voxels(ones)
            pv = np.maximum(pv, 0.25)
            data_mean = data_mean / pv

        f_vox = tf.math.maximum(data_mean, 0.1)
        f_init = self.structure.to_nodes(f_vox)
        return f_init, f_init / 10 

    def _init_att(self) -> tf.Tensor:
        """
        Initial value for the att parameter

        :param data: voxel-wise data of shape [V,T]

        :return tensor of size [W]
        """

        ones = np.ones(self.structure.n_nodes, NP_DTYPE)
        pv = self.structure.to_voxels(ones)

        data = self.data_model.data_flattened
        max_idx = tf.math.argmax(data, axis=1)
        time_max = tf.gather(self.tpts_vol(), max_idx, batch_dims=1)

        att_vox = (pv * (time_max - self.tau)) + ((1 - pv) * self.att)
        att_vox = self.data_model.as_volume(att_vox)
        att_vox = ndimage.percentile_filter(att_vox, percentile=70, size=(3, 3, 1))
        att_vox = att_vox.flatten()[self.data_model.vol_mask.flatten()]

        att_init = self.structure.to_nodes(att_vox)
        att_init = tf.maximum(att_init, np.percentile(att_init, 33))
        if np.allclose(np.min(att_init), 0):
            raise ValueError("calculated initial att value of 0")

        return att_init, att_init / 10 

    def _init_artcbf(self) -> tf.Tensor:
        """
        Initial value for the artcbf parameter

        :param data: voxel-wise data of shape [V,T]

        :return tensor of size [W]
        """
        data = self.data_model.data_flattened
        dmax = tf.reduce_mean(data, axis=1)
        thr = np.percentile(dmax, 90)
        init = tf.where(dmax > thr, 10, 1)
        return init, None

    def data_normaliser(self, data) -> TF_DTYPE:
        mn = np.mean(data)
        pow_ten = np.floor(np.log10(mn)) - 1
        return tf.cast(10**pow_ten, TF_DTYPE)
