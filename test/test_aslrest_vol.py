import itertools
import os.path as op
import shutil
import sys

sys.path.insert(0, op.join(op.dirname(__file__), ".."))
sys.path.insert(0, "/Users/thomaskirk/modules/svb_models_asl")

import numpy as np
import regtricks as rt
import tensorflow as tf
from ssvb import DataModel, SSVBFit, run_inference
from ssvb.scripts.ssvb_asl import TISSUE_PROPERTIES

GM_ATT = 1.3
GM_CBF = 60
SIM_PARAMS = {"voxels": {"cbf": GM_CBF, "att": GM_ATT}}
NOISE = 1e-6
RPT = 1

TISSUE_PROPERTIES = TISSUE_PROPERTIES
SPC = rt.ImageSpace.create_axis_aligned([0, 0, 0], [5, 5, 5], [1, 1, 1])

FIT_OPTIONS = {
    "debug": True,
    "display_step": 50,
    "learning_rate": 0.1,
    "epochs": -5,
    "lr_decay_rate": 0.9,
}

PLDS = [[1.8], [0.5, 1.0, 1.5, 2.0]]
PARAMS = ["att", "cbf"]
PRIORS = ["M", "N"]
RTOL = 0.1


def test_single_parameter_without_pve():
    for plds, param, prior in itertools.product(PLDS, PARAMS, PRIORS):
        if param == "att":
            hardcode = {"cbf": GM_CBF}
        else:
            hardcode = {"att": GM_ATT}

        data = tf.zeros((*SPC.size, len(plds) * RPT))
        data_model = DataModel.default_volumetric(data)
        data_model.set_fwd_model(
            "aslrest",
            plds=plds,
            repeats=RPT,
            infer_cbf=(param == "cbf"),
            infer_att=(param == "att"),
            prior_dist=prior,
            **hardcode,
        )
        data = data_model.test_voxel_data(
            params=SIM_PARAMS,
            tpts=data_model.tpts(),
            noise_sd=NOISE,
            masked=False,
        )

        data_model.set_new_data(data)

        outdir = "vol_ssvb_asl"
        fit1 = SSVBFit(data_model)
        _, fit, _ = run_inference(
            fit1,
            outdir,
            batch_size=len(PLDS) * RPT,
            **FIT_OPTIONS,
        )
        shutil.rmtree(outdir)

        assert np.allclose(
            fit.structures[0][param].post.mean(), SIM_PARAMS["voxels"][param], rtol=RTOL
        )


def test_single_parameter_with_pve():
    for plds, param, prior in itertools.product(PLDS, PARAMS, PRIORS):
        if param == "att":
            hardcode = {"cbf": GM_CBF}
        else:
            hardcode = {"att": GM_ATT}

        data = tf.zeros((*SPC.size, len(plds) * RPT))
        pvgm = np.random.uniform(0.2, 0.8, size=SPC.size)
        data_model = DataModel.volumetric_pvec(data, {"voxels": pvgm})
        data_model.set_fwd_model(
            "aslrest",
            plds=plds,
            repeats=RPT,
            infer_cbf=(param == "cbf"),
            infer_att=(param == "att"),
            prior_dist=prior,
            **hardcode,
        )
        data = data_model.test_voxel_data(
            params=SIM_PARAMS,
            tpts=data_model.tpts(),
            noise_sd=NOISE,
            masked=False,
        )

        data_model.set_new_data(data)

        outdir = "vol_ssvb_asl"
        fit1 = SSVBFit(data_model)
        _, fit, _ = run_inference(
            fit1,
            outdir,
            batch_size=len(PLDS) * RPT,
            **FIT_OPTIONS,
        )
        shutil.rmtree(outdir)

        m = fit.structures[0][param].post.mean().numpy()
        low, high = np.quantile(m, [0.05, 0.95])
        assert np.allclose([low, high], SIM_PARAMS["voxels"][param], rtol=RTOL)


if __name__ == "__main__":
    test_single_parameter_without_pve()
