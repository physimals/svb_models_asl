import os.path as op
import shutil
import sys

sys.path.insert(0, op.join(op.dirname(__file__), ".."))
sys.path.insert(0, "/Users/thomaskirk/modules/svb_models_asl")

import numpy as np
import tensorflow as tf
import toblerone as tob
from ssvb import DataModel, SSVBFit, run_inference
from ssvb.scripts.ssvb_asl import get_default_tissue_properties

PROJ = tob.Projector.load(
    "/Users/thomaskirk/Modules/ssvb_module/experiments/brain_proj_3.h5"
)

RPT = 1
NOISE = 1e-6

GM_ATT = 1.3
GM_CBF = 60
WM_ATT = 1.6
WM_CBF = 20
SIM_PARAMS = {
    "WM": {"cbf": WM_CBF, "att": WM_ATT},
    **{f"{s}_cortex": {"cbf": GM_CBF, "att": GM_ATT} for s in PROJ.hemi_dict.keys()},
    **{
        rn.replace("-", "_"): {"cbf": GM_CBF, "att": GM_ATT}
        for idx, rn in enumerate(PROJ.roi_names)
    },
}

FIT_OPTIONS = {
    "debug": True,
    "display_step": 50,
    "learning_rate": 0.1,
    "epochs": -10,
    "lr_decay_rate": 0.9,
}


def get_data_model(plds):
    data = tf.zeros((*PROJ.spc.size, len(plds) * RPT))
    data_model = DataModel.hybrid_from_projector(data, PROJ)
    return data_model


def test_single_pld_cbf():
    plds = [1.8]

    for param in ["att", "cbf"]:
        data_model = get_data_model(plds)
        overrides = get_default_tissue_properties(data_model, "hybrid")

        # Fix either ATT or CBF to the ground truth value and infer the other
        for s in data_model.structures:
            if param == "cbf":
                overrides[s.name]["att"] = SIM_PARAMS[s.name]["att"]
            else:
                overrides[s.name]["cbf"] = SIM_PARAMS[s.name]["cbf"]

        data_model.set_fwd_model(
            "aslrest",
            overrides=overrides,
            plds=plds,
            repeats=RPT,
            infer_cbf=(param == "cbf"),
            infer_att=(param == "att"),
            prior_dist="M",
        )

        data = data_model.test_voxel_data(
            params=SIM_PARAMS,
            tpts=data_model.tpts(),
            noise_sd=NOISE,
            masked=False,
        )

        data_model.set_new_data(data)
        outdir = "test_single_pld_hybrid_asl"
        fit1 = SSVBFit(data_model)
        _, fit, _ = run_inference(
            fit1,
            outdir,
            **FIT_OPTIONS,
        )

        for s in fit.structures:
            m = s[param].post.mean().numpy()
            low, high = np.quantile(m, [0.1, 0.9])
            assert np.allclose([low, high], SIM_PARAMS[s.name][param], rtol=0.1)

        shutil.rmtree(outdir)


def test_multi_pld_cbf():
    plds = [0.5, 1.0, 1.5, 2.0]

    for param in ["att", "cbf"]:
        data_model = get_data_model(plds)
        overrides = get_default_tissue_properties(data_model, "hybrid")

        # Fix either ATT or CBF to the ground truth value and infer the other
        for s in data_model.structures:
            if param == "cbf":
                overrides[s.name]["att"] = SIM_PARAMS[s.name]["att"]
            else:
                overrides[s.name]["cbf"] = SIM_PARAMS[s.name]["cbf"]

        data_model.set_fwd_model(
            "aslrest",
            overrides=overrides,
            plds=plds,
            repeats=RPT,
            infer_cbf=(param == "cbf"),
            infer_att=(param == "att"),
            prior_dist="M",
        )

        data = data_model.test_voxel_data(
            params=SIM_PARAMS,
            tpts=data_model.tpts(),
            noise_sd=NOISE,
            masked=False,
        )

        data_model.set_new_data(data)
        outdir = "test_multi_pld_hybrid_asl"
        fit1 = SSVBFit(data_model)
        _, fit, _ = run_inference(
            fit1,
            outdir,
            **FIT_OPTIONS,
        )

        for s in fit.structures:
            m = s[param].post.mean().numpy()
            low, high = np.quantile(m, [0.1, 0.90])
            assert np.allclose([low, high], SIM_PARAMS[s.name][param], rtol=0.1)

        shutil.rmtree(outdir)


if __name__ == "__main__":
    test_single_pld_cbf()
