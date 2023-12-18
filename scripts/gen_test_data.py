# import sys
# import os.path
# import subprocess

# import numpy as np
# import sklearn.metrics

# import nibabel as nib

# from ssvb import DataModel
# from svb_models_asl import AslRestModel, AslNNModel
# import logging

# logging.getLogger().setLevel(logging.INFO)

# TIS = [2.05, 2.3, 2.55, 2.8, 3.05, 3.3]
# NOISE_SD = 0.0
# CUBE_SIDE = 10

# OPTIONS = {
#     "model": "buxton",
#     # "inctiss" : True,
#     # "infertiss" : True,
#     # "incbat" : True,
#     # "inferbat" : True,
#     "lambda": 0.9,
#     "tau": 1.8,
#     "ti": tis,
#     "t1b": 1.6,
#     "prior-noise-stddev": 1,
#     "casl": True,
#     "repeats": 1,
#     "t1": 1.3,
# }

# # Generate test data
# n = CUBE_SIDE**3
# sig = np.zeros((1, len(tis)), dtype=np.float32)
# data_model = DataModel(sig)
# model = AslRestModel(data_model, tis=tis, **options)
# ftiss = np.random.uniform(1.0, 20.0, size=(n,))
# delttiss = np.random.uniform(0.6, 2.5, size=(n,))
# params = np.zeros((2, n, 1), dtype=np.float32)
# tpts = np.zeros((n, len(tis)), dtype=np.float32)
# tpts[..., :] = TIS
# params[0, :, 0] = ftiss
# params[1, :, 0] = delttiss
# sim_sig = np.random.normal(model.ievaluate(params, tpts), NOISE_SD)
# print("Generated %i instances of test data" % n)

# # Save test data to Nifti files
# nii = nib.Nifti1Image(ftiss.reshape((10, 10, 10)), None)
# nii.to_filename("ftiss.nii.gz")
# nii = nib.Nifti1Image(delttiss.reshape((10, 10, 10)), None)
# nii.to_filename("delttiss.nii.gz")
# nii = nib.Nifti1Image(sim_sig.reshape((10, 10, 10, -1)), None)
# nii.to_filename("sig.nii.gz")

# # Try throwing the ftiss/delttiss ground truth at the NN model and
# # see if it matches the ASLREST simulated signal. Note that this may
# # not give good accuracDataModel(y if there is noise
# data_model = sig)
# model = AslNNModel(data_model, tis=TIS, train_load="trained_data", **options)
# tpts = np.zeros((n, len(TIS)), dtype=np.float32)
# tpts[..., :] = TIS
# modelsig = model.ievaluate(
#     np.array([ftiss.reshape((n, 1)), delttiss.reshape((n, 1))]), tpts
# )
# for idx in range(len(TIS)):
#     accuracy = sklearn.metrics.r2_score(modelsig[..., idx], sig[..., idx])
#     print("accuracy %i: %.3f" % (idx, accuracy))

# # Save the NN model prediction to a Nifti file
# nii = nib.Nifti1Image(modelsig.reshape((10, 10, 10, -1)), None)
# nii.to_filename("modelsig.nii.gz")
