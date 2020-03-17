import sys
import os.path
import subprocess

import numpy as np

import nibabel as nib
from fabber import Fabber

from svb import DataModel
from svb_models_asl import AslRestModel, AslNNModel
import logging
logging.getLogger().setLevel(logging.INFO)

tis = [2.05, 2.3, 2.55, 2.8, 3.05, 3.3]
n = 1000

options = {
    "model" : "buxton",
    #"inctiss" : True,
    #"infertiss" : True,
    #"incbat" : True,
    #"inferbat" : True,
    'lambda': 0.9,
    'tau' : 1.8,
    'ti' : tis, 
    't1b': 1.6,
    "prior-noise-stddev" : 1,
    'casl': True,
    'repeats': 1,
    't1': 1.3
}

def get_sample_data(tis, n=1000, **kwargs):
    """
    Generate training data by evaluating Fabber model
    """
    fab = Fabber()
   
    options.update(**kwargs)

    params = {}
    n_ti = len(tis)
    fit_data = np.zeros((n, 2 + n_ti), dtype=np.float32)
    for idx in range(n):
        ftiss = np.random.uniform(4.0, 5.0)
        delttiss = np.random.uniform(1.2, 1.4)
        params["ftiss"] = ftiss
        params['delttiss'] = delttiss
        output = fab.model_evaluate(options, params, nvols=6)
        #output = np.random.normal(output, 0.1)
        fit_data[idx,0] = ftiss
        fit_data[idx,1] = delttiss
        fit_data[idx,2:(3 + n_ti)] = np.array((output))
    print("Generated %i instances of test data" % n)

    return fit_data[:, 0], fit_data[:, 1], fit_data[:, 2:]

def get_sample_data_svb(tis, n=1000, **kwargs):
    sig = np.zeros((1, len(tis)), dtype=np.float32)
    data_model = DataModel(sig)
    model = AslRestModel(data_model, tis=tis, **options)
    params = np.zeros((2, n, 1), dtype=np.float32)
    tpts = np.zeros((n, len(tis)), dtype=np.float32)
    tpts[..., :] = tis
    params[0, :, 0] = np.random.uniform(4.0, 5.0, size=(n,))
    params[1, :, 0] = np.random.uniform(1.2, 1.4, size=(n,))
    modelsig = model.ievaluate(params, tpts)
    print("Generated %i instances of test data" % n)

    return params[0, :, 0], params[1, :, 0], modelsig

ftiss, delttiss, sig = get_sample_data_svb(tis=tis, n=n)
nii = nib.Nifti1Image(ftiss.reshape((10, 10, 10)), None)
nii.to_filename("ftiss.nii.gz")
nii = nib.Nifti1Image(delttiss.reshape((10, 10, 10)), None)
nii.to_filename("delttiss.nii.gz")
nii = nib.Nifti1Image(sig.reshape((10, 10, 10, -1)), None)
nii.to_filename("sig.nii.gz")

data_model = DataModel(sig)
model = AslNNModel(data_model, tis=tis, train_load="trained_data", **options)
tpts = np.zeros((n, len(tis)), dtype=np.float32)
tpts[..., :] = tis
modelsig = model.ievaluate(np.array([ftiss.reshape((n, 1)), delttiss.reshape((n, 1))]), tpts)
from sklearn.metrics import r2_score
for idx in range(len(tis)):
    accuracy = r2_score(modelsig[..., idx], sig[..., idx])
    print('accuracy %i: %.3f' % (idx, accuracy))

nii = nib.Nifti1Image(modelsig.reshape((10, 10, 10, -1)), None)
nii.to_filename("modelsig.nii.gz")
