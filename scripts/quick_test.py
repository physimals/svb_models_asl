import logging
import numpy as np

from svb import DataModel

from fabber import Fabber

from svb_models_asl import AslNNModel

tis = [2.05, 2.3, 2.55, 2.8, 3.05, 3.3]
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

params = {}
params['delttiss'] = 1.3
fab = Fabber()
for ftiss in (1, 5, 10):
    params["ftiss"] = float(ftiss)
    output = fab.model_evaluate(options, params, nvols=6)
    print(output)
    
logging.getLogger().setLevel(logging.INFO)
sig = np.zeros((1, 6), dtype=np.float32)
data_model = DataModel(sig)
model = AslNNModel(data_model, tis=tis, train_load="trained_data", **options)
tpts = np.zeros((1, 6), dtype=np.float32)
tpts[..., :] = tis

# 2 params, 3 voxels, 1 samples
params = np.zeros((2, 3, 1), dtype=np.float32)
params[0, 0, :] = 1.0
params[0, 1, :] = 5.0
params[0, 2, :] = 10.0
params[1, :, :] = 1.3
modelsig = model.ievaluate(params, tpts)
print(modelsig)