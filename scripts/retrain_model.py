import logging

import numpy as np

from svb import DataModel

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

logging.getLogger().setLevel(logging.INFO)
sig = np.zeros((1, 6), dtype=np.float32)
data_model = DataModel(sig)
model = AslNNModel(data_model, tis=tis, 
                   train_examples=500, train_steps=200, train_lr=0.05, train_batch_size=1000,
                   train_delttiss_max=3.0,
                   train_save="trained_data", **options)
