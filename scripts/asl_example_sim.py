# Example fitting of biexponential model
#
# This example runs SVB on a set of instances of biexponential data
# (by default just 4, to make it possible to display the results
# graphically, but you can up this number and not plot the output
# if you want a bigger data set)
#
# Usage: python biexp_exam
import sys
import math

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from svb.main import run

model = "aslrest"
outdir = "asl_example_sim_out"

# Inference options
# Note for complete convergence should probably have epochs=500 but this takes a while
options = {
    "tau" : 1.8,
    "casl" : True,
    "plds" : [0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    "repeats" : [1],
    "learning_rate" : 0.01,
    "sample_size" : 10,
    "epochs" : 5000,
    "log_stream" : sys.stdout,
    "save_mean" : True,
    "save_var" : True,
    "save_param_history" : True,
    "save_cost" : True,
    "save_cost_history" : True,
    "save_model_fit" : True,
    "save_log" : True,
    "force_num_latent_loss" : True,
}

runtime, svb, training_history = run("sig.nii.gz", model, outdir, **options)

