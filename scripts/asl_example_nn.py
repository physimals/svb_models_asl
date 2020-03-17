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

model = "aslnn"
outdir = "asl_example_out_nn"

# Inference options
# Note for complete convergence should probably have epochs=500 but this takes a while
options = {
    "tau" : 1.8,
    "casl" : True,
    "plds" : [0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    "repeats" : [8],
    "slicedt" : 0.0452,
    "learning_rate" : 0.01,
    "batch_size" : 6,
    "sample_size" : 10,
    "epochs" : 500,
    "log_stream" : sys.stdout,
    "save_mean" : True,
    "save_var" : True,
    "save_param_history" : True,
    "save_cost" : True,
    "save_cost_history" : True,
    "save_model_fit" : True,
    "save_log" : True,
    "force_num_latent_loss" : True,
    "train_load" : "trained_data",
}

runtime, svb, training_history = run("asldata_diff.nii.gz", model, outdir, mask="asldata_mask.nii.gz", **options)

ftiss_img = nib.load("%s/mean_ftiss.nii.gz" % outdir).get_data()
delttiss_img = nib.load("%s/mean_delttiss.nii.gz" % outdir).get_data()

# Display a single slice (z=10)
plt.figure("F")
plt.imshow(ftiss_img[:, :, 10])
plt.figure("delt")
plt.imshow(delttiss_img[:, :, 10])
plt.show()
