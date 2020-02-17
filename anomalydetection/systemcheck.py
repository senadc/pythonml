import os
import scipy.io
import scipy.optimize
import numpy as np


dir_path = os.path.dirname(os.path.realpath(__file__))

# Load Training Data
print("Loading training data ...\n")
mat = scipy.io.loadmat(dir_path + '\\ex8data1.mat')
X = mat['X']
Xval = mat['Xval']
yval = mat['yval']

