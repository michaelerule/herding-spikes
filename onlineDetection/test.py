#import pyximport
#pyximport.install()

from detect import detect
import h5py
import numpy as np

# raw data file
rawpath = 'data/'
rawfile = rawpath+'P29_16_05_14_retina02_left_stim2_smallarray_fullfield_raw3'
print(rawfile)

sampling = 23199.0903585

# run detection
detect(rawfile, sampling)