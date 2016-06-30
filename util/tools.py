#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import scipy
import numpy as np
import neurotools.nlab as nl
nl.set_cmap(nl.parula)
from matplotlib.pyplot import *
import warnings
warnings.filterwarnings('ignore')

datasets = {
    '/home/mrule/Workspace2/Hennig_data/Chip268_CNQX/HdfFilesSpkD45':
        ['Chip_268_Phase_00_basal_v28.hdf5',
        'Chip_268_Phase_01_CNQX_5uM_2h_v28.hdf5',
        'Chip_268_Phase_02_CNQX_5uM_20h_v28.hdf5',
        'Chip_268_Phase_03_CNQX_5uM_44h_v28.hdf5',
        'Chip_268_Phase_04_wash_2h_v28.hdf5',
        'Chip_268_Phase_05_wash_20h_v28.hdf5',
        'Chip_268_Phase_06_wash_44h_v28.hdf5'],
    '/home/mrule/Workspace2/Hennig_data/Chip264_CNQX/HdfFilesSpkD45/':
        ['Chip_264_Phase_00_basal_v28.hdf5',
        'Chip_264_Phase_01_CNQX_5uM_2h_v28.hdf5',
        'Chip_264_Phase_02_CNQX_5uM_20h_v28.hdf5',
        'Chip_264_Phase_03_CNQX_5uM_44h_v28.hdf5',
        'Chip_264_Phase_04_wash_2h_v28.hdf5',
        'Chip_264_Phase_05_wash_20h_v28.hdf5',
        'Chip_264_Phase_06_wash_44h_v28.hdf5']
}

epoch_names = [
    'Baseline',
    'CNQX 2h',
    'CNQX 20h',
    'CNQX 44h',
    'Post-CNQX 2h',
    'Post-CNQX 20h',
    'Post-CNQX 44h']

def remove_recalibration(Locations,Times,ROffsets):
    '''
    Remove recalibration events
    Consult Oliver about how/why this works
    '''
    keep      = np.abs(ROffsets-5)>5
    Locations = Locations[keep,...]
    Times     = Times[keep,...]
    print(Locations.shape,Times.shape)
    return Locations,Times

def get_locations(f):
    Locations = nl.getVariable(f,'Locations') # Locations (x,y)
    Times     = nl.getVariable(f,'Times')     # Spikes times
    ROffsets  = nl.getVariable(f,'RecalibrationOffsets')
    Locations,Times = remove_recalibration(Locations,Times,ROffsets)
    x,y = Locations.T
    return x,y,Times

def preview_spatial(f):
    Locations = nl.getVariable(f,'Locations') # Locations (x,y)
    Fs        = nl.getVariable(f,'Sampling')  # Sample rate Hz
    Times     = nl.getVariable(f,'Times')     # Spikes times
    nFrames   = nl.getVariable(f,'nFrames')   # Frames ~ tMax*Fs
    tMax      = nl.getVariable(f,'tMax')      # Experiment duration s
    ROffsets  = nl.getVariable(f,'RecalibrationOffsets')
    # Recalibration events contaminate the spiking data
    Location,Times = remove_recalibration(Locations,Times,ROffsets)
    x,y = Locations.T
    SKIP = max(1,len(x)//20000)
    x,y = Locations.T
    scatter(x[::SKIP],y[::SKIP],color='k',s=0.01)
    xlim(0,64)
    ylim(0,64)
    nl.force_aspect()