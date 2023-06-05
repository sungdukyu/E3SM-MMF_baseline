#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:20:34 2023

@author: mohamedazizbhouri
"""

import numpy as np

import pickle
from matplotlib import pyplot as plt
plt.close('all')

plt.rcParams.update(plt.rcParamsDefault)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 32,
                     'lines.linewidth': 2,
                     'axes.labelsize': 32,
                     'axes.titlesize': 32,
                     'xtick.labelsize': 32,
                     'ytick.labelsize': 32,
                     'legend.fontsize': 32,
                     'axes.linewidth': 2,
                     "pgf.texsystem": "pdflatex"
                     })
######################################
############ User's input ############
######################################

# please provide your model_name. The latter will be used in saving the results
# files (npy and plots)
model_name = 'RPN'

# load model target values of shape Npoints x 128

##########################################
################## cVAE ##################
##########################################
#import h5py
#hf = h5py.File('/ocean/projects/atm200007p/mbhouri/neurips_data/cvae.h5', 'r')
#Datasetnames=hf.keys()
#n1 = hf.get('pred')
#pred_y = np.array(n1)

##########################################
################## RPN ###################
##########################################
pred_y = np.load('/ocean/projects/atm200007p/mbhouri/neurips_data/rpn_pred_v1_stride6.npy')

##########################################
################## MLP ###################
##########################################
#pred_y = np.load('/ocean/projects/atm200007p/sungduk/for_aziz/MLP_v1_ne4/001_backup_phase-7_retrained_models_step2_lot-147_trial_0027.best.h5.npy')

mlo_scale = np.array( np.load('/ocean/projects/atm200007p/mbhouri/neurips_data/mlo_scale.npy'), dtype=np.float32)
pred_y = pred_y / mlo_scale

######################################
############# Test steup #############
######################################


# Please specify the temporal stride used in generating validation/test dataset
# and original GCM time-step (dt_GCM) in minutes
stride = 6
dt_GCM = 20

# load true target values of shape Ndata x 128
test_y = np.load('/ocean/projects/atm200007p/jlin96/neurips_proj/e3sm_train_npy/val_target_stride6.npy')
test_y = test_y / mlo_scale

# file 'pressures_val_stride6_60lvls.npy' contains pressure levels in Pa
pressure = np.load('/ocean/projects/atm200007p/mbhouri/neurips_data/pressures_val_stride6_60lvls.npy')/100
N_pressure = pressure.shape[0] # Number of pressure levels: 60

# load dictionary of latitude-longitude coordinates of GCM grid on which data is saved
with open('/ocean/projects/atm200007p/mbhouri/neurips_data/indextolatlons.pkl', 'rb') as f:
    data = pickle.load(f)
N_lat_long = len(data) # = 384 There are 384 points in latitude-longitude GCM grid

import netCDF4
dset = netCDF4.Dataset('/ocean/projects/atm200007p/mbhouri/neurips_data/mass_weight_area_pressure.nc')

mweightpre = np.array(dset['mweightpre']).T # 60x384 ==> 384x60
area = np.array(dset['area'])[:,None] # 384 ==> 384x1
Lv = 2.26e6
cp = 1.00464e3
rho = 997

weight = np.concatenate( (cp*mweightpre,Lv*mweightpre,area,area,rho*Lv*area,rho*Lv*area,area,area,area,area), axis=1 )

# numpy return erronous errors when summing over "large" number of points with float32 format, 
# so we transform arrays to float64 type before computing coefficient of determination R2
pred_y = np.array(pred_y,dtype=np.float64)
test_y = np.array(test_y,dtype=np.float64)

dim_y = test_y.shape[1]

# Ndata is total number of data points in validation / test dataset
# N_time_steps is the number of time-steps considered in validation / test dataset
Ndata = test_y.shape[0]
N_time_steps = Ndata//N_lat_long

# To compute the coefficient of determination we need to compute daily averages
# given stride and dt_GCM specified above we compute how many time-steps correspond to a daily window
N_dt_day = int( (60*24)/(stride*dt_GCM) )

###########################################
# Global metrics for 128 output variables #
###########################################

def reshape_npy(y):
    # reshape true data into: N_time_steps x N_lat_long x N_pressure
    y = y.reshape( (N_time_steps, N_lat_long, dim_y) )
    # compute daily avereages
    # resulting array of shape: Nday x N_lat_long x dim_y, where Nday = N_time_steps//N_dt_day is the number of days
    return np.mean( y.reshape( (N_time_steps//N_dt_day, N_dt_day, N_lat_long, dim_y) ) , axis=1 )

test_daily = weight * reshape_npy(test_y)

test_daily = np.concatenate( ( np.sum(test_daily[:,:,:N_pressure],axis=(1,2))[:,None],
                               np.sum(test_daily[:,:,N_pressure:2*N_pressure],axis=(1,2))[:,None],
                               np.sum(test_daily[:,:,2*N_pressure:],axis=1)), axis=1  )

pred_daily = weight * reshape_npy(pred_y)

pred_daily = np.concatenate( ( np.sum(pred_daily[:,:,:N_pressure],axis=(1,2))[:,None],
                               np.sum(pred_daily[:,:,N_pressure:2*N_pressure],axis=(1,2))[:,None],
                               np.sum(pred_daily[:,:,2*N_pressure:],axis=1)), axis=1  )

MAE = np.mean(np.abs(test_daily-pred_daily),axis=0)
RMSE = np.sqrt(np.mean((test_daily-pred_daily)**2,axis=0))

np.save(model_name+'_MAE.npy',MAE)
np.save(model_name+'_RMSE.npy',RMSE)

R2 = 1 - np.sum( (pred_daily-test_daily)**2, axis=0)/np.sum( (test_daily-np.mean(test_daily, axis=0))**2, axis=0)
np.save(model_name+'_R2.npy',R2)
