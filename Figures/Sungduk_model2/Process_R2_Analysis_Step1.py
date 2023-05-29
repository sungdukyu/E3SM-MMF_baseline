#!/usr/bin/env python
# coding: utf-8

import json
import pickle
import csv
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob, os
import random

import tensorflow as tf
from tensorflow import keras

# in/out variable lists
vars_mli = ['state_t','state_q0001','state_ps','pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX']
vars_mlo = ['ptend_t','ptend_q0001','cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC','cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD']

# new dataset generator function
# that has new options (latlim, lonlim)

mli_mean = xr.open_dataset('./norm_factors/mli_mean.nc')
mli_min = xr.open_dataset('./norm_factors/mli_min.nc')
mli_max = xr.open_dataset('./norm_factors/mli_max.nc')
mlo_scale = xr.open_dataset('./norm_factors/mlo_scale.nc')
ne4_grid_info = xr.open_dataset('./test_data/E3SM-MMF_ne4_grid-info.orig.nc')

def load_nc_dir_with_generator_test(filelist:list, latlim=[-999,999], lonlim=[-999,999]):
    def gen():
        for file in filelist:
            
            # read mli
            ds = xr.open_dataset(file, engine='netcdf4')
            ds = ds[vars_mli]
            ds = ds.merge(ne4_grid_info[['lat','lon']])
            ds = ds.where((ds['lat']>latlim[0])*(ds['lat']<latlim[1]),drop=True)
            ds = ds.where((ds['lon']>lonlim[0])*(ds['lon']<lonlim[1]),drop=True)
            
            # read mlo
            dso = xr.open_dataset(file.replace('.mli.','.mlo.'), engine='netcdf4')
            dso = dso.merge(ne4_grid_info[['lat','lon']])
            dso = dso.where((dso['lat']>latlim[0])*(dso['lat']<latlim[1]),drop=True)
            dso = dso.where((dso['lon']>lonlim[0])*(dso['lon']<lonlim[1]),drop=True)
            
            # make mlo variales: ptend_t and ptend_q0001
            dso['ptend_t'] = (dso['state_t'] - ds['state_t'])/1200 # T tendency [K/s]
            dso['ptend_q0001'] = (dso['state_q0001'] - ds['state_q0001'])/1200 # Q tendency [kg/kg/s]
            dso = dso[vars_mlo]
            
            # normalizatoin, scaling
            ds = (ds-mli_mean)/(mli_max-mli_min)
            dso = dso*mlo_scale

            # stack
            #ds = ds.stack({'batch':{'sample','ncol'}})
            ds = ds.stack({'batch':{'ncol'}})
            ds = ds.to_stacked_array("mlvar", sample_dims=["batch"], name='mli')
            #dso = dso.stack({'batch':{'sample','ncol'}})
            dso = dso.stack({'batch':{'ncol'}})
            dso = dso.to_stacked_array("mlvar", sample_dims=["batch"], name='mlo')
            
            yield (ds.values, dso.values)

    return tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float64, tf.float64),
        output_shapes=((None,124),(None,128))
    )


lon = ne4_grid_info.lon.values
lat = ne4_grid_info.lat.values
area = ne4_grid_info.area.values
hyam = ne4_grid_info.hyam.values
hybm = ne4_grid_info.hybm.values
PS = ne4_grid_info.PS.values
P0 = ne4_grid_info.P0.values

# Convert to pressure
def convert_to_Pressure(hyam,hybm,PS,P0):
    Dimension1=hyam.shape
    Dimension2=PS.shape
    Pressure = np.zeros([Dimension1[0],Dimension2[1]])
    for i in range(Dimension2[1]):
        #temp = (P0*hyam[:] + PS[i]*hybm[:])
        #print(temp.shape)
        Pressure[:,i] = (hyam[:] + PS[0,i]*hybm[:])
    return Pressure

Pre = convert_to_Pressure(hyam,hybm,PS,P0)

# Estimate R2 based on Mooers et al. diagnostics
def estimate_R2(y_true,y_pred,ncol,nlev):
    Dimension=y_true.shape
    R2 = np.zeros([ncol,nlev])
    SSE = np.zeros([ncol,nlev])
    SVAR = np.zeros([ncol,nlev])
    for i in range(ncol):
        for z in range(nlev):
            y_true_temp = y_true[i::ncol,z]
            y_pred_temp = y_pred[i::ncol,z]
            y_true_temp.shape
            SSE[i,z] = np.sum((y_true_temp-y_pred_temp)**2.0)
            SVAR[i,z] = np.sum((y_true_temp-np.mean(y_true_temp))**2.0)
            R2[i,z] = 1-(SSE[i,z]/SVAR[i,z])

    return SSE,SVAR,R2

# Estimate R2 based on Mooers et al. diagnostics
def average_data(data,window_size):

    num_elements = data.shape[0]
    num_windows = num_elements // window_size

    # Reshape the data into windows of size 384 along the first dimension
    data_windows = np.reshape(data[:num_windows * window_size], (num_windows, window_size, *data.shape[1:]))
    #print(data_windows.shape)
    # Calculate the average along the first dimension
    averaged_data = np.mean(data_windows, axis=1)
    return averaged_data

# This function turns 20 minute increments into daily averages
# (384)*ndays*72 = Dim[0]
# Output dim: (384)*ndays
# Griffin's original version is based on regrided data
# This version is based on ne4pg2 grid (384 columns)
# This change allow me to use Sungduk's code directly
# since he generaged the min/max/scale already
def average_data_Griffin_LiranEdited(reconstructed_targets,reconstructed_features):
    Dim = reconstructed_targets.shape
    x = 384
    tnum = 72
    t = Dim[0]
    z = Dim[1]
    ndays = int(t/(tnum*x))

    target_days = np.zeros(shape=(x*ndays,tnum, z))
    feat_days = np.zeros(shape=(x*ndays,tnum, z))
    day_array_targ = np.zeros(shape=(x,tnum,ndays, z))
    day_array_feat = np.zeros(shape=(x,tnum,ndays, z))
    #print(day_array_feat.shape)
    ncol_count = 0
    tstep_count = 0
    day_count = 0

    for i in range(t):
        temp_targ = reconstructed_targets[i, :]
        day_array_targ[ncol_count,tstep_count,day_count, :] = temp_targ
        temp_feat = reconstructed_features[i, :]
        day_array_feat[ncol_count,tstep_count,day_count,:] = temp_feat

        if (ncol_count == x-1):
            ncol_count = 0
            tstep_count = tstep_count+1
        else:
            ncol_count = ncol_count+1

        if (tstep_count == tnum):
            tstep_count = 0
            day_count = day_count+1


    day_array_targ_out = np.nanmean(day_array_targ, axis = 1)
    day_array_feat_out = np.nanmean(day_array_feat, axis = 1)

    return day_array_targ_out,day_array_feat_out

# validation dataset for HPO
stride_sample = 5 # about ~11% assuming we will use 1/5 subsampled dataset for full training.
f_mli1 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0008-0[23456789]-*-*.nc')
f_mli2 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0008-1[012]-*-*.nc')
f_mli3 = glob.glob('/pscratch/sd/s/sungduk/hugging/E3SM-MMF_ne4/train/*/E3SM-MMF.mli.0009-01-*-*.nc')
f_mli_val = sorted([*f_mli1, *f_mli2, *f_mli3])
f_mli = f_mli_val
print(f'#files: {len(f_mli_val)}')

# creating numpy array defeats the purpose of tf Dataset pipeline,
# but, just doing it here for quick sanity check.
tds_test = load_nc_dir_with_generator_test(f_mli)
work = list(tds_test.as_numpy_iterator())
x_true = np.concatenate([ work[k][0] for k in range(len(work)) ])
y_true = np.concatenate([ work[k][1] for k in range(len(work)) ])

ncol = 384
nlev = 60

import os
import netCDF4 as nc
folder_path = "/pscratch/sd/h/heroplr/step2_retrain/backup_phase-4_retrained_models"  # Replace with the actual folder path
out_folder = "/pscratch/sd/h/heroplr/R2_analysis_all/"
Dim_true = x_true.shape
# Loop through all files in the folder
numday = int(Dim_true[0]/ncol/72)
filename="fffffff"
file_path = os.path.join(folder_path, filename)

if os.path.isfile(file_path):
        # Perform operations on the file
        print(file_path)  # Example: Print the file path
        model = keras.models.load_model(file_path,compile=False)
        y_pred = model(x_true)
        T_tend_true = y_true[:,:60]
        T_pred_true = y_pred[:,:60]
        Q_tend_true = y_true[:,60:120]
        Q_pred_true =y_pred[:,60:120]
        T_tend_true_avg = average_data(T_tend_true,len(f_mli))
        T_pred_true_avg = average_data(T_pred_true,len(f_mli))
        Q_tend_true_avg = average_data(Q_tend_true,len(f_mli))
        Q_pred_true_avg = average_data(Q_pred_true,len(f_mli))
        T_tend_true2,T_pred_true2 = average_data_Griffin_LiranEdited(T_tend_true,T_pred_true)
        Q_tend_true2,Q_pred_true2 = average_data_Griffin_LiranEdited(Q_tend_true,Q_pred_true)
        T_tend_true2_reshaped_array = T_tend_true2.reshape((384*numday, 60))
        T_pred_true2_reshaped_array = T_pred_true2.reshape((384*numday, 60))
        Q_tend_true2_reshaped_array = Q_tend_true2.reshape((384*numday, 60))
        Q_pred_true2_reshaped_array = Q_pred_true2.reshape((384*numday, 60))
        TSSE,TSVAR,TR_temp = estimate_R2(T_tend_true2_reshaped_array,T_pred_true2_reshaped_array,384*numday,nlev)
        QSSE,QSVAR,QR_temp = estimate_R2(Q_tend_true2_reshaped_array,Q_pred_true2_reshaped_array,384*numday,nlev)
        TR1 = TR_temp.reshape((384,numday, 60))
        QR1 = QR_temp.reshape((384,numday, 60))
        TR2 =  np.nanmean(TR1, axis = 1)
        QR2 =  np.nanmean(QR1, axis = 1)
        # Create a new NetCDF file
        filename = file_path[-31:]+".nc"
        file_path_out = os.path.join(out_folder, filename)

        ncfile = nc.Dataset(file_path_out, "w", format="NETCDF4")

        # Define the dimensions
        time_dim = ncfile.createDimension("time", None)  # Unlimited dimension
        lat_dim = ncfile.createDimension("ncol", ncol)
        lon_dim = ncfile.createDimension("nlev", nlev)
        day_dim = ncfile.createDimension("nday", numday)
        # Create variables
        time_var = ncfile.createVariable("time", "f8", ("time",))
        lon_var = ncfile.createVariable("lon", "f4", ("ncol",))
        lat_var = ncfile.createVariable("lat", "f4", ("ncol",))
        PRE_var = ncfile.createVariable("P", "f8", ("nlev","ncol"))
        data_var = ncfile.createVariable("TR2", "f8", ("nlev","ncol"))
        data_var2 = ncfile.createVariable("QR2", "f8", ("nlev","ncol"))
        data_var3 = ncfile.createVariable("T_tend_true_avg", "f8", ("nlev","nday","ncol"))
        data_var4 = ncfile.createVariable("T_pred_true_avg", "f8", ("nlev","nday","ncol"))
        data_var5 = ncfile.createVariable("Q_tend_true_avg", "f8", ("nlev","nday","ncol"))
        data_var6 = ncfile.createVariable("Q_pred_true_avg", "f8", ("nlev","nday","ncol"))

        # Assign values to variables
        time_var[:] = [1]  # Example time values
        lon_var[:] = lon    # Example latitude values
        lat_var[:] = lat  # Example longitude values
        PRE_var[:] = (Pre)
        data_var[:,:] = np.transpose(TR2 )              # Example data values
        data_var2[:,:] = np.transpose(QR2)

        data_var3[:,:,:] = np.transpose(T_tend_true2)
        data_var4[:,:,:] = np.transpose(T_pred_true2)
        data_var5[:,:,:] = np.transpose(Q_tend_true2)
        data_var6[:,:,:] = np.transpose(Q_pred_true2)
        # Add global attributes
        ncfile.description = "R2"
        ncfile.history = "Created by Liran"

        # Close the NetCDF file
        ncfile.close()








