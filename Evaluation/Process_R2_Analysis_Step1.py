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
from scipy.ndimage import gaussian_filter
import tensorflow as tf
from tensorflow import keras
import cartopy
import cartopy.feature as cfeature
import cartopy.crs as ccrs
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
        Pressure[:,i] = (P0*hyam[:] + PS[0,i]*hybm[:])
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

# Estimate R2 based on Mooers et al. diagnostics
def estimate_R2_outv2(y_true,y_pred,nlev,nday,nlat):
    Dimension=y_true.shape
    R2 = np.zeros([nlev,nlat])
    SSE = np.zeros([nlev,nlat])
    SVAR = np.zeros([nlev,nlat])
    for i in range(nlat):
        for z in range(nlev):
            y_true_temp = y_true[z,:,i]
            y_pred_temp = y_pred[z,:,i]
            y_true_temp.shape
            SSE[z,i] = np.sum((y_true_temp-y_pred_temp)**2.0)
            SVAR[z,i] = np.sum((y_true_temp-np.mean(y_true_temp))**2.0)
            R2[z,i] = 1-(SSE[z,i]/SVAR[z,i])

    return SSE,SVAR,R2
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

# Estimate R2 based on Mooers et al. diagnostics
def estimate_R2_out_lonlat(y_true,y_pred,nlat,nlon):
    Dimension=y_true.shape
    R2 = np.zeros([nlat,nlon])
    SSE = np.zeros([nlat,nlon])
    SVAR = np.zeros([nlat,nlon])
    for j in range(nlat):
        for i in range(nlon):
            y_true_temp = y_true[:,j,i]
            y_pred_temp = y_pred[:,j,i]
            y_true_temp.shape
            SSE[j,i] = np.sum((y_true_temp-y_pred_temp)**2.0)
            SVAR[j,i] = np.sum((y_true_temp-np.mean(y_true_temp))**2.0)
            R2[j,i] = 1-(SSE[j,i]/SVAR[j,i])
        
    return SSE,SVAR,R2

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
        T_tend_true20,T_pred_true20 = average_data_Griffin_LiranEdited(T_tend_true,T_pred_true)
        Q_tend_true20,Q_pred_true20 = average_data_Griffin_LiranEdited(Q_tend_true,Q_pred_true)
        
        # Now reverse the normalization.
        T_tend_true2 = np.zeros([ncol,365,nlev])
        T_pred_true2 = np.zeros([ncol,365,nlev])
        Q_tend_true2 = np.zeros([ncol,365,nlev])
        Q_pred_true2 = np.zeros([ncol,365,nlev])
        for c in range(ncol):
            for d in range(365):
                T_tend_true2[c,d,:] = T_tend_true20[c,d,:]/(mlo_scale.ptend_t.values)*1200
                T_pred_true2[c,d,:] = T_pred_true20[c,d,:]/(mlo_scale.ptend_t.values)*1200
                Q_tend_true2[c,d,:] = Q_tend_true20[c,d,:]/(mlo_scale.ptend_q0001.values)*1200
                Q_pred_true2[c,d,:] = Q_pred_true20[c,d,:]/(mlo_scale.ptend_q0001.values)*1200
        
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
        
        from nco import Nco
        os.environ['NCOpath'] = '/global/homes/h/heroplr/.conda/envs/Sungduk/bin'
        nco = Nco()
        filename2 = file_path[-31:]+"aave.nc"
        file_path_outregrid = os.path.join(out_folder, filename2)
        nco.ncks(input=file_path_out, output=file_path_outregrid, map='map_ne4pg2_to_CERES1x1_aave.20230516.nc')
        
    
        import os
        import netCDF4 as nc
        folder_path = "/pscratch/sd/h/heroplr/R2_analysis_all"  # Replace with the actual folder path
        out_folder = "/pscratch/sd/h/heroplr/R2_analysis_allFigures"

        TR2_out_file = xr.open_dataset(file_path_outregrid)
        TR_2 = TR2_out_file.TR2.values
        QR_2 = TR2_out_file.QR2.values
        P_2 = TR2_out_file.P.values
        lon_C = TR2_out_file.lon.values
        lat_C = TR2_out_file.lat.values
        T_tend_true_avg_2 = TR2_out_file.T_tend_true_avg.values
        T_pred_true_avg_2 = TR2_out_file.T_pred_true_avg.values
        Q_tend_true_avg_2 = TR2_out_file.Q_tend_true_avg.values
        Q_pred_true_avg_2 = TR2_out_file.Q_pred_true_avg.values
        TR2_reshape_lat = np.nanmean(TR_2,axis=2)
        QR2_reshape_lat = np.mean(QR_2,axis=2)
        T_tend_true_avg_2_reshape_lat = np.nanmean(T_tend_true_avg_2,axis=3)
        T_pred_true_avg_2_reshape_lat = np.mean(T_pred_true_avg_2,axis=3)
        Q_tend_true_avg_2_reshape_lat = np.nanmean(Q_tend_true_avg_2,axis=3)
        Q_pred_true_avg_2_reshape_lat = np.mean(Q_pred_true_avg_2,axis=3)
        P2_reshape_lat = np.mean(P_2,axis=2)
        TSSE,TSVAR,TR2v2 = estimate_R2_outv2(T_tend_true_avg_2_reshape_lat,T_pred_true_avg_2_reshape_lat,60,31,180)
        QSSE,QSVAR,QR2v2 = estimate_R2_outv2(Q_tend_true_avg_2_reshape_lat,Q_pred_true_avg_2_reshape_lat,60,31,180)
        XpC, YpC = np.meshgrid(lat_C,P2_reshape_lat[:,0])
        
        #fig, ax = plt.subplots(2,2, figsize=(15,15))
        fig, ax = plt.subplots(1,2, figsize=(15,5))
        fz = 20
        contour_plot = ax[0].pcolor(XpC, P2_reshape_lat/100, TR2v2, cmap = 'Blues', vmin = 0, vmax = 1.0)
        ax[0].contour(XpC, P2_reshape_lat/100 , TR2v2, [0.7], colors='pink', linewidths=[4])
        ax[0].contour(XpC, P2_reshape_lat/100, TR2v2, [0.9], colors='orange', linewidths=[4])
        ax[0].set_title("(a) Heating", fontsize = fz)
        ax[0].set_ylim(ax[0].get_ylim()[::-1])
        ax[0].set_ylabel("Pressure (hPa)", fontsize = fz)
        ax[0].locator_params(nbins=8)
        ax[0].tick_params(axis='x', labelsize=fz*0.9)
        ax[0].tick_params(axis='y', labelsize=fz*0.9)

        ax[1].pcolor(XpC, P2_reshape_lat/100, QR2v2,  cmap = 'Blues', vmin = 0, vmax = 1.0)
        ax[1].contour(XpC, P2_reshape_lat/100, QR2v2,  [0.7], colors='pink', linewidths=[4])
        ax[1].contour(XpC, P2_reshape_lat/100, QR2v2,  [0.9], colors='orange', linewidths=[4])
        ax[1].set_title("(b) Moistening", fontsize = fz)
        ax[1].set_ylim(ax[1].get_ylim()[::-1])
        ax[1].locator_params(nbins=8)
        ax[1].set_yticks([])
        ax[1].tick_params(axis='x', labelsize=fz*0.9)

        #ax[1].yaxis.set_label_coords(-1.38,-0.09)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.82, 0.12, 0.02, 0.76])
        fig.colorbar(contour_plot, label="Skill Score "+r'$\left(\mathrm{R^{2}}\right)$', cax=cbar_ax)
        #plt.suptitle("Trained DNN Skill for Vertically Resolved Tendencies", y = 0.97)
        plt.subplots_adjust(hspace=0.13)
        figure_filename = file_path_outregrid[-37:-15]+'.png'
        plt.savefig(figure_filename)
        
        
        TR_2 = TR2_out_file.TR2.values
        QR_2 = TR2_out_file.QR2.values
        P_2 = TR2_out_file.P.values
        lon_C = TR2_out_file.lon.values
        lat_C = TR2_out_file.lat.values
        T_tend_true_avg_2 = TR2_out_file.T_tend_true_avg.values
        T_pred_true_avg_2 = TR2_out_file.T_pred_true_avg.values
        Q_tend_true_avg_2 = TR2_out_file.Q_tend_true_avg.values
        Q_pred_true_avg_2 = TR2_out_file.Q_pred_true_avg.values
        
        T_tend_true_avg_3 = np.reshape(T_tend_true_avg_2, (365*60,180,360))
        T_pred_true_avg_3 = np.reshape(T_pred_true_avg_2, (365*60,180,360))
        Q_tend_true_avg_3 = np.reshape(Q_tend_true_avg_2, (365*60,180,360))
        Q_pred_true_avg_3 = np.reshape(Q_pred_true_avg_2, (365*60,180,360))
        TSSE,TSVAR,TR2v2 = estimate_R2_out_lonlat(T_tend_true_avg_3,T_pred_true_avg_3,180,360)
        QSSE,QSVAR,QR2v2 = estimate_R2_out_lonlat(Q_tend_true_avg_3,Q_pred_true_avg_3,180,360)

        fig, ax = plt.subplots(2,1,subplot_kw={'projection':ccrs.Robinson(central_longitude=180)})
        fig.set_size_inches(15,15)
        fz = 20
        #SPCAM5_lat_lon_new = gaussian_filter(SPCAM5_lat_lon, 2, mode='nearest')
        SPCAM5_lat_lon_new = gaussian_filter((TR2v2), 2, mode='nearest')
        contour_plot = ax[0].pcolormesh(lon_C, lat_C, SPCAM5_lat_lon_new,cmap='Blues', vmin = 0, vmax = 1.0, transform=ccrs.PlateCarree())
        ax[0].contour(lon_C, lat_C, SPCAM5_lat_lon_new, [0.7], colors='pink', linewidths=[4], transform=ccrs.PlateCarree())
        ax[0].contour(lon_C, lat_C, SPCAM5_lat_lon_new, [0.9], colors='orange', linewidths=[4], transform=ccrs.PlateCarree())
        ax[0].set_title('(a) Surface Heat DNN Skill', fontsize = fz)
        ax[0].coastlines(linewidth=2.0,edgecolor='0.25')
        ax[0].gridlines()
        ax[0].add_feature(cfeature.BORDERS,linewidth=0.5,edgecolor='0.25')
        cbar_ax = fig.add_axes([0.87, 0.12, 0.03, 0.76])
        fig.colorbar(contour_plot, cax=cbar_ax)
        cbar_ax.tick_params(labelsize=fz)
        cbar_ax.set_ylabel("Skill Score "+r'$\left(\mathrm{R^{2}}\right)$', fontsize=fz)
        SPCAM5_lat_lon_new = gaussian_filter((QR2v2), 2, mode='nearest')
        contour_plot = ax[1].pcolormesh(lon_C, lat_C, SPCAM5_lat_lon_new,cmap='Blues', vmin = 0, vmax = 1.0, transform=ccrs.PlateCarree())
        ax[1].contour(lon_C, lat_C, SPCAM5_lat_lon_new, [0.7], colors='pink', linewidths=[4], transform=ccrs.PlateCarree())
        ax[1].contour(lon_C, lat_C, SPCAM5_lat_lon_new, [0.9], colors='orange', linewidths=[4], transform=ccrs.PlateCarree())
        ax[1].set_title('(b) Surface Moisture DNN Skill', fontsize = fz)
        ax[1].coastlines(linewidth=2.0,edgecolor='0.25')
        ax[1].gridlines()
        ax[1].add_feature(cfeature.BORDERS,linewidth=0.5,edgecolor='0.25')
        figure_filename = file_path_outregrid[-37:-15]+'lonlat.png'
        plt.savefig(figure_filename)










