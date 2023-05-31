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

# Please specify the temporal stride used in generating validation/test dataset
# and original GCM time-step (dt_GCM) in minutes
stride = 6
dt_GCM = 20

# load true target values of shape Ndata x 128
test_y = np.load('val_target_stride6.npy')

# load model target values of shape Npoints x 128
# or plug in your code to make predictions here using val_input_stride6.npy
pred_y = np.load('pred_y.npy')

# file 'pressures_val_stride6_60lvls.npy' contains pressure levels in Pa
pressure = np.load('pressures_val_stride6_60lvls.npy')/100
N_pressure = pressure.shape[0] # Number of pressure levels: 60

# load dictionary of latitude-longitude coordinates of GCM grid on which data is saved
with open('indextolatlons.pkl', 'rb') as f:
    data = pickle.load(f)
N_lat_long = len(data) # = 384 There are 384 points in latitude-longitude GCM grid

# Since in the current training and validation datasets all first 8 altitude levels
# from TOA show exactly zeros values for moistening tendency, we can discard the
# corresponding pressure levels in the moistening tendency plot by setting n_remove_moist = 8
# If you want to keep all pressure levels in the moistening tendency plot, please set n_remove_moist = 0
n_remove_moist = 8

######################################
####### Processing and plotting ######
######################################

# Ndata is total number of data points in validation / test dataset
# N_time_steps is the number of time-steps considered in validation / test dataset
Ndata = test_y.shape[0]
N_time_steps = Ndata//N_lat_long

# To compute the coefficient of determination we need to compute daily averages
# given stride and dt_GCM specified above we compute how many time-steps correspond to a daily window
N_dt_day = int( (60*24)/(stride*dt_GCM) )

# first we identify the latitude and longitude values among the N_lat_long points in latitude-longitude grid
lats_unique = []
lats_unique.append(data[0][0])
long_unique = []
long_unique.append(data[0][1])

for i in range(N_lat_long-1):
    if not(data[i+1][0] in lats_unique):
        lats_unique.append(data[i+1][0])
    if not(data[i+1][1] in long_unique):
        long_unique.append(data[i+1][1])
Nlats = len(lats_unique) # Number of unique latititude levels: 87
Nlong = len(long_unique) # Number of unique longitude levels: 199

# ind_lat is a list of length len(lats_unique), each of its elements contains the indices of the corresponding points among the N_lat_long ones
ind_lat = []
for i in range(Nlats):
    ind_lat_loc = []
    for j in range(N_lat_long):
        if data[j][0] == lats_unique[i]:
            ind_lat_loc.append(j)
    ind_lat.append(ind_lat_loc)

# reshape true heating tendency data into: N_time_steps x N_lat_long x N_pressure
test_heat = test_y[:,:N_pressure].reshape( (N_time_steps, N_lat_long, N_pressure) )

# compute daily avereage for true heating tendency
# resulting array of shape: Nday x N_lat_long x N_pressure, where Nday = N_time_steps//N_dt_day is the number of days
test_heat_daily = np.mean( test_heat.reshape( (N_time_steps//N_dt_day, N_dt_day, N_lat_long, N_pressure) ) , axis=1 )

# Do the last two operation for predicted heating tendency, and true and predicted moistening tendency
pred_heat = pred_y[:,:N_pressure].reshape( (N_time_steps, N_lat_long, N_pressure) )
pred_heat_daily = np.mean(pred_heat.reshape( (N_time_steps//N_dt_day, N_dt_day, N_lat_long, N_pressure) ) , axis=1 )
test_moist = test_y[:,N_pressure:N_pressure*2].reshape( (N_time_steps, N_lat_long, N_pressure) )
test_moist_daily = np.mean(test_moist.reshape( (N_time_steps//N_dt_day, N_dt_day, N_lat_long, N_pressure) ) , axis=1)
pred_moist = pred_y[:,N_pressure:N_pressure*2].reshape( (N_time_steps, N_lat_long, N_pressure) )
pred_moist_daily = np.mean(pred_moist.reshape( (N_time_steps//N_dt_day, N_dt_day, N_lat_long, N_pressure) ) , axis=1)

# To plots the coefficient of determination in pressure-latitude cross section,
# we need to average true data and predictions along longitude axis which is 
# below for true and predicted data for both heating and moistening tendencies
# The 4 resulting arrays all have a shape: Nlats x Nday x N_pressure
test_heat_daily_long = []
pred_heat_daily_long = []
test_moist_daily_long = []
pred_moist_daily_long = []
for i in range(Nlats):
    test_heat_daily_long.append(np.mean(test_heat_daily[:,ind_lat[i],:],axis=1))
    pred_heat_daily_long.append(np.mean(pred_heat_daily[:,ind_lat[i],:],axis=1))
    test_moist_daily_long.append(np.mean(test_moist_daily[:,ind_lat[i],:],axis=1))
    pred_moist_daily_long.append(np.mean(pred_moist_daily[:,ind_lat[i],:],axis=1))
test_heat_daily_long = np.array(test_heat_daily_long) # Nlats x Nday x N_pressure
pred_heat_daily_long = np.array(pred_heat_daily_long) # Nlats x Nday x N_pressure
test_moist_daily_long = np.array(test_moist_daily_long) # Nlats x Nday x N_pressure
pred_moist_daily_long = np.array(pred_moist_daily_long) # Nlats x Nday x N_pressure
  
# matplotlib.pcolor has an issue if axis coordinate are not ordered. Hence we order the unique latitude values
lats_unique = np.array(lats_unique)
arg_lats_unique = np.argsort(lats_unique)
lats_unique = lats_unique[arg_lats_unique]
X, Y = np.meshgrid(lats_unique, pressure)

# numpy return erronous errors when summing over "large" number of points with float32 format, 
# so we transform arrays to float64 type before computing coefficient of determination R2
pred_heat_daily_long = np.array(pred_heat_daily_long,dtype=np.float64)
test_heat_daily_long = np.array(test_heat_daily_long,dtype=np.float64)

# now we compute the coefficient of determination R2 on daily averaged data
coeff = 1 - np.sum( (pred_heat_daily_long-test_heat_daily_long)**2, axis=1)/np.sum( (test_heat_daily_long-np.mean(test_heat_daily_long, axis=1)[:,None,:])**2, axis=1)
# we need to order R2 results based on latitude ordering
coeff = coeff[arg_lats_unique,:]
coeff = coeff.T

fig, ax = plt.subplots(2,1, figsize=(14,28))
contour_plot = ax[0].pcolor(X, Y, coeff,cmap='Blues', vmin = 0, vmax = 1)
ax[0].contour(X, Y, coeff, [0.7], colors='pink', linewidths=[4])
ax[0].contour(X, Y, coeff, [0.9], colors='orange', linewidths=[4])
ax[0].set_ylim(ax[0].get_ylim()[::-1])
ax[0].set_ylabel("Pressure (hPa)")
ax[0].yaxis.set_label_coords(-0.2,-0.09)
ax[0].set_title("Heating Tendency")

# remove first n_remove_moist pressure levels from TOA
pressure = pressure[n_remove_moist:]
X, Y = np.meshgrid(lats_unique, pressure)
pred_moist_daily_long = pred_moist_daily_long[:,:,n_remove_moist:]
test_moist_daily_long = test_moist_daily_long[:,:,n_remove_moist:]

# float64 format before computing error metrics
pred_moist_daily_long = np.array(pred_moist_daily_long,dtype=np.float64)
test_moist_daily_long = np.array(test_moist_daily_long,dtype=np.float64)

# now we compute the coefficient of determination R2 on daily averaged data
coeff = 1 - np.sum( (pred_moist_daily_long-test_moist_daily_long)**2, axis=1)/np.sum( (test_moist_daily_long-np.mean(test_moist_daily_long, axis=1)[:,None,:])**2, axis=1)
# we need to order R2 results based on latitude ordering
coeff = coeff[arg_lats_unique,:]
coeff = coeff.T

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.82, 0.12, 0.05, 0.76])
fig.colorbar(contour_plot, label="Skill Score "+r'$\left(\mathrm{R^{2}}\right)$', cax=cbar_ax)
ax[1].pcolor(X, Y, coeff,cmap='Blues', vmin = 0, vmax = 1)
ax[1].contour(X, Y, coeff, [0.7], colors='pink', linewidths=[4])
ax[1].contour(X, Y, coeff, [0.9], colors='orange', linewidths=[4])
ax[1].set_ylim(ax[1].get_ylim()[::-1])
ax[1].set_xlabel("Degrees Latitude")
ax[1].set_title("Moistening Tendency")

plt.savefig('coeff_det_heat_moist_press_lat.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
    