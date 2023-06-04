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

with open('indextolatlons.pkl', 'rb') as f:
    data = pickle.load(f)

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
model_name_l = ['cVAE','RPN','MLP']
n_model = len(model_name_l)

lats_unique = []
lats_unique.append(data[0][0])
long_unique = []
long_unique.append(data[0][1])

for i in range(len(data)-1):
    if not(data[i+1][0] in lats_unique):
        lats_unique.append(data[i+1][0])
    if not(data[i+1][1] in long_unique):
        long_unique.append(data[i+1][1])

Nlats = len(lats_unique) # 87
Nlong = len(long_unique) # 199

ind_lat = []
for i in range(Nlats):
    ind_lat_loc = []
    for j in range(len(data)):
        if data[j][0] == lats_unique[i]:
            ind_lat_loc.append(j)
    ind_lat.append(ind_lat_loc)

lats_unique = np.array(lats_unique)
arg_lats_unique = np.argsort(lats_unique)
lats_unique = lats_unique[arg_lats_unique]

test_y = np.load('/ocean/projects/atm200007p/jlin96/neurips_proj/e3sm_train_npy/val_target_stride6.npy')
test_y = np.array(test_y,dtype=np.float64)

pred_y_l = []

import h5py
hf = h5py.File('/ocean/projects/atm200007p/mbhouri/neurips_data/cvae.h5', 'r')
Datasetnames=hf.keys()
n1 = hf.get('pred')
pred_y = np.array(n1)
pred_y_l.append(pred_y)

pred_y = np.load('/ocean/projects/atm200007p/mbhouri/neurips_data/rpn_pred_v1_stride6.npy')
pred_y = np.array(pred_y,dtype=np.float64)
pred_y_l.append(pred_y)

pred_y = np.load('/ocean/projects/atm200007p/sungduk/for_aziz/MLP_v1_ne4/001_backup_phase-7_retrained_models_step2_lot-147_trial_0027.best.h5.npy')
pred_y_l.append(pred_y)

def reshape(std_test_H):
    std_test_H_heat = std_test_H[:,:60].reshape((int(std_test_H[:,:60].shape[0]/384), 384, 60))
    
    std_test_H_heat_daily = np.mean(std_test_H_heat.reshape((std_test_H_heat.shape[0]//12,12,384,60)) , axis=1) # Nday x 384 x 60
    std_test_H_moist = std_test_H[:,60:120].reshape((int(std_test_H[:,60:120].shape[0]/384), 384, 60))
    std_test_H_moist_daily = np.mean(std_test_H_moist.reshape((std_test_H_moist.shape[0]//12,12,384,60)) , axis=1) # Nday x 384 x 60
    
    std_test_H_heat_daily_long = []
    std_test_H_moist_daily_long = []
    for i in range(Nlats):
        std_test_H_heat_daily_long.append(np.mean(std_test_H_heat_daily[:,ind_lat[i],:],axis=1))
        std_test_H_moist_daily_long.append(np.mean(std_test_H_moist_daily[:,ind_lat[i],:],axis=1))
    std_test_H_heat_daily_long = np.array(std_test_H_heat_daily_long) # lat x Nday x 60
    std_test_H_moist_daily_long = np.array(std_test_H_moist_daily_long) # lat x Nday x 60
    
    return std_test_H_heat_daily_long, std_test_H_moist_daily_long

test_heat_daily_long, test_moist_daily_long = reshape(test_y)

# may need to adjust figure size if number of models considered is different from 3
fig, ax = plt.subplots(2,n_model, figsize=(n_model*12,18))
y = np.load('pressures_val_stride6_60lvls.npy')/100
X, Y = np.meshgrid(lats_unique, y)
    
for i in range(n_model):
    pred_heat_daily_long, pred_moist_daily_long = reshape(pred_y_l[i])
    
    
    coeff = 1 - np.sum( (pred_heat_daily_long-test_heat_daily_long)**2, axis=1)/np.sum( (test_heat_daily_long-np.mean(test_heat_daily_long, axis=1)[:,None,:])**2, axis=1)
    coeff = coeff[arg_lats_unique,:]
    coeff = coeff.T
    
    contour_plot = ax[0,i].pcolor(X, Y, coeff,cmap='Blues', vmin = 0, vmax = 1) # pcolormesh
    ax[0,i].contour(X, Y, coeff, [0.7], colors='pink', linewidths=[4])
    ax[0,i].contour(X, Y, coeff, [0.9], colors='orange', linewidths=[4])
    ax[0,i].set_ylim(ax[0,i].get_ylim()[::-1])
    ax[0,i].set_title(model_name_l[i] + " - Heating")
    ax[0,i].set_xticks([])
    
    coeff = 1 - np.sum( (pred_moist_daily_long-test_moist_daily_long)**2, axis=1)/np.sum( (test_moist_daily_long-np.mean(test_moist_daily_long, axis=1)[:,None,:])**2, axis=1)
    coeff = coeff[arg_lats_unique,:]
    coeff = coeff.T
    
    contour_plot = ax[1,i].pcolor(X, Y, coeff,cmap='Blues', vmin = 0, vmax = 1) # pcolormesh
    ax[1,i].contour(X, Y, coeff, [0.7], colors='pink', linewidths=[4])
    ax[1,i].contour(X, Y, coeff, [0.9], colors='orange', linewidths=[4])
    ax[1,i].set_ylim(ax[1,i].get_ylim()[::-1])
    ax[1,i].set_title(model_name_l[i] + " - Moistening")
    
    if i != 0:
        ax[0,i].set_yticks([])
        ax[1,i].set_yticks([])
        
# lines below for x and y label axes are valid if 3 models are considered
# we want to put only one label for each axis
# if nbr of models is different from 3 please adjust label location to center it
ax[1,1].set_xlabel("Degrees Latitude")
#ax[1,1].xaxis.set_label_coords(-0.10,-0.10)

ax[0,0].set_ylabel("Pressure (hPa)")
ax[0,0].yaxis.set_label_coords(-0.2,-0.09) # (-1.38,-0.09)
  
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.82, 0.12, 0.02, 0.76])
fig.colorbar(contour_plot, label="Skill Score "+r'$\left(\mathrm{R^{2}}\right)$', cax=cbar_ax)
plt.suptitle("Baseline models Skill for Vertically Resolved Tendencies", y = 0.97)
plt.subplots_adjust(hspace=0.13)

plt.savefig('press_lat_diff_models.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
