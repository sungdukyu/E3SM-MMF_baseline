import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob, os
import re
import tensorflow as tf

class data_preprocessing:
    def __init__(self, 
                 input_vars, 
                 output_vars, 
                 grid_info,
                 inp_mean,
                 inp_max,
                 inp_min,
                 out_scale):
        self.latlim = [-999,999]
        self.lonlim = [-999,999]
        self.latlonnum = 384 # number of unique lat/lon grid points
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.grid_info = grid_info
        self.inp_mean = inp_mean
        self.inp_max = inp_max
        self.inp_min = inp_min
        self.out_scale = out_scale

    def get_xrdata(self, file, file_vars = ''):
        '''
        This function reads in a file and returns an xarray dataset with the variables specified.
        '''
        ds = xr.open_dataset(file, engine = 'netcdf4')
        if file_vars != "":
            ds = ds[file_vars]
        ds = ds.merge(self.grid_info[['lat','lon']])
        ds = ds.where((ds['lat']>self.latlim[0])*(ds['lat']<self.latlim[1]),drop=True)
        ds = ds.where((ds['lon']>self.lonlim[0])*(ds['lon']<self.lonlim[1]),drop=True)
        return ds

    def get_input(self, input_file):
        '''
        This function reads in a file and returns an xarray dataset with the input variables for the emulator.
        '''
        # read inputs
        return self.get_xrdata(input_file, self.input_vars)

    def get_output(self, output_file, input_file = ''):
        '''
        This function reads in a file and returns an xarray dataset with the output variables for the emulator.
        '''
        # read inputs
        if input_file == '':
            input_file = output_file.replace('.mlo.','.mli.')
        ds_input = self.get_input(input_file)
        
        ds_output = self.get_xrdata(output_file)
        # each timestep is 20 minutes which corresponds to 1200 seconds
        ds_output['ptend_t'] = (ds_output['state_t'] - ds_input['state_t'])/1200 # T tendency [K/s]
        ds_output['ptend_q0001'] = (ds_output['state_q0001'] - ds_input['state_q0001'])/1200 # Q tendency [kg/kg/s]
        ds_output = ds_output[self.output_vars]
        return ds_output
    
    def load_ncdata_with_generator(self, filelist:list):
        '''
        This function works as a dataloader when training the emulator with raw netCDF files.
        mli corresponds to input
        mlo corresponds to output
        '''
        def gen():
            for file in filelist:
                
                # read inputs
                ds_input = self.get_input(file)
                # read outputs
                ds_output = self.get_output(file)
                
                # normalizatoin, scaling
                ds_input = (ds_input - self.inp_mean)/(self.inp_max - self.inp_min)
                ds_output = ds_output*self.out_scale

                # stack
                #ds = ds.stack({'batch':{'sample','ncol'}})
                ds_input = ds_input.stack({'batch':{'ncol'}})
                ds_input = ds_input.to_stacked_array('mlvar', sample_dims=['batch'], name='mli')
                #dso = dso.stack({'batch':{'sample','ncol'}})
                ds_output = ds_output.stack({'batch':{'ncol'}})
                ds_output = ds_output.to_stacked_array('mlvar', sample_dims=['batch'], name='mlo')
                
                yield (ds_input.values, ds_output.values)

        return tf.data.Dataset.from_generator(
            gen,
            output_types=(tf.float64, tf.float64),
            output_shapes=((None,124),(None,128))
        )
    
    def make_npy(self, 
                 filelist:list, 
                 prefix = '', 
                 save_path = '', 
                 stride_sample = 7,
                 save_latlontime_dict = False):
        '''
        This function saves the training data as a .npy file. Prefix should be train or val
        '''
        prefix = save_path + prefix
        data_files = sorted(filelist)[::stride_sample]
        data_loader = self.load_ncdata_with_generator(data_files)
        npy_iterator = list(data_loader.as_numpy_iterator())
        npy_input = np.concatenate([npy_iterator[x][0] for x in range(len(npy_iterator))])
        npy_output = np.concatenate([npy_iterator[x][1] for x in range(len(npy_iterator))])
        with open(save_path + prefix + '_input.npy', 'wb') as f:
            np.save(f, np.float32(npy_input))
        with open(save_path + prefix + '_output.npy', 'wb') as f:
            np.save(f, np.float32(npy_output))
        if save_latlontime_dict:
            dates = [re.sub('^.*mli\.', '', x) for x in data_files]
            dates = [re.sub('\.nc$', '', x) for x in dates]
            repeat_dates = []
            for date in dates:
                for i in range(self.latlonnum):
                    repeat_dates.append(date)
            latlontime = {i: [(self.grid_info['lat'].values[i%self.latlonnum], self.grid_info['lon'].values[i%self.latlonnum]), repeat_dates[i]] for i in range(npy_input.shape[0])}
            with open(save_path + prefix + '_indextolatlontime.pkl', 'wb') as f:
                pickle.dump(latlontime, f)
        return
    
    @classmethod
    def set_latlim(cls, latlim):
        '''
        This function sets the latitude limits for reading in the data.
        @Sungduk please provide clarity on why this is necessary.
        '''
        cls.latlim = latlim

    @classmethod
    def set_lonlim(cls, lonlim):
        '''
        This function sets the longitude limits for reading in the data.
        @Sungduk please provide clarity on why this is necessary.
        '''
        cls.lonlim = lonlim

    @staticmethod
    def ls(data_path = ''):
        '''
        You can treat this as a Python wrapper for the bash command "ls".
        '''
        return os.popen(' '.join(['ls', data_path])).read().splitlines()
    

    



  




