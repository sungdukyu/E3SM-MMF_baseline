import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob, os
import re
import tensorflow as tf
import netCDF4
import h5py

class e3sm_preprocessing:
    def __init__(self,
                 data_path, 
                 input_vars, 
                 target_vars, 
                 grid_info,
                 inp_mean,
                 inp_max,
                 inp_min,
                 out_scale):
        self.latlim = [-999,999]
        self.lonlim = [-999,999]
        self.data_path = data_path
        self.latlonnum = 384 # number of unique lat/lon grid points
        self.input_vars = input_vars
        self.target_vars = target_vars
        self.grid_info = grid_info
        self.inp_mean = inp_mean
        self.inp_max = inp_max
        self.inp_min = inp_min
        self.out_scale = out_scale
        self.train_regexps = None
        self.train_stride_sample = None
        self.train_filelist = None
        self.val_regexps = None
        self.val_stride_sample = None
        self.val_filelist = None
        self.scoring_regexps = None
        self.scoring_stride_sample = None
        self.scoring_filelist = None
        self.test_regexps = None
        self.test_stride_sample = None
        self.test_filelist = None

    def get_xrdata(self, file, file_vars = ''):
        '''
        This function reads in a file and returns an xarray dataset with the variables specified.
        '''
        ds = xr.open_dataset(file, engine = 'netcdf4')
        if file_vars != "":
            ds = ds[file_vars]
        ds = ds.merge(self.grid_info[['lat','lon']])
        ds = ds.where((ds['lat']>self.latlim[0])*(ds['lat']<self.latlim[1]), drop=True)
        ds = ds.where((ds['lon']>self.lonlim[0])*(ds['lon']<self.lonlim[1]), drop=True)
        return ds

    def get_input(self, input_file):
        '''
        This function reads in a file and returns an xarray dataset with the input variables for the emulator.
        '''
        # read inputs
        return self.get_xrdata(input_file, self.input_vars)

    def get_target(self, target_file, input_file = ''):
        '''
        This function reads in a file and returns an xarray dataset with the target variables for the emulator.
        '''
        # read inputs
        if input_file == '':
            input_file = target_file.replace('.mlo.','.mli.')
        ds_input = self.get_input(input_file)
        
        ds_target = self.get_xrdata(target_file)
        # each timestep is 20 minutes which corresponds to 1200 seconds
        ds_target['ptend_t'] = (ds_target['state_t'] - ds_input['state_t'])/1200 # T tendency [K/s]
        ds_target['ptend_q0001'] = (ds_target['state_q0001'] - ds_input['state_q0001'])/1200 # Q tendency [kg/kg/s]
        ds_target = ds_target[self.target_vars]
        return ds_target
    
    @classmethod
    def set_stride_sample(cls, data_split, stride_sample):
        '''
        This function sets the stride_sample for train, val, scoring, and test.
        '''
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        if data_split == 'train':
            cls.train_stride_sample = stride_sample
        elif data_split == 'val':
            cls.val_stride_sample = stride_sample
        elif data_split == 'scoring':
            cls.scoring_stride_sample = stride_sample
        elif data_split == 'test':
            cls.test_stride_sample = stride_sample
    
    @classmethod
    def set_filelist(cls, data_split):
        '''
        This function sets the filelists corresponding to data splits for train, val, scoring, and test.
        '''
        filelist = []
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        if data_split == 'train':
            assert cls.train_reg_exps is not None, 'reg_exps for train is not set.'
            assert cls.train_stride_sample is not None, 'stride_sample for train is not set.'
            for reg_exp in cls.train_reg_exps:
                filelist = filelist + glob.glob(cls.data_path + "*/" + reg_exp)
            cls.train_filelist = sorted(filelist)[::cls.train_stride_sample]
        elif data_split == 'val':
            assert cls.val_reg_exps is not None, 'reg_exps for val is not set.'
            assert cls.val_stride_sample is not None, 'stride_sample for val is not set.'
            for reg_exp in cls.val_reg_exps:
                filelist = filelist + glob.glob(cls.data_path + "*/" + reg_exp)
            cls.val_filelist = sorted(filelist)[::cls.val_stride_sample]
        elif data_split == 'scoring':
            assert cls.scoring_reg_exps is not None, 'reg_exps for scoring is not set.'
            assert cls.scoring_stride_sample is not None, 'stride_sample for scoring is not set.'
            for reg_exp in cls.scoring_reg_exps:
                filelist = filelist + glob.glob(cls.data_path + "*/" + reg_exp)
            cls.scoring_filelist = sorted(filelist)[::cls.scoring_stride_sample]
        elif data_split == 'test':
            assert cls.test_reg_exps is not None, 'reg_exps for test is not set.'
            assert cls.test_stride_sample is not None, 'stride_sample for test is not set.'
            for reg_exp in cls.test_reg_exps:
                filelist = filelist + glob.glob(cls.data_path + "*/" + reg_exp)
            cls.test_filelist = sorted(filelist)[::cls.test_stride_sample]
    
    def load_ncdata_with_generator(self, data_split):
        '''
        This function works as a dataloader when training the emulator with raw netCDF files.
        mli corresponds to input
        mlo corresponds to target
        '''
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        if data_split == 'train':
            assert self.train_filelist is not None, 'train_filelist is not set.'
            filelist = self.train_filelist
        elif data_split == 'val':
            assert self.val_filelist is not None, 'val_filelist is not set.'
            filelist = self.val_filelist
        elif data_split == 'scoring':
            assert self.scoring_filelist is not None, 'scoring_filelist is not set.'
            filelist = self.scoring_filelist
        elif data_split == 'test':
            assert self.test_filelist is not None, 'test_filelist is not set.'
            filelist = self.test_filelist
        def gen():
            for file in filelist:
                # read inputs
                ds_input = self.get_input(file)
                # read targets
                ds_target = self.get_target(file)
                
                # normalization, scaling
                ds_input = (ds_input - self.inp_mean)/(self.inp_max - self.inp_min)
                ds_target = ds_target*self.out_scale

                # stack
                #ds = ds.stack({'batch':{'sample','ncol'}})
                ds_input = ds_input.stack({'batch':{'ncol'}})
                ds_input = ds_input.to_stacked_array('mlvar', sample_dims=['batch'], name='mli')
                #dso = dso.stack({'batch':{'sample','ncol'}})
                ds_target = ds_target.stack({'batch':{'ncol'}})
                ds_target = ds_target.to_stacked_array('mlvar', sample_dims=['batch'], name='mlo')
                
                yield (ds_input.values, ds_target.values)

        return tf.data.Dataset.from_generator(
            gen,
            target_types=(tf.float64, tf.float64),
            target_shapes=((None,124),(None,128))
        )
    
    def make_npy(self, 
                 reg_exps = None, 
                 prefix = '',
                 save_path = '', 
                 stride_sample = 7,
                 save_latlontime_dict = False):
        '''
        This function saves the training data as a .npy file. Prefix should be train or val
        '''
        prefix = save_path + prefix
        data_files = sorted(self.get_filelist(self.data_path, reg_exps, stride_sample))[::stride_sample]
        data_loader = self.load_ncdata_with_generator(data_files)
        npy_iterator = list(data_loader.as_numpy_iterator())
        npy_input = np.concatenate([npy_iterator[x][0] for x in range(len(npy_iterator))])
        npy_target = np.concatenate([npy_iterator[x][1] for x in range(len(npy_iterator))])
        with open(save_path + prefix + '_input.npy', 'wb') as f:
            np.save(f, np.float32(npy_input))
        with open(save_path + prefix + '_target.npy', 'wb') as f:
            np.save(f, np.float32(npy_target))
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
    
    def create_pressure_grid(self, ps, save_path = ""):
        '''
        This function unscales the pressure variable so that it can be used to create a pressure grid.
        '''
        unscaled_ps = ps * (self.inp_max["state_ps"].values - self.inp_min["state_ps"].values) + self.inp_mean["state_ps"].values
        unscaled_ps = unscaled_ps[:, np.newaxis]
        num_samples = unscaled_ps.shape[0]
        hyam_component = np.repeat(self.grid_info["hyam"].values[np.newaxis, :], num_samples, axis=0)*1e5
        hybm_component = np.repeat(self.grid_info["hybm"].values[np.newaxis, :], num_samples, axis=0)*unscaled_ps
        pressure_grid = hyam_component + hybm_component
        if save_path != "":
            with open(save_path + "pressure_grid.npy", "wb") as f:
                np.save(f, pressure_grid)
        return pressure_grid
    
    def reshape_npy(self, var_arr, var_arr_dim):
        '''
        This function reshapes the a variable in numpy such that time gets its own axis (instead of being num_samples x num_levels).
        Shape of target would be (timestep, lat/lon combo, num_levels)
        '''
        var_arr = var_arr.reshape((int(var_arr.shape[0]/self.latlonnum), self.latlonnum, var_arr_dim))
        return var_arr

    @staticmethod
    def ls(dir_path = ''):
        '''
        You can treat this as a Python wrapper for the bash command "ls".
        '''
        return os.popen(' '.join(['ls', dir_path])).read().splitlines()
    
    @staticmethod
    def set_plot_params():
        '''
        This function sets the plot parameters for matplotlib.
        '''
        plt.close('all')
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rc('font', family='sans')
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
        # %config InlineBackend.figure_format = 'retina'
        # use the above line when working in a jupyter notebook
        return
    
    @staticmethod
    def reshape_input_for_cnn(npy_input, save_path = ''):
        '''
        This function reshapes a numpy input array to be compatible with CNN training.
        Each variable becomes its own channel.
        For the input there are 6 channels, each with 60 vertical levels.
        The last 4 channels correspond to scalars repeated across all 60 levels.
        '''
        npy_input_cnn = np.stack([
            npy_input[:, 0:60],
            npy_input[:, 60:120],
            np.repeat(npy_input[:, 120][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_input[:, 121][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_input[:, 122][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_input[:, 123][:, np.newaxis], 60, axis = 1)], axis = 2)
        
        if save_path != '':
            with open(save_path + 'train_input_cnn.npy', 'wb') as f:
                np.save(f, np.float32(npy_input_cnn))
        return npy_input_cnn
    
    @staticmethod
    def reshape_target_for_cnn(npy_target, save_path = ''):
        '''
        This function reshapes a numpy target array to be compatible with CNN training.
        Each variable becomes its own channel.
        For the input there are 6 channels, each with 60 vertical levels.
        The last 4 channels correspond to scalars repeated across all 60 levels.
        '''
        npy_target_cnn = np.stack([
            npy_target[:, 0:60],
            npy_target[:, 60:120],
            np.repeat(npy_target[:, 120][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_target[:, 121][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_target[:, 122][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_target[:, 123][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_target[:, 124][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_target[:, 125][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_target[:, 126][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_target[:, 127][:, np.newaxis], 60, axis = 1)], axis = 2)
        
        if save_path != '':
            with open(save_path + 'train_target_cnn.npy', 'wb') as f:
                np.save(f, np.float32(npy_target_cnn))
        return npy_target_cnn
    
    @staticmethod
    def reshape_target_from_cnn(npy_predict_cnn, save_path = ''):
        '''
        This function reshapes CNN target to (num_samples, 128) for standardized metrics
        '''
        npy_predict_cnn_reshaped = np.concatenate([
            npy_predict_cnn[:,:,0],
            npy_predict_cnn[:,:,1],
            np.mean(npy_predict_cnn[:,:,2], axis = 1)[:, np.newaxis],
            np.mean(npy_predict_cnn[:,:,3], axis = 1)[:, np.newaxis],
            np.mean(npy_predict_cnn[:,:,4], axis = 1)[:, np.newaxis],
            np.mean(npy_predict_cnn[:,:,5], axis = 1)[:, np.newaxis],
            np.mean(npy_predict_cnn[:,:,6], axis = 1)[:, np.newaxis],
            np.mean(npy_predict_cnn[:,:,7], axis = 1)[:, np.newaxis],
            np.mean(npy_predict_cnn[:,:,8], axis = 1)[:, np.newaxis],
            np.mean(npy_predict_cnn[:,:,9], axis = 1)[:, np.newaxis]], axis = 1)
        
        if save_path != '':
            with open(save_path + 'cnn_predict_reshaped.npy', 'wb') as f:
                np.save(f, np.float32(npy_predict_cnn_reshaped))
        return npy_predict_cnn_reshaped




  




