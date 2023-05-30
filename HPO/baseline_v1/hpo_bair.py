import xarray as xr
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models
from keras import layers
from keras import callbacks
import keras_tuner as kt
from keras_tuner import HyperModel
from keras_tuner import RandomSearch
import os
import tensorflow_addons as tfa
# from qhoptim.tf import QHAdamOptimizer
import sys
import argparse
import glob
import random


def set_environment(num_gpus_per_node=4, oracle_port = "8000"):
    '''
    This function sets up the environment variables for the Keras Tuner Oracle.
    It should be called at the beginning of the main function.
    The default oracle port is 8000, but 8000 is also very popular.
    When running into GPU issues, scanning for alternative ports is recommended.
    '''

    print('<< set_environment START >>')
    num_gpus_per_node = str(num_gpus_per_node)
    nodename = os.environ['SLURMD_NODENAME']
    procid = os.environ['SLURM_LOCALID']
    print(f'node name: {nodename}')
    print(f'procid:    {procid}')
    stream = os.popen('scontrol show hostname $SLURM_NODELIST')
    output = stream.read()
    oracle = output.split("\n")[0]
    # oracle_ip = os.environ["NERSC_NODE_HSN_IP"]
    print(f'oracle ip: {oracle}')
    if procid==str(num_gpus_per_node): # This takes advantage of the fact that procid numbering starts with ZERO
        os.environ["KERASTUNER_TUNER_ID"] = "chief"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("Keras Tuner Oracle has been assigned.")
    else:
        os.environ["KERASTUNER_TUNER_ID"] = "tuner-" + str(nodename) + "-" + str(procid)
        # os.environ["CUDA_VISIBLE_DEVICES"] = f"{int(int(procid)//workers_per_gpu)}"
        os.environ["CUDA_VISIBLE_DEVICES"] = procid
    print(f'SY DEBUG: procid-{procid} / GPU-ID-{os.environ["CUDA_VISIBLE_DEVICES"]}')

    os.environ["KERASTUNER_ORACLE_IP"] = oracle + ".ib.bridges2.psc.edu" # Use full hostname
    os.environ["KERASTUNER_ORACLE_PORT"] = oracle_port
    print("KERASTUNER_TUNER_ID:    %s"%os.environ["KERASTUNER_TUNER_ID"])
    print("KERASTUNER_ORACLE_IP:   %s"%os.environ["KERASTUNER_ORACLE_IP"])
    print("KERASTUNER_ORACLE_PORT: %s"%os.environ["KERASTUNER_ORACLE_PORT"])
    #print(os.environ)
    print('<< set_environment END >>')

def build_model(hp):
    '''
    This function builds the model for the Keras Tuner.
    Choose the hyperparameters to scan over and your preferred architecture in this function.
    INPUTS (124 features per sample):
    state_t: 60 levels (lower index = higher in atmosphere), units: K, air temperature
    state_q0001: 60 levels (lower index = higher in atmosphere), units: kg/kg, specific humidity
    state_ps: scalar, units: Pa, surface pressure
    pbuf_SOLIN: scalar, units: W/m2, solar insolation
    pbuf_LHFLX: scalar, units: W/m2, surface latent heat flux
    pbuf_SHFLX: scalar, units: W/m2, surface sensible heat flux


    OUTPUTS (128 features per sample):
    ptend_t: 60 levels (lower index = higher in atmosphere), units: K/s, tendency of air temperature
    ptend_q0001: 60 levels (lower index = higher in atmosphere), units: kg/kg/s, tendency of specific humidity
    cam_out_NETSW: scalar, units: W/m2, net shortwave flux at surface
    cam_out_FLWDS: scalar, units: W/m2, downward longwave flux at surface
    cam_out_PRECSC: scalar, units: m/s, snow rate (liquid water equivalent)
    cam_out_PRECC: scalar, units: m/s, rain rate
    cam_out_SOLS: scalar, units: W/m2, downward visible direct solar flux to surface
    cam_out_SOLL: scalar, units: W/m2, downward near-infrared direct solar flux to surface
    cam_out_SOLSD: scalar, units: W/m2, downward visible diffuse solar flux to surface
    cam_out_SOLLD: scalar, units: W/m2, downward near-infrared diffuse solar flux to surface
    '''        
    # hyperparameters to be tuned:
    n_layers = hp.Int("num_layers", 4, 12, default=4)
    hp_batch_size = hp.Choice("batch_size",
                                [160, 320,  640, 1280, 1536, 2560],
                                default=2560)
    hp_optimizer = hp.Choice("optimizer", ['Adam', 'RAdam'], default='Adam')
    
    # construct a model
    # input layer
    x = keras.layers.Input(shape=(124,), name='input')
    input_layer = x
    
    # hidden layers
    for klayer in range(n_layers):
        n_units = hp.Int(f"units_{klayer}", min_value=128, max_value=1024, step=128, default=128)
        x = keras.layers.Dense(n_units)(x)
        x = keras.layers.LeakyReLU(alpha=.15)(x)
            
    # output layer (upper)
    x = keras.layers.Dense(128)(x)
    x = keras.layers.LeakyReLU(alpha=.15)(x)
    
    # output layer (lower)
    output_lin   = keras.layers.Dense(120, activation='linear')(x)
    output_relu  = keras.layers.Dense(8, activation='relu')(x)
    output_layer = keras.layers.Concatenate()([output_lin, output_relu])

    model = keras.Model(input_layer, output_layer, name='trial_model')
    
    # Optimizer
    # Set up cyclic learning rate
    INIT_LR = 2.5e-4
    MAX_LR  = 2.5e-3
    steps_per_epoch = 10091520 // hp_batch_size
    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=INIT_LR,
                                                maximal_learning_rate=MAX_LR,
                                                scale_fn = lambda x: 1/(2.**(x-1)),
                                                step_size = 2 * steps_per_epoch,
                                                scale_mode = 'cycle'
                                                )

    # Set up optimizer
    # clr = hp.Float("lr", min_value=1e-4, max_value=1e-3, sampling="log")
    if hp_optimizer == "Adam":
        my_optimizer = keras.optimizers.Adam(learning_rate=clr)
    elif hp_optimizer == "RAdam":
        my_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=clr)
                                    
    # compile
    model.compile(optimizer=my_optimizer, #optimizer=keras.optimizers.Adam(learning_rate=clr),
                    loss='mse',
                    metrics=['mse','mae','accuracy'])
    
    # model summary
    print(model.summary())
    
    return model

def main():

    training_data_path = '/ocean/projects/atm200007p/jlin96/neurips_proj/e3sm_train_npy/'

    if os.environ["KERASTUNER_TUNER_ID"] != "chief":
        with open(training_data_path + 'train_input.npy', 'rb') as f:
            train_input = np.load(f)
            # train_input.shape returns (10091520, 124)
        with open(training_data_path + 'train_target.npy', 'rb') as f:
            train_target = np.load(f)
            # train_target.shape returns (10091520, 128)
        with open(training_data_path + 'val_input.npy', 'rb') as f:
            val_input = np.load(f)
            # val_input.shape returns (1441920, 124)
        with open(training_data_path + 'val_target.npy', 'rb') as f:
            val_target = np.load(f)
            # val_target.shape returns (1441920, 128)
    tuner = kt.RandomSearch(
        hypermodel = build_model,
        objective = 'val_loss',
        max_trials = 10,
        executions_per_trial = 1,
        overwrite = False,
        directory = 'results',
        project_name = 'bair'
    )

    # def make_model_id(*identifiers):
    #     return "_".join(list(identifiers))
    # model_id = make_model_id(os.environ["KERASTUNER_TUNER_ID"])
    
    filepath_checkpoint = 'results/bair/best_models/'
    filepath_csv = 'logs/bair/training_csvs/'

    # early_stopping = callbacks.EarlyStopping('val_loss', patience = 10)
    # tboard_callback = callbacks.TensorBoard(log_dir = 'logs/bair/logs_tensorboard', histogram_freq = 1)
    # checkpoint_callback = callbacks.ModelCheckpoint(filepath=filepath_checkpoint,
    #                                                         save_weights_only=False,
    #                                                         monitor='val_mse',
    #                                                         mode='min',
    #                                                         save_best_only=True)
    # csv_callback = callbacks.CSVLogger(filepath_csv, separator=",", append=True)

    # my_callbacks = [early_stopping, 
    #                 tboard_callback, 
    #                 checkpoint_callback, 
    #                 csv_callback]
    
    kwargs = {
        'epochs':100,
        'verbose':2,
        'shuffle': True,
        'callbacks': [callbacks.EarlyStopping('val_loss', patience = 10),
                      callbacks.TensorBoard(log_dir = 'logs/bair/logs_tensorboard', histogram_freq = 1),
                      callbacks.ModelCheckpoint(filepath=filepath_checkpoint,
                                                            verbose = 1,
                                                            save_weights_only=False,
                                                            monitor='val_mse',
                                                            mode='min',
                                                            save_best_only=True),
                      callbacks.CSVLogger(filepath_csv, separator=",", append=True),
                      callbacks.BackupAndRestore('results/bair_backup')]
    }
    # print("---SEARCH SPACE---")
    # tuner.search_space_summary()

    # search
    if os.environ["KERASTUNER_TUNER_ID"] != "chief":
        tuner.search(train_input, train_target, validation_data=(val_input, val_target), **kwargs)

if __name__ == '__main__':

    # command line argument

    # assign GPUs for workers
    # gpus_per_node = 4 # NERSC Perlmutter
    # ntasks = int(os.environ['SLURM_NTASKS']) # "total number of workers" + 1 (for oracle)
    # nnodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    # workers_per_node = int((ntasks - 1) / nnodes)
    # workers_per_gpu  = int(workers_per_node / gpus_per_node)
    set_environment(num_gpus_per_node=4, oracle_port = "8000")

    # limit memory preallocation
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True) # only using a single GPU per trial

    # query available GPU (as debugging info only)
    print(tf.config.list_physical_devices('GPU'))

    # run main program
    main()
