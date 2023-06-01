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

from keras.layers.convolutional import Conv1D


def set_environment(num_gpus_per_node=4, oracle_port="8000"):
    """
    This function sets up the environment variables for the Keras Tuner Oracle.
    It should be called at the beginning of the main function.
    The default oracle port is 8000, but 8000 is also very popular.
    When running into GPU issues, scanning for alternative ports is recommended.
    """

    print("<< set_environment START >>")
    num_gpus_per_node = str(num_gpus_per_node)
    nodename = os.environ["SLURMD_NODENAME"]
    procid = os.environ["SLURM_LOCALID"]
    print(f"node name: {nodename}")
    print(f"procid:    {procid}")
    stream = os.popen("scontrol show hostname $SLURM_NODELIST")
    output = stream.read()
    oracle = output.split("\n")[0]
    # oracle_ip = os.environ["NERSC_NODE_HSN_IP"]
    print(f"oracle ip: {oracle}")
    if procid == str(
        num_gpus_per_node
    ):  # This takes advantage of the fact that procid numbering starts with ZERO
        os.environ["KERASTUNER_TUNER_ID"] = "chief"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("Keras Tuner Oracle has been assigned.")
    else:
        os.environ["KERASTUNER_TUNER_ID"] = "tuner-" + str(nodename) + "-" + str(procid)
        # os.environ["CUDA_VISIBLE_DEVICES"] = f"{int(int(procid)//workers_per_gpu)}"
        os.environ["CUDA_VISIBLE_DEVICES"] = procid
    print(f'SY DEBUG: procid-{procid} / GPU-ID-{os.environ["CUDA_VISIBLE_DEVICES"]}')

    os.environ["KERASTUNER_ORACLE_IP"] = (
        oracle + ".ib.bridges2.psc.edu"
    )  # Use full hostname
    os.environ["KERASTUNER_ORACLE_PORT"] = oracle_port
    print("KERASTUNER_TUNER_ID:    %s" % os.environ["KERASTUNER_TUNER_ID"])
    print("KERASTUNER_ORACLE_IP:   %s" % os.environ["KERASTUNER_ORACLE_IP"])
    print("KERASTUNER_ORACLE_PORT: %s" % os.environ["KERASTUNER_ORACLE_PORT"])
    # print(os.environ)
    print("<< set_environment END >>")


class CNNHyperModel(kt.HyperModel):
    
    def build_model(self, hp):
        """
        Create a ResNet-style 1D CNN. The data is of shape (batch, lev, vars)
        where lev is treated as the spatial dimension. The architecture
        consists of residual blocks with each two conv layers.
        """
        # Define output shapes
        in_shape = (60, 6)
        out_shape = (60, 10)
        output_length_lin = 2
        output_length_relu = out_shape[-1] - 2

        hp_depth = hp.Int("depth", 2, 50, default=2)
        hp_channel_width = hp.Int("channel_width", 32, 512, default=32)
        hp_kernel_width = hp.Choice("kernel_width", [3, 5, 7, 9], default=3)
        hp_activation = hp.Choice(
            "activation", ["gelu", "elu", "relu", "swish"], default="gelu"
        )
        hp_pre_out_activation = hp.Choice(
            "pre_out_activation", ["gelu", "elu", "relu", "swish"], default="elu"
        )
        hp_norm = hp.Boolean("norm", default=False)
        hp_dropout = hp.Float("dropout", 0.0, 0.5, default=0.0)
        hp_optimizer = hp.Choice("optimizer", ["SGD", "Adam"], default="Adam")

        channel_dims = [hp_channel_width] * hp_depth
        kernels = [hp_kernel_width] * hp_depth

        # Initialize special layers
        norm_layer = self.get_normalization_layer(hp_norm)
        if len(channel_dims) != len(kernels):
            print(
                f"[WARNING] Length of channel_dims and kernels does not match. Using 1st argument in kernels, {kernels[0]}, for every layer"
            )
            kernels = [kernels[0]] * len(channel_dims)

        # Initialize model architecture
        input_layer = keras.Input(shape=in_shape)
        x = input_layer  # Set aside input layer
        previous_block_activation = x  # Set aside residual
        for filters, kernel_size in zip(channel_dims, kernels):
            # First conv layer in block
            # 'same' applies zero padding.
            x = Conv1D(filters=filters, kernel_size=kernel_size, padding="same")(x)
            # todo: add se_block
            if norm_layer:
                x = norm_layer(x)
            x = keras.layers.Activation(hp_activation)(x)
            x = keras.layers.Dropout(hp_dropout)(x)

            # Second convolution layer
            x = Conv1D(filters=filters, kernel_size=kernel_size, padding="same")(x)
            if norm_layer:
                x = norm_layer(x)
            x = keras.layers.Activation(hp_activation)(x)
            x = keras.layers.Dropout(hp_dropout)(x)

            # Project residual
            residual = Conv1D(filters=filters, kernel_size=1, strides=1, padding="same")(
                previous_block_activation
            )
            x = keras.layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Output layers.
        # x = keras.layers.Dense(filters[-1], activation='gelu')(x) # Add another last layer.
        x = Conv1D(
            out_shape[-1], kernel_size=1, activation=hp_pre_out_activation, padding="same"
        )(x)
        # Assume that vertically resolved variables follow no particular range.
        output_lin = keras.layers.Dense(output_length_lin, activation="linear")(x)
        # Assume that all globally resolved variables are positive.
        output_relu = keras.layers.Dense(output_length_relu, activation="relu")(x)
        output_layer = keras.layers.Concatenate()([output_lin, output_relu])

        model = keras.Model(input_layer, output_layer, name="cnn")

        # Optimizer
        # Set up cyclic learning rate
        INIT_LR = 1e-4
        MAX_LR = 1e-3
        steps_per_epoch = 10091520 // hp_depth
        clr = tfa.optimizers.CyclicalLearningRate(
            initial_learning_rate=INIT_LR,
            maximal_learning_rate=MAX_LR,
            scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
            step_size=2 * steps_per_epoch,
            scale_mode="cycle",
        )

        # Set up optimizer
        if hp_optimizer == "Adam":
            my_optimizer = keras.optimizers.Adam(learning_rate=clr)
        elif hp_optimizer == "SGD":
            my_optimizer = keras.optimizers.SGD(learning_rate=clr)

        # compile
        model.compile(
            optimizer=my_optimizer,
            loss="mse",
            metrics=["mse", "mae", "accuracy"],
        )

        print(model.summary())

        return model


    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]),
            **kwargs,
        )

    def get_normalization_layer(self, norm=None, axis=[1, 2]):
        """
        Return normalization layer given string
        Args:
            norm string
            axis indices for layer normalization. todo: don't hard-code
        """
        if norm == "layer_norm":
            norm_layer = tf.keras.layers.LayerNormalization(axis=axis)
        elif norm == "batch_norm":
            norm_layer = tf.keras.layers.BatchNormalization()
        else:
            norm_layer = None
        return norm_layer


def main():
    training_data_path = (
        "/ocean/projects/atm200007p/jlin96/neurips_proj/e3sm_train_npy/"
    )

    if os.environ["KERASTUNER_TUNER_ID"] != "chief":
        with open(training_data_path + "train_input_cnn.npy", "rb") as f:
            train_input = np.load(f)
            # train_input.shape returns (10091520, 60, 6)
        with open(training_data_path + "train_target_cnn.npy", "rb") as f:
            train_target = np.load(f)
            # train_target.shape returns (10091520, 60, 10)
        with open(training_data_path + "val_input_cnn.npy", "rb") as f:
            val_input = np.load(f)
            # val_input.shape returns (1441920, 60, 6)
        with open(training_data_path + "val_target_cnn.npy", "rb") as f:
            val_target = np.load(f)
            # val_target.shape returns (1441920, 10)
    tuner = kt.Hyperband(
        hypermodel=CNNHyperModel(),
        objective="val_loss",
        max_trials=10,
        executions_per_trial=1,
        overwrite=False,
        directory="results",
        project_name="bair",
    )

    kwargs = {
        "epochs": 100,
        "verbose": 2,
        "shuffle": True,
        "callbacks": [
            callbacks.EarlyStopping("val_loss", patience=10),
            callbacks.TensorBoard(
                log_dir="logs/bair/logs_tensorboard", histogram_freq=1
            ),
            callbacks.BackupAndRestore("results/bair_backup"),
        ],
    }
    # print("---SEARCH SPACE---")
    # tuner.search_space_summary()

    # search
    if os.environ["KERASTUNER_TUNER_ID"] != "chief":
        tuner.search(
            train_input, train_target, validation_data=(val_input, val_target), **kwargs
        )


if __name__ == "__main__":
    # command line argument

    # assign GPUs for workers
    # gpus_per_node = 4 # NERSC Perlmutter
    # ntasks = int(os.environ['SLURM_NTASKS']) # "total number of workers" + 1 (for oracle)
    # nnodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    # workers_per_node = int((ntasks - 1) / nnodes)
    # workers_per_gpu  = int(workers_per_node / gpus_per_node)
    set_environment(num_gpus_per_node=4, oracle_port="8000")

    # limit memory preallocation
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(
        physical_devices[0], True
    )  # only using a single GPU per trial

    # query available GPU (as debugging info only)
    print(tf.config.list_physical_devices("GPU"))

    # run main program
    main()
