import argparse
import numpy as np
import os
import psutil
import resource
import time
import socket

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, TextIO, Union
from .ap1_utils import Dnn, Unet
from .models import build_cnn, build_fullcnn, build_rnn, load_model
from climetlab_maelstrom_radiation.benchmarks import losses
from tensorflow.keras.optimizers import Adam


APPLICATONS_DICT={
    'AP1' : AP1_norway_forecast,
    'AP3': AP3_radiation_emulator
}

class ApplicationTemplate(ABC):
    """TODO"""

    def __init(
        self,
        args,
        num_processes,
        with_horovod
    ):
        self.args = args
        self.num_processes=num_processes
        self.with_horovod=with_horovod

    def get_dataset(self):
        return

    def get_model(self):
        return

    def get_loss(self):
        return 

    def get_optimizer(self):
        return

    def get_callbacks(self):
        return

class AP1_norway_forecast(ApplicationTemplate):

    def __init(
        self,
    ):
        super().__init__(self)


    @property
    def target_shape(self):
        return [1, self.args.patch_size, self.args.patch_size, 1]


    @property
    def pred_shape(self):
        num_predictors = 17
        return [1, selg.args.patch_size, selg.args.patch_size, num_predictors]

    def get_dataset(num_batches):
        """ Creates a tf dataset with specified sizes
        Args:
            pred_shape (list): Shape of predictors (for a single sample)
            target_shape (list): Shape of targets (for a single sample)
            num_batches (int): Number of batches in the dataset
            batch_size (int): Number of samples in one batch

        Returns:
            tf.data.Dataset
        """
        def get_generator(pred_shape, target_shape, num_samples):
            # device = "CPU:0"
            # with tf.device(device):
            def gen():
                    for i in range(num_samples):
                        pred = tf.random.uniform(pred_shape, dtype=tf.float32)
                        target = tf.random.uniform(target_shape, dtype=tf.float32)
                        yield pred, target
            return gen

        output_signature = (tf.TensorSpec(shape=self.pred_shape, dtype=tf.float32), tf.TensorSpec(shape=self.target_shape, dtype=tf.float32))
        dataset = tf.data.Dataset.from_generator(get_generator(self.pred_shape, self.target_shape, int(num_batches * self.args.batch_size)), output_signature=output_signature)

        # drop_remainder needed for IPU:
        dataset = dataset.batch(self.args.batch_size, drop_remainder=True)
        return dataset

    def get_model(self):
        num_outputs = 3
        if self.args.model == "unet":
            return Unet(self.pred_shape, num_outputs)
        elif self.args.model == "dnn":
            return Dnn(self.pred_shape, num_outputs)
        else:
            raise NotImplementedError()

    def get_loss(self):
        return quantile_score

    def get_optimizer(self):
        learning_rate = 1.0e-5   
        return Adam(learning_rate) 

    def get_callbacks(self,callbacks):
        if self.with_horovod:
            callbacks += [hvd.keras.callbacks.BroadcastGlobalVariablesCallback(0)]
            callbacks += [hvd.keras.callbacks.MetricAverageCallback()]             
        return callbacks

class AP3_radiation_emulator(ApplicationTemplate):

    def __init(
        self,
    ):
        super().__init__(self)

    def get_dataset(self):
        #! TODO
        # RETURN EQUIVALENT TO function load_data()
        # input_spec
        # https://git.ecmwf.int/projects/MLFET/repos/maelstrom-radiation/browse/climetlab_maelstrom_radiation/benchmarks/data.py
        # RETURN A DATASET! 
        return


    @property
    def target_shape(self):
        #! TODO
        # RETURN THE EQUIVALENT TO RETURN A train.element_spec[1]
        return


    @property
    def pred_shape(self):
        #! TODO
        # RETURN THE EQUIVALENT TO RETURN A train.element_spec[0]
        return 

    def get_model(self):
        dl_test=False
        attention=False
        if self.args.model == "min_cnn":
            model = build_cnn(
                self.pred_shape,
                self.target_shape,
                dl_test=dl_test
            )
        elif self.args.model == "rnn":
            model = build_rnn(
                self.pred_shape,
                self.target_shape,
                dl_test=dl_test
            )
        elif self.args.model == "cnn":
            model = build_fullcnn(
                self.pred_shape,
                self.target_shape,
                attention=attention,
                dl_test=dl_test
            )
        else:
            assert False, f"{self.args.model} not configured"

        return model

    def get_loss(self):
        if self.args.model == "min_cnn":
            weights = {"hr_sw": 10 ** (3), "sw": 1}
            loss = {"hr_sw": "mse", "sw": "mse"}
        elif self.args.model == "rnn":
            weights = {"hr_sw": 10 ** (-1), "sw": 1}
            loss = {"hr_sw": "mae", "sw": losses.top_scaledflux_mae}
        elif self.args.model == "cnn":
            weights = {"hr_sw": 10 ** (-1), "sw": 1}
            loss = {"hr_sw": "mae", "sw": losses.top_scaledflux_mae}
        else:
            assert False, f"{self.args.model} not configured"

        return loss, weights

    def get_optimizer(self):

        if self.args.model == "min_cnn":
            lr = 10 ** (-4)
        elif self.args.model == "rnn":
            lr = 0.5 * 10 ** (-3)
            if self.with_horovod:
                if hvd.size() == 4: #! TODO CHECK IF THIS PROBLEMATIC
                    lr = lr / 2
        elif self.args.model == "cnn":
            lr = 2 * 10 ** (-4)
        else:
            assert False, f"{self.args.model} not configured"

        true_lr = lr * self.args.batch_size / 256 * self.num_processes 
        opt = Adam(true_lr)
        return opt

    def get_callbacks(self,callbacks):
        callbacks += [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.25, patience=4, verbose=1, min_lr=10 ** (-6)
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=6,
                verbose=2,
                mode="auto",
                restore_best_weights=True,
            ),
        ]
        if self.with_horovod:
            callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
            callbacks.append(hvd.callbacks.MetricAverageCallback())

        return callbacks

        