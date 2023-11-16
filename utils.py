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

class NoStrategy:
    def scope(self):
        class Scope:
            def __enter__(self):
                pass
            def __exit__(self, exc_type, exc_value, exc_traceback):
                pass
        return Scope()

def check_horovod():
    """Check if we should run with horovod based on environment variables

    Returns:
        bool: True if we should run with horovod, False otherwise
    """
    # Program is run with horovodrun
    with_horovod = "HOROVOD_RANK" in os.environ

    if not with_horovod:
        # Program is run with srun
        with_horovod = "SLURM_STEP_NUM_TASKS" in os.environ and int(os.environ["SLURM_STEP_NUM_TASKS"]) > 1

    return with_horovod


def set_gpu_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


def print_gpu_usage(message="", show_line=False):
    try:
        usage = tf.config.experimental.get_memory_info("GPU:0")
        output = message + ' - '.join([f"{k}: {v / 1024**3:.2f} GB" for k,v in usage.items()])
    except ValueError as e:
        output = message + ' None'

    if show_line:
        frameinfo = inspect.getouterframes(inspect.currentframe())[1]
        output += " (%s:%s)" % (frameinfo.filename, frameinfo.lineno)

    print(output)

def print_cpu_usage(message="", show_line=False):
    """Prints the current and maximum memory useage of this process
    Args:
        message (str): Prepend with this message
        show_line (bool): Add the file and line number making this call at the end of message
    """

    output = "current: %.2f GB - peak: %.2f GB" % (
        get_memory_usage() / 1024 ** 3,
        get_max_memory_usage() / 1024 ** 3,
    )
    output = message + output
    if show_line:
        frameinfo = inspect.getouterframes(inspect.currentframe())[1]
        output += " (%s:%s)" % (frameinfo.filename, frameinfo.lineno)

    print(output)

def get_memory_usage():
    p = psutil.Process(os.getpid())
    mem = p.memory_info().rss
    for child in p.children(recursive=True):
        mem += child.memory_info().rss
    return mem


def get_max_memory_usage():
    """In bytes"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1000



class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        self.s_time = time.time()
        self.start_times = dict()
        self.end_times = dict()

    # def on_train_batch_begin(self, batch, logs = {}):
        # print("Current batch", batch)

    def on_epoch_begin(self, epoch, logs = {}):
        self.start_times[epoch] = time.time()

    def on_epoch_end(self,epoch,logs = {}):
        self.end_times[epoch] = time.time()
        self.times.append(time.time() - self.s_time)

    def get_epoch_times(self):
        times = list()
        keys = list(self.start_times.keys())
        keys.sort()
        for key in keys:
            if key not in self.end_times:
                print(f"WARNING: Did not find epoch={key} in end times")
                continue
            times += [self.end_times[key] - self.start_times[key]]
        return times

    # def on_train_end(self,logs = {}):