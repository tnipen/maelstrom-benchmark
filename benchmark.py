import argparse
import inspect
import numpy as np
import os
import psutil
import resource
import socket
import time
import horovod.tensorflow as hvd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

import applications

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

"""This benchmark scripts checks training performance on GPU, CPU, and IPU"""
def main():
    parser = argparse.ArgumentParser("Program to compare the performance of IPUs and GPUs")
    parser.add_argument('-a', default="ap1", help="Run this model", dest="app_name")
    parser.add_argument('-e', default=3, type=int, help='Number of epochs', dest="epochs")
    parser.add_argument('-d', type=float, help='Dataset size in bytes', dest="dataset_size", required=True)
    parser.add_argument('-b', default=1, type=int, help="Batch size (number of samples per batch)", dest="batch_size")
    parser.add_argument('-w', help='What hardware to run this on', dest='hardware', choices=["gpu", "cpu", "ipu"], required=True)
    parser.add_argument('--debug', help='Turn on debugging information', action="store_true")

    # IPU specific arguments
    parser.add_argument('-s', type=int, help='Steps per execution (for IPU)', dest='steps_per_execution')
    parser.add_argument('-r', default=1, type=int, help='Replica (for IPU)', dest='replica')

    # AP1 specific arguments
    parser.add_argument('-p', default=128, type=int, help='Patch size', dest="patch_size")
    parser.add_argument('-ps', '--patch_size', default=[96, 120], type=int, nargs="+", dest="patch_size_rect", help="Rectangular patch size")
    args = parser.parse_args()

    # strategy 1
    #   batch size (samples/batch)
    #   batches/epoch
    #   patch size
    # strategy 2
    #   patch size
    #   dataset size
    #   batch size

    main_process = True
    num_processes = 1
    with_horovod = check_horovod()
    print(f"Running with horovod? {with_horovod}")
    if args.hardware == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        strategy = NoStrategy()
    elif args.hardware == "ipu":
        from tensorflow.python import ipu
        ipu_config = ipu.config.IPUConfig()
        ipu_config.device_connection.type = (
                    ipu.config.DeviceConnectionType.ON_DEMAND
                    )  # Optional - allows parallel execution
        ipu_config.auto_select_ipus = args.replica
        ipu_config.configure_ipu_system()
        # ipu_config.io_tiles.num_io_tiles = 128
        # ipu_config.io_tiles.place_ops_on_io_tiles = True

        strategy = ipu.ipu_strategy.IPUStrategy()
    else:
        gpus = tf.config.experimental.list_physical_devices("GPU")

        set_gpu_memory_growth()
        if with_horovod:
            hvd.init()
            print(hvd.rank(), hvd.size())
            if len(gpus) == 0:
                raise Exception("No GPUs available")
            if len(gpus) > 1:
            # if hvd.size() == len(gpus):
                # Probably using horovodrun (not srun)
                tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
            main_process = hvd.rank() == 0
            num_processes = hvd.size()
        print("Num GPUs Available: ", len(gpus))
        strategy = NoStrategy()

    
    if args.app_name == "ap5":
        app = applications.get(args.app_name, args.patch_size_rect)
    else:
        app = applications.get(args.app_name, args.patch_size)

    # Settings
    print(app.input_shape)
    print(args.batch_size)
    batch_size_mb = 4 * np.product(app.input_shape) * args.batch_size / 1024 / 1024

    steps_per_epoch = int(args.dataset_size / 4 / np.product(app.input_shape) / args.batch_size / num_processes)
    # Adjust steps so that it is a multiple of steps_per_execution * replicas

    if args.steps_per_execution is None:
        steps_per_epoch = (steps_per_epoch // (args.replica)) * args.replica
        steps_per_execution = steps_per_epoch
    else:
        steps_per_execution = args.steps_per_execution
        if steps_per_execution * args.replica > steps_per_epoch:
            steps_per_execution = steps_per_epoch // args.replica
        steps_per_epoch = (steps_per_epoch // (steps_per_execution * args.replica)) * steps_per_execution * args.replica

    if main_process:
        print("steps_per_epoch:", steps_per_epoch)
        print("steps_per_execution:", steps_per_execution)

    dataset = app.get_dataset(steps_per_epoch * args.epochs, args.batch_size)
    dataset_size_mb = batch_size_mb * steps_per_epoch * num_processes

    with strategy.scope():
        model = app.get_model()
        optimizer = app.get_optimizer()
        loss = app.get_loss_function()
        if args.hardware == "ipu":
            model.compile(optimizer=optimizer, loss=loss, steps_per_execution=steps_per_execution)
        else:
            model.compile(optimizer=optimizer, loss=loss)

        # Print out dataset
        # for k,v in dataset:
        #     print(k.shape, v.shape)

        callbacks = list()
        if with_horovod:
            callbacks += [hvd.keras.callbacks.BroadcastGlobalVariablesCallback(0)]
            callbacks += [hvd.keras.callbacks.MetricAverageCallback()]
        if main_process:
            timing_callback = TimingCallback()
            callbacks += [timing_callback]

        # Train the model
        start_time = time.time()
        history = model.fit(dataset, epochs=args.epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks, verbose=main_process)
        training_time = time.time() - start_time

    # Write out results
    if main_process:
        times = timing_callback.get_epoch_times()
        num_trainable_weights = int(np.sum([K.count_params(w) for w in model.trainable_weights]))
        hostname = socket.gethostname().split('.')[0]
        print("Benchmark stats:")
        print(f"   Application: ", args.app_name)
        print(f"   Hardware: ", args.hardware.upper())
        print(f"   Hostname: {hostname}")
        print(f"   Pred sample shape: {app.input_shape}")
        print(f"   Target sample shape: {app.target_shape}")
        print(f"   Num epochs: ", args.epochs)
        print(f"   Samples per batch: ", args.batch_size)
        print(f"   Dataset size: {dataset_size_mb:.2f} MB")
        print(f"   Steps per execution: {steps_per_execution}")
        print(f"   Num replicas: {args.replica}")
        print(f"   Batches per epoch: {steps_per_epoch}")
        print(f"   Batch size: {batch_size_mb:.2f} MB")
        print(f"   Num trainable weights: {num_trainable_weights}")
        print(f"   Num processes: {num_processes}")
        print("Training performance:")
        print(f"   Total training time: {training_time:.2f} s")
        print(f"   Average performance: {dataset_size_mb / training_time * args.epochs:.2f} MB/s")
        print(f"   First epoch time: {times[0]:.2f} s")
        print(f"   Non-first epoch time: {np.mean(times[1:]):.2f} s")
        print(f"   Performance non-first epoch: {dataset_size_mb / np.mean(times[1:]):.2f} MB/s")
        print(f"   Min epoch time: {np.min(times):.2f} s")
        print(f"   Performance min epoch: {dataset_size_mb / np.min(times):.2f} MB/s")
        print(f"   Mean epoch time: {np.mean(times):.2f} s")
        print(f"   Performance mean epoch: {dataset_size_mb / np.mean(times):.2f} MB/s")
        print(f"   Max epoch time: {np.max(times):.2f} s")
        print(f"   Performance max epoch: {dataset_size_mb / np.max(times):.2f} MB/s")
        print_gpu_usage("   GPU memory: ")
        print_cpu_usage("   CPU memory: ")


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


def set_gpu_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

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


class NoStrategy:
    # My favourite kind of strategy...
    def scope(self):
        class Scope:
            def __enter__(self):
                pass
            def __exit__(self, exc_type, exc_value, exc_traceback):
                pass
        return Scope()


if __name__ == "__main__":
    main()
