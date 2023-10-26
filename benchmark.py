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

with_horovod = check_horovod()
if with_horovod:
    # Import it
    print("Running with horovod")
    import horovod.tensorflow as hvd

"""This benchmark scripts checks training performance on GPU, CPU, and IPU"""
def main():
    parser = argparse.ArgumentParser("Program to compare the performance of IPUs and GPUs")
    parser.add_argument('-b', default=1, type=int, help="Batch size (number of samples per batch)", dest="batch_size")
    parser.add_argument('-e', default=3, type=int, help='Number of epochs', dest="epochs")
    # parser.add_argument('-s', default=10, type=int, help='Number of batches per epoch', dest="steps_per_epoch")
    parser.add_argument('-d', type=float, help='Dataset size in bytes', dest="dataset_size")
    parser.add_argument('-m', default="unet", help="Model", dest="model", choices=["unet", "dnn"])
    parser.add_argument('-p', default=128, type=int, help='Patch size', dest="patch_size")
    parser.add_argument('-w', help='What hardware to run this on', dest='hardware', choices=["gpu", "cpu", "ipu"], required=True)
    parser.add_argument('-s', type=int, help='Steps per execution (for IPU)', dest='steps_per_execution')
    parser.add_argument('-r', default=1, type=int, help='Replica (for IPU)', dest='replica')
    parser.add_argument('--debug', help='Turn on debugging information', action="store_true")
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

    # Settings
    patch_size = args.patch_size
    num_predictors = 17
    pred_shape = [1, patch_size, patch_size, num_predictors]
    target_shape = [1, patch_size, patch_size, 1]
    loss = quantile_score
    num_outputs = 3
    batch_size_mb = 4 * np.product(pred_shape) * args.batch_size / 1024 / 1024

    steps_per_epoch = int(args.dataset_size / 4 / np.product(pred_shape) / args.batch_size / num_processes)
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
    dataset = get_dataset(pred_shape, target_shape, steps_per_epoch * args.epochs, args.batch_size)
    dataset_size_mb = batch_size_mb * steps_per_epoch * num_processes

    with strategy.scope():
        model = get_model(args.model, pred_shape, num_outputs)
        learning_rate = 1.0e-5  # Doesn't matter for this benchmark
        optimizer = keras.optimizers.Adam(learning_rate)
        if with_horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, backward_passes_per_step=1,
                    average_aggregated_gradients=True)
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
        print(f"   Hardware: ", args.hardware.upper())
        print(f"   Hostname: {hostname}")
        print(f"   Pred sample shape: {pred_shape}")
        print(f"   Target sample shape: {target_shape}")
        print(f"   Num epochs: ", args.epochs)
        print(f"   Samples per batch: ", args.batch_size)
        print(f"   Dataset size: {dataset_size_mb:.2f} MB")
        print(f"   Steps per execution: {steps_per_execution}")
        print(f"   Num replicas: {args.replica}")
        print(f"   Batches per epoch: {steps_per_epoch}")
        print(f"   Batch size: {batch_size_mb:.2f} MB")
        print(f"   Model: {args.model.upper()}")
        print(f"   Num trainable weights: {num_trainable_weights}")
        print(f"   Num processes: {num_processes}")
        print("Training performance:")
        print(f"   Total training time: {training_time:.2f} s")
        print(f"   Average performance: {dataset_size_mb / training_time * args.epochs:.2f} MB/s")
        print(f"   First epoch time: {times[0]:.2f} s")
        print(f"   Min epoch time: {np.min(times):.2f} s")
        print(f"   Performance min epoch: {dataset_size_mb / np.min(times):.2f} MB/s")
        print(f"   Mean epoch time: {np.mean(times):.2f} s")
        print(f"   Performance mean epoch: {dataset_size_mb / np.mean(times):.2f} MB/s")
        print(f"   Max epoch time: {np.max(times):.2f} s")
        print(f"   Performance max epoch: {dataset_size_mb / np.max(times):.2f} MB/s")
        print_gpu_usage("   GPU memory: ")
        print_cpu_usage("   CPU memory: ")


"""
Data loader
"""
def get_dataset(pred_shape, target_shape, num_batches, batch_size):
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

    output_signature = (tf.TensorSpec(shape=pred_shape, dtype=tf.float32), tf.TensorSpec(shape=target_shape, dtype=tf.float32))
    dataset = tf.data.Dataset.from_generator(get_generator(pred_shape, target_shape, int(num_batches * batch_size)), output_signature=output_signature)

    # drop_remainder needed for IPU:
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

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

"""
ML model
"""

def get_model(name, input_shape, num_outputs, *args, **kwargs):
    if name == "unet":
        return Unet(input_shape, num_outputs, *args, **kwargs)
    elif name == "dnn":
        return Dnn(input_shape, num_outputs, *args, **kwargs)
    else:
        raise NotImplementedError()

class Dnn(keras.Model):
    def __init__(
        self,
        input_shape,
        num_outputs,
        features=16,
        levels=3
        ):
        self._num_outputs = num_outputs
        self._features = features
        self._levels = levels

        # Build the model
        inputs = keras.layers.Input(input_shape)
        outputs = inputs
        for i in range(levels):
            outputs = keras.layers.Dense(features, activation="tanh")(outputs)
        outputs = keras.layers.Dense(num_outputs)(outputs)

        super().__init__(inputs, outputs)

class Unet(keras.Model):
    def __init__(
        self,
        input_shape,
        num_outputs,
        features=16,
        levels=3,
        pool_size=2,
        conv_size=3,
        upsampling_type="conv_transpose",
    ):
        """U-net

        Args:
            features (int): Number of features in the first layer
            levels (int): Depth of the U-net
            pool_size (int): Pooling ratio (> 0)
            upsampling_type (str): One of "upsampling" or "conv_transpose"
            conv_size (int): Convolution size (> 0)
        """
        if upsampling_type not in ["upsampling", "conv_transpose"]:
            raise ValueError(f"Unknown upsampling type {upsampling_type}")

        # print(f"Initializing a U-Net with shape {input_shape}")

        self._num_outputs = num_outputs
        self._features = features
        self._levels = levels
        self._pool_size = pool_size
        self._conv_size = conv_size
        self._upsampling_type = upsampling_type

        # Build the model
        inputs = keras.layers.Input(input_shape)
        outputs = self.get_outputs(inputs)

        super().__init__(inputs, outputs)

    def get_outputs(self, inputs):
        outputs = inputs
        levels = list()

        features = self._features

        pool_size = [1, self._pool_size, self._pool_size]
        hood_size = [1, self._conv_size, self._conv_size]

        Conv = keras.layers.Conv3D
        if self._upsampling_type == "upsampling":
            UpConv = keras.layers.UpSampling3D
        elif self._upsampling_type == "conv_transpose":
            UpConv = keras.layers.Conv3DTranspose

        # Downsampling
        # conv -> conv -> max_pool
        for i in range(self._levels - 1):
            outputs = Conv(features, hood_size, activation="relu", padding="same")(
                outputs
            )
            outputs = Conv(features, hood_size, activation="relu", padding="same")(
                outputs
            )
            levels += [outputs]
            # print(i, outputs.shape)

            outputs = keras.layers.MaxPooling3D(pool_size=pool_size)(outputs)
            features *= 2

        # conv -> conv
        outputs = Conv(features, hood_size, activation="relu", padding="same")(
            outputs
        )
        outputs = Conv(features, hood_size, activation="relu", padding="same")(
            outputs
        )

        # upconv -> concat -> conv -> conv
        for i in range(self._levels - 2, -1, -1):
            features /= 2
            outputs = UpConv(features, hood_size, strides=pool_size, padding="same")(outputs)

            # print(levels[i].shape, outputs.shape)
            outputs = keras.layers.concatenate((levels[i], outputs), axis=-1)
            outputs = Conv(features, hood_size, activation="relu", padding="same")(
                outputs
            )
            outputs = Conv(features, hood_size, activation="relu", padding="same")(
                outputs
            )

        # Dense layer at the end
        outputs = keras.layers.Dense(self._num_outputs, activation="linear")(
            outputs
        )

        return outputs

"""
Loss function
"""
def quantile_score(y_true, y_pred):
    quantiles = [0.1, 0.5, 0.9]
    qtloss = 0
    for i, quantile in enumerate(quantiles):
        err = y_true[..., 0] - y_pred[..., i]
        qtloss += (quantile - tf.cast((err < 0), tf.float32)) * err
    return K.mean(qtloss) / len(quantiles)


class NoStrategy:
    def scope(self):
        class Scope:
            def __enter__(self):
                pass
            def __exit__(self, exc_type, exc_value, exc_traceback):
                pass
        return Scope()


def set_gpu_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":
    main()
