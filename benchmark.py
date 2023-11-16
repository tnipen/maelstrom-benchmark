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
from utils import check_horovod, set_gpu_memory_growth,NoStrategy, TimingCallback,print_cpu_usage,print_gpu_usage

from applications import APPLICATIONS_DICT

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
    parser.add_argument('-m', default="unet", help="Model", dest="model", required=True) #choices=["unet", "dnn"],)
    parser.add_argument('-p', default=128, type=int, help='Patch size', dest="patch_size",required=False) 
    parser.add_argument('-w', help='What hardware to run this on', dest='hardware', choices=["gpu", "cpu", "ipu"], required=True)
    parser.add_argument('-s', type=int, help='Steps per execution (for IPU)', dest='steps_per_execution')
    parser.add_argument('-r', default=1, type=int, help='Replica (for IPU)', dest='replica')
    parser.add_argument('--app', default=1, type=int, help='Replica (for IPU)', dest='application')
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


    application=args.application
    application_instance = APPLICATIONS_DICT[application](args,num_processes,with_horovod)    

    if application != 3:
        loss = application_instance.get_loss()
    else:
        loss,loss_weigths = application_instance.get_loss()

    # Settings
    batch_size_mb = application_instance.get_batch_size_mb()

    steps_per_epoch = int(args.dataset_size / 4 / application_instance.batch_bytes / args.batch_size / num_processes)
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

    num_batches = steps_per_epoch * args.epochs
    dataset = application_instance.get_dataset(num_batches)
    print(num_batches)
    dataset_size_mb = batch_size_mb * steps_per_epoch * num_processes

    with strategy.scope():
        model = application_instance.get_model()
         # Doesn't matter for this benchmark
        optimizer = application_instance.get_optimizer()
        if with_horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, backward_passes_per_step=1,
                    average_aggregated_gradients=True)
        if args.hardware == "ipu":
            if application != 3 :
                model.compile(optimizer=optimizer, loss=loss, steps_per_execution=steps_per_execution)
            else:
                model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weigths,steps_per_execution=steps_per_execution)
        else:
            if application != 3 :
                model.compile(optimizer=optimizer, loss=loss)
            else:
                model.compile(optimizer=optimizer, loss=loss,loss_weights=loss_weigths)


        # Print out dataset
        # for k,v in dataset:
        #     print(k.shape, v.shape)

        callbacks = list()
        callbacks = application_instance.get_callbacks(callbacks)
        if main_process:
            timing_callback = TimingCallback()
            callbacks += [timing_callback]

        # Train the model
        start_time = time.time()
        history = model.fit(dataset, epochs=args.epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks, verbose=main_process)
        #history = model.fit(dataset, epochs=args.epochs, callbacks=callbacks, verbose=main_process)
        training_time = time.time() - start_time

    # Write out results
    if main_process:
        times = timing_callback.get_epoch_times()
        num_trainable_weights = int(np.sum([K.count_params(w) for w in model.trainable_weights]))
        hostname = socket.gethostname().split('.')[0]
        print("Benchmark stats:")
        print(f"   Hardware: ", args.hardware.upper())
        print(f"   Hostname: {hostname}")
        print(f"   Pred sample shape: {application_instance.pred_shape}")
        print(f"   Target sample shape: {application_instance.target_shape}")
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



if __name__ == "__main__":
    main()
