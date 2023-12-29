import argparse
import numpy as np
import os
import socket
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras.backend as K
from utils import check_horovod, set_gpu_memory_growth,NoStrategy, TimingCallback,print_cpu_usage,print_gpu_usage
import energy_utils 
import applications
import sys


"""This benchmark scripts checks training performance on GPU, CPU, and IPU"""
def main():
    parser = argparse.ArgumentParser("Program to compare the performance of IPUs and GPUs")
    parser.add_argument('-a', default="ap1", help="Run this model", dest="app_name")
    parser.add_argument('-e', default=3, type=int, help='Number of epochs', dest="epochs")
    parser.add_argument('-d', type=float, help='Dataset size in bytes', dest="dataset_size", required=True)
    parser.add_argument('-b', default=1, type=int, help="Batch size (number of samples per batch)", dest="batch_size")
    parser.add_argument('-w', help='What hardware to run this on', dest='hardware', choices=["gpu", "cpu", "ipu"], required=True)
    parser.add_argument('-wn', help='Hardware name', dest='hardware_name', choices=["A100_GPU", "V100_GPU", "GC200_IPU", "H100_GPU","MI250_GPU"], required=True)
    parser.add_argument('--debug', help='Turn on debugging information', action="store_true")

    # IPU specific arguments
    parser.add_argument('-s', type=int, help='Steps per execution (for IPU)', dest='steps_per_execution')
    parser.add_argument('-r', default=1, type=int, help='Replica (for IPU)', dest='replica')

    # AP1 specific arguments
    parser.add_argument('-p', default=128, type=int, help='Patch size', dest="patch_size")
    parser.add_argument('-g', default=10, type=int, help='Gradient accumluation steps (IPU only)', dest='gradient_accumulation_steps')
    args = parser.parse_args()
    
    if args.hardware_name=='MI250_GPU':
        sys.path.append('/opt/rocm/libexec/rocm_smi')

    
    energy_profiler = energy_utils.get_energy_profiler(args.hardware_name)
    
    with energy_profiler() as measured_scope:
        print('Measuring Energy during main() call')
        try:

            main_process = True
            num_processes = 1
            with_horovod = check_horovod()    
            if with_horovod:
                import horovod.tensorflow as hvd
            print(f"Running with horovod? {with_horovod}")

            if args.hardware == "cpu":
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                strategy = NoStrategy()

            elif args.hardware == "ipu":
                import gcipuinfo
                ipu_info = gcipuinfo.gcipuinfo()
                num_devices = len(ipu_info.getDevices())
                print('devices detected', num_devices)
                if num_devices == 0:
                    print("gc_power_consumption.py: error: no IPUs detected", file=sys.stderr)
                    exit(-1)
                
                from tensorflow.python import ipu
                ipu_config = ipu.config.IPUConfig()
                ipu_config.device_connection.type = (
                            ipu.config.DeviceConnectionType.ON_DEMAND
                            )# Optional - allows parallel execution
                ipu_config.auto_select_ipus = args.replica
                # ipu_config.io_tiles.num_io_tiles = 128
                # ipu_config.io_tiles.place_ops_on_io_tiles = True
                ipu_config.configure_ipu_system()

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

            app = applications.get(args.app_name,args,num_processes,with_horovod)

            # Settings
            batch_size_mb = app.get_batch_size_mb()

            steps_per_epoch = int(args.dataset_size / 4 / app.batch_bytes / args.batch_size / num_processes)

            steps_per_epoch = int(steps_per_epoch // (args.gradient_accumulation_steps * args.replica) * args.gradient_accumulation_steps * args.replica)

            # Adjust steps_per_execution so that it is a multiple of steps_per_execution * replicas
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

            num_batches=steps_per_epoch * args.epochs
            dataset = app.get_dataset(num_batches)
            dataset_size_mb = batch_size_mb * steps_per_epoch * num_processes
            effective_batch_size_mb = batch_size_mb * num_processes * args.replica * args.gradient_accumulation_steps

            with strategy.scope():

                model = app.get_model()
                optimizer = app.get_optimizer()

                if args.app_name != 'ap3':
                    loss = app.get_loss_function()
                else:
                    loss,loss_weights = app.get_loss_function()

                if args.hardware == "ipu":
                    # optimizer = ipu.optimizers.CrossReplicaGradientAccumulationOptimizerV2(optimizer, 10)
                    if args.app_name != 'ap3' :
                        model.compile(optimizer=optimizer, loss=loss, steps_per_execution=steps_per_execution)
                    else:
                        model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights,steps_per_execution=steps_per_execution)
                    model.set_gradient_accumulation_options(gradient_accumulation_steps_per_replica=args.gradient_accumulation_steps)
                else:
                    if args.app_name != 'ap3' :
                        model.compile(optimizer=optimizer, loss=loss)
                    else:
                        model.compile(optimizer=optimizer, loss=loss,loss_weights=loss_weights) 

                # Print out dataset
                # for k,v in dataset:
                #     print(k.shape, v.shape)

                callbacks = list()
                callbacks = app.get_callbacks(callbacks)
                if main_process:
                    timing_callback = TimingCallback()
                    callbacks += [timing_callback]


                # Train the model
                start_time = time.time()
                history = model.fit(dataset, epochs=args.epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks, verbose=1)
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
                print(f"   Effective batch size: {effective_batch_size_mb:.2f} MB")
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

                
        except Exception as exc:
            import traceback
            print(f"Errors occured during training: {exc}")
            print(f"Traceback: {traceback.format_exc()}")
                

    f = open(f"EnergyFile-NVDA-integrated.txt", 'a')
    #print("Energy data:")
    measured_scope.df.to_csv("EnergyFile-NVDA.csv") 
    print("Energy-per-GPU-list:")
    max_power=measured_scope.df.loc[:,(measured_scope.df.columns != 'timestamps')].max().max()
    print(f"Max Power: {max_power:.2f} W")
    
    if args.hardware_name=='MI250_GPU':
        energy_int,energy_cnt = measured_scope.energy()
        print(f"Integrated Total Energy: {np.sum(energy_int):.2f} J")
        print(f"Counter Total Energy: {np.sum(energy_cnt):.2f} J")
        
        f.write(f"integrated: {energy_int}") 
        f.write(f"from counter: {energy_cnt}")
        f.close()
    elif args.hardware_name in ['A100_GPU','H100_GPU','GC200_IPU']:
        energy_int = measured_scope.energy() 
        print(f"Integrated Total Energy: {np.sum(energy_int):.2f} J")
        f.write(f"integrated: {energy_int}") 
        f.close()
        
        
        
if __name__ == "__main__":
    

    
    main()