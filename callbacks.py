import tensorflow as tf

class PowerCallback(tf.keras.callbacks.Callback):
    """This callback starts an energy profiler at the beginning of a specific epoch and stops it
       when the training is done
    """
    def __init__(self, power_profiler, epoch_start):
        """
        Args:
            power_profiler: See energy_utils.
            epoch_start (int): Which epoch index to start the power profiler
        """
        self.power_profiler = power_profiler
        self.epoch_start = epoch_start

    def on_epoch_begin(self, epoch, logs={}):
        print(f"CALLBACK {epoch}")
        if epoch == self.epoch_start:
            print(f"CALLBACK START")
            # self.power_profiler.start()
            self.power_profiler.reset()

    def on_train_end(self, logs={}):
        print(f"CALLBACK stop")
        self.power_profiler.stop()
