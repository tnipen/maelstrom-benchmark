from tensorflow import keras
import horovod.tensorflow as hvd
import tensorflow as tf

import losses
import models


def get(name, patch_size):
    """Returns an initialized application"""
    if name == "ap1":
        return AP1(patch_size=patch_size)
    else:
        raise NotImplementedError()

class Application:
    """Abstract class representing a MAELSTROM application"""
    def get_model(self):
        """Returns a keras.Model object"""
        raise NotImplementedError()

    @property
    def input_shape(self):
        """Returns the shape of predictors for a single sample (list)"""
        raise NotImplementedError()

    @property
    def target_shape(self):
        """Returns the shape of targets for a single sample (list)"""
        raise NotImplementedError()

    def get_loss_function(self):
        """Returns the application's loss function handle. Must be a function
        that takes two inputs (y_true, y_pred)"""
        raise NotImplementedError()

    def get_optimizer(self, with_horovod=False):
        """Returns the optimizer for this application

        Applications can override this, if needed.
        """
        learning_rate = 1.0e-5  # Doesn't matter for this benchmark
        optimizer = keras.optimizers.Adam(learning_rate)
        if with_horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, backward_passes_per_step=1,
                    average_aggregated_gradients=True)
        return optimizer

    def get_dataset(self, num_batches, batch_size):
        """ Creates a tf dataset with specified sizes
        Args:
            num_batches (int): Number of batches in the dataset
            batch_size (int): Number of samples in one batch

        Returns:
            tf.data.Dataset
        """
        def get_generator(input_shape, target_shape, num_samples):
            # device = "CPU:0"
            # with tf.device(device):
            def gen():
                    for i in range(num_samples):
                        pred = tf.random.uniform(input_shape, dtype=tf.float32)
                        target = tf.random.uniform(target_shape, dtype=tf.float32)
                        yield pred, target
            return gen

        output_signature = (tf.TensorSpec(shape=self.input_shape, dtype=tf.float32), tf.TensorSpec(shape=self.target_shape, dtype=tf.float32))
        dataset = tf.data.Dataset.from_generator(get_generator(self.input_shape, self.target_shape, int(num_batches * batch_size)), output_signature=output_signature)

        # drop_remainder needed for IPU:
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset

class AP1(Application):
    def __init__(self,
        features=16,
        levels=3,
        pool_size=2,
        conv_size=3,
        upsampling_type="conv_transpose",
        num_predictors=17,
        patch_size=128,
        num_outputs=3,
        
        ):
        self.features = features
        self.levels = levels
        self.pool_size = pool_size
        self.conv_size = conv_size
        self.upsampling_type = upsampling_type
        self.num_predictors = num_predictors
        self.patch_size = patch_size
        self.num_outputs = num_outputs

    def get_model(self):
        return models.Unet(self.input_shape, self.num_outputs, self.features, self.levels, self.pool_size, self.conv_size, self.upsampling_type)

    @property
    def input_shape(self):
        shape = [1, self.patch_size, self.patch_size, self.num_predictors]
        return shape

    @property
    def target_shape(self):
        shape = [1, self.patch_size, self.patch_size, 1]
        return shape
    
    def get_loss_function(self):
        return losses.quantile_score
