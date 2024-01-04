from tensorflow import keras
import horovod.tensorflow as hvd
import tensorflow as tf

import losses
import models


def get(name, patch_size):
    """Returns an initialized application"""
    if name == "ap1":
        return AP1(patch_size=patch_size)
    elif name == "ap5":
        return AP5(patch_size=patch_size)
    else:
        raise NotImplementedError(f"Unknown application {name}")

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


class AP5(Application):
    """
    Application 5: WGAN for statistical downscaling of ERA5 data to COSMO-REA6
    """
    def __init__(self, patch_size: tuple = (120, 96), num_predictors: int = 15, ntargets: int = 1, 
            hparams: dict = {"batch_size": None, "generator": {"l_avgpool": False, "activation": "swish", "z_branch": True}, 
                             "critic": {"activation": "swish"}}):
        """
        :param patch_size: size of the input patch
        :param num_predictors: number of predictors
        :param ntargets: number of 'dynamic' targets (excl. z)
        :param hparams: hyperparameters
        """
        self.patch_size = patch_size
        self.num_predictors = num_predictors
        self.hparams = hparams
        self.ntargets = ntargets +1 if self.hparams["generator"].get("z_branch", False) else ntargets    # +1 for z

        # to be set in get_dataset
        self.batch_size = None


    def get_model(self):
        assert self.batch_size is not None, "batch_size must be set before calling get_model, i.e. run get_dataset first."

        self.hparams["batch_size"] = self.batch_size

        sha_unet = models.Sha_Unet(self.input_shape, self.hparams["generator"], self.ntargets, concat_out=True)
        critic = models.Critic((*self.input_shape[:-1], 1), self.hparams["critic"])

        return models.WGAN(sha_unet, critic, self.hparams)
    
    @property
    def input_shape(self):
        shape = [*self.patch_size, self.num_predictors]
        return shape

    @property
    def target_shape(self):
        shape = [*self.patch_size, self.ntargets]
        return shape
    
    def get_optimizer(self, with_horovod=False):
        """
        Returns the optimizers in a dictionary for the WGAN of AP5.
        """
        learning_rate = 1.0e-5  # Doesn't matter for this benchmark
        optimizers = {}
        optimizers["c_optimizer"] = keras.optimizers.Adam(learning_rate)
        optimizers["g_optimizer"] = keras.optimizers.Adam(learning_rate)
        if with_horovod:
            for optimizer_model, optimizer in optimizers.items():
                optimizers[optimizer_model] = hvd.DistributedOptimizer(optimizer, backward_passes_per_step=1,
                                                                       average_aggregated_gradients=True)
        return optimizers

    def get_loss_function(self):
        """
        Returns the loss functions in a dictionary for the WGAN of AP5.
        """
        loss_dict = {}
        loss_dict["recon_loss"] = self.recon_loss
        loss_dict["critic_loss"] = self.critic_loss
        loss_dict["critic_gen_loss"] = self.critic_gen_loss

        return loss_dict
    
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
        
        # effective batch-size during training of WGAN (with d_steps)
        self.batch_size = batch_size
        batch_size_eff = batch_size * self.hparams.get("d_steps", 5)

        output_signature = (tf.TensorSpec(shape=self.input_shape, dtype=tf.float32), tf.TensorSpec(shape=self.target_shape, dtype=tf.float32))
        dataset = tf.data.Dataset.from_generator(get_generator(self.input_shape, self.target_shape, int(num_batches * batch_size_eff)), output_signature=output_signature)

        # drop_remainder needed for IPU:
        dataset = dataset.batch(batch_size_eff, drop_remainder=True)
        return dataset

    def recon_loss(self, real_data, gen_data):
        """
        Reconstruction loss in WGAN (L1 loss).
        :param real_data: real data
        :param gen_data: generated data
        :return rloss: reconstruction loss
        """
        rloss = 0.
        for i in range(self.ntargets):
            rloss += tf.reduce_mean(tf.abs(tf.squeeze(gen_data[..., i]) - real_data[..., i]))
        return rloss

    @staticmethod
    def critic_loss(critic_real, critic_gen):
        """
        The critic is optimized to maximize the difference between the generated and the real data max(real - gen).
        This is equivalent to minimizing the negative of this difference, i.e. min(gen - real) = max(real - gen)
        :param critic_real: critic on the real data
        :param critic_gen: critic on the generated data
        :return c_loss: loss to optimize the critic
        """
        c_loss = tf.reduce_mean(critic_gen - critic_real)
        return c_loss

    @staticmethod
    def critic_gen_loss(critic_gen):
        """
        The generator is optimized to minimize the critic on the generated data.
        :param critic_gen: critic on the generated data
        :return cg_loss: loss to optimize the generator
        """
        cg_loss = -tf.reduce_mean(critic_gen)
        return cg_loss
