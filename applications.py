from tensorflow import keras
import tensorflow as tf

import losses
import models
import numpy as np


def get(name, args,num_processes, with_horovod):
    
    """Returns an initialized application"""
    if name == "ap1":
        return AP1(args, num_processes, with_horovod)
    elif name == 'ap3':
        return AP3(args, num_processes, with_horovod)
    elif name == "ap5":
        return AP5(args, num_processes, with_horovod)
    else:
        raise NotImplementedError(f"Unknown application {name}")

class Application:
    """Abstract class representing a MAELSTROM application"""
    
    def __init__(
        self,
        args,
        num_processes,
        with_horovod
    ):
        self.args = args
        self.num_processes=num_processes
        self.with_horovod=with_horovod
    
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

    @property
    def batch_bytes(self):
        raise NotImplementedError()
    
    def get_batch_size_mb(self):
        return 4 * self.batch_bytes * self.args.batch_size / 1024 / 1024
        
    def get_loss_function(self):
        """Returns the application's loss function handle. Must be a function
        that takes two inputs (y_true, y_pred)"""
        raise NotImplementedError()
        
    def get_callbacks(self):
        """Returns the application's callbacs"""
        raise NotImplementedError()

    def get_optimizer(self):
        """Returns the optimizer for this application

        Applications can override this, if needed.
        """
        learning_rate = 1.0e-5  # Doesn't matter for this benchmark
        optimizer = keras.optimizers.Adam(learning_rate)
        if self.with_horovod:
            import horovod.tensorflow as hvd
            import horovod.keras.callbacks as hvd_callbacks
            optimizer = hvd.DistributedOptimizer(optimizer, backward_passes_per_step=1,
                    average_aggregated_gradients=True)
        return optimizer

    def get_dataset(self,num_batches):
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
        
        num_samples = int(num_batches * self.args.batch_size)
        output_signature = (tf.TensorSpec(shape=self.input_shape, dtype=tf.float32), tf.TensorSpec(shape=self.target_shape, dtype=tf.float32))
        dataset = tf.data.Dataset.from_generator(get_generator(self.input_shape, self.target_shape, num_samples), output_signature=output_signature)

        # drop_remainder needed for IPU:
        dataset = dataset.batch(self.args.batch_size, drop_remainder=True)
        return dataset

class AP1(Application):
        
    def __init__(
        self,
        args,
        num_processes,
        with_horovod
    ):
        super().__init__(args = args,
                        num_processes=num_processes,
                        with_horovod=with_horovod
                        )
        
        self.features = 16
        self.levels = 3
        self.pool_size = 2
        self.conv_size = 3
        self.upsampling_type = 'conv_transpose'
        self.num_predictors = 17
        self.patch_size = self.args.patch_size
        self.num_outputs = 3
        
        
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
    
    @property
    def batch_bytes(self):
        return np.product(self.input_shape)
    
    def get_loss_function(self):
        return losses.quantile_score

    def get_callbacks(self,callbacks):
        if self.with_horovod:
            import horovod.tensorflow as hvd
            import horovod.keras.callbacks as hvd_callbacks
            callbacks += [hvd_callbacks.BroadcastGlobalVariablesCallback(0)]
            callbacks += [hvd_callbacks.MetricAverageCallback()]             
        return callbacks
    
    
class AP3(Application):
    """
    TODO
    """
    def __init__(
        self,
        args,
        num_processes,
        with_horovod
    ):
        super().__init__(args = args,
        num_processes=num_processes,
        with_horovod=with_horovod)


    def get_dataset(self,num_batches):
        num_samples=int(num_batches * self.args.batch_size)

        # Your dictionary of arrays
        preds = {'sca_inputs': tf.random.uniform(shape=(num_samples,17),dtype=tf.float32),
                 'col_inputs': tf.random.uniform(shape=(num_samples,137,27),dtype=tf.float32),
                 'hl_inputs': tf.random.uniform(shape=(num_samples,138,2),dtype=tf.float32),
                 'inter_inputs': tf.random.uniform(shape=(num_samples,136,1),dtype=tf.float32) ,
                 'pressure_hl': tf.random.uniform(shape=(num_samples,138,1),dtype=tf.float32),
                   }


        targets = {'sw':tf.random.uniform(shape=(num_samples,138,2),dtype=tf.float32),
                  'hr_sw':tf.random.uniform(shape=(num_samples,137,1),dtype=tf.float32),
                  }

        # Convert dictionary to TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((preds,targets))

        # Batch the dataset
        dataset = dataset.batch(self.args.batch_size,drop_remainder=True)

        # Optionally, you can shuffle the dataset
        dataset = dataset.shuffle(buffer_size=num_samples)
        return dataset
    
    @property
    def batch_bytes(self):
        return np.sum([np.product(val.shape.as_list()[1:]) for val in self.input_shape.values()])


    @property
    def target_shape(self):
        return {'sw': tf.TensorSpec(shape=(None, 138, 2), dtype=tf.float32, name=None),
              'hr_sw': tf.TensorSpec(shape=(None, 137, 1), dtype=tf.float32, name=None)}


    @property
    def input_shape(self):
        return {'sca_inputs': tf.TensorSpec(shape=(None, 17), dtype=tf.float32, name=None),
         'col_inputs':  tf.TensorSpec(shape=(None, 137, 27), dtype=tf.float32, name=None),
         'hl_inputs':  tf.TensorSpec(shape=(None, 138, 2), dtype=tf.float32, name=None),
         'inter_inputs':  tf.TensorSpec(shape=(None, 136, 1), dtype=tf.float32, name=None),
         'pressure_hl':  tf.TensorSpec(shape=(None, 138, 1), dtype=tf.float32, name=None)}


    def get_model(self):
        dl_test=False
        nneur=64
        lstm=True

        return models.build_rnn(
                self.input_shape,
                self.target_shape,
                nneur=nneur,
                dl_test=dl_test,
                lstm=lstm
            )

    def get_loss_function(self):

        loss_weights = {"hr_sw": 10 ** (-1), "sw": 1}
        #loss = {"hr_sw": "mae", "sw": top_scaledflux_mae}
        loss = {"hr_sw": "mae", "sw": "mae"}
        return loss, loss_weights

    def get_optimizer(self):
        lr = 0.5 * 10 ** (-3)
        if self.with_horovod:
            import horovod.tensorflow as hvd
            import horovod.keras.callbacks as hvd_callbacks
            if hvd.size() == 4: #! TODO CHECK IF THIS PROBLEMATIC
                lr = lr / 2

        true_lr = lr * self.args.batch_size / 256 * self.num_processes 
        opt = keras.optimizers.Adam(true_lr)
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
            import horovod.tensorflow as hvd
            import horovod.keras.callbacks as hvd_callbacks
            callbacks.append(hvd_callbacks.BroadcastGlobalVariablesCallback(0))
            callbacks.append(hvd_callbacks.MetricAverageCallback())

        return callbacks


class AP5(Application):
    """
    Application 5: WGAN for statistical downscaling of ERA5 data to COSMO-REA6
    """
    def __init__(self, args, num_processes, with_horovod):
        """
        :param patch_size: size of the input patch
        :param num_predictors: number of predictors
        :param ntargets: number of 'dynamic' targets (excl. z)
        :param hparams: hyperparameters
        """
        super().__init__(args = args,
                        num_processes=num_processes,
                        with_horovod=with_horovod
                        )
        
        self.patch_size = self.args.patch_size_rect
        self.num_predictors = 15
        # hyperparameters for WGAN according to tuning results
        self.hparams = {"batch_size": None, "generator": {"l_avgpool": False, "activation": "swish", "z_branch": True}, 
                        "critic": {"activation": "swish"}}
        self.ntargets_dyn = 1
        self.ntargets = self.ntargets_dyn +1 if self.hparams["generator"].get("z_branch", False) else self.ntargets_dyn    # +1 for z-output (cf. Sha et al. 2020)
        
        self.batch_size = args.batch_size


    def get_model(self):
        """ 
        Returns a WGAN model for AP5.
        """
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

    @property
    def batch_bytes(self):
        return np.prod(self.input_shape)
    
    def get_optimizer(self, with_horovod=False):
        """
        Returns the optimizers in a dictionary for the WGAN of AP5.
        """
        learning_rate = 1.0e-5  # Doesn't matter for this benchmark
        optimizers = {}
        #opt = keras.optimizers.legacy.Adam             # so far, better performance for H100-experiments
        opt = keras.optimizers.Adam                     # default optimizer
        optimizers["c_optimizer"] = opt(learning_rate)
        optimizers["g_optimizer"] = opt(learning_rate)
        if with_horovod:
            import horovod.tensorflow as hvd
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
    
    def get_dataset(self, num_batches):
        """ Creates a tf dataset with specified sizes
        Args:
            num_batches (int): Number of batches in the dataset

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
        batch_size_eff = self.batch_size * self.hparams.get("d_steps", 5)

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
    
    def get_callbacks(self,callbacks):
        if self.with_horovod:
            import horovod.tensorflow as hvd
            import horovod.keras.callbacks as hvd_callbacks
            callbacks += [hvd_callbacks.BroadcastGlobalVariablesCallback(0)]
            callbacks += [hvd_callbacks.MetricAverageCallback()]             
        return callbacks
