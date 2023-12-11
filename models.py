# import modules
from collections import OrderedDict
import numpy as np
from typing import List
from tensorflow import keras
import tensorflow as tf

"""
ML models
"""
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
        """U-net in Application 1

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


"""
ML models AP5
""" 

def advanced_activation(activation_name, *args, **kwargs):
    """
    Get layer to enable one of Keras' advanced activation, see https://keras.io/api/layers/activation_layers/
    :param activation_name: name of the activation function to apply
    :return: the respective layer to deploy desired activation
    """
    known_activations = ["LeakyReLU", "Softmax", "PReLU", "ELU", "ThresholdedReLU"]

    activation_name = activation_name.lower()

    if activation_name == "leakyrelu":
        layer = keras.layers.LeakyReLU(*args, **kwargs)
    elif activation_name == "softmax":
        layer = keras.layers.Softmax(*args, **kwargs)
    elif activation_name == "elu":
        layer = keras.layers.ELU(*args, **kwargs)
    elif activation_name == "prelu":
        layer = keras.layers.PReLU(*args, **kwargs)
    elif activation_name == "thresholdedrelu":
        layer = keras.layers.ThresholdedReLU(*args, **kwargs)
    else:
        raise ValueError("{0} is not a valid advanced activation function. Choose one of the following: {1}"
                         .format(activation_name, ", ".join(known_activations)))

    return layer


# building blocks for U-net

def conv_block(inputs, num_filters: int, kernel: tuple = (3, 3), strides: tuple = (1, 1), padding: str = "same",
               activation: str = "swish", activation_args={}, kernel_init: str = "he_normal",
               l_batch_normalization: bool = True):
    """
    A convolutional layer with optional batch normalization
    :param inputs: the input data with dimensions nx, ny and nc
    :param num_filters: number of filters (output channel dimension)
    :param kernel: tuple for convolution kernel size
    :param strides: tuple for stride of convolution
    :param padding: technique for padding (e.g. "same" or "valid")
    :param activation: activation fuction for neurons (e.g. "relu")
    :param activation_args: arguments for activation function given that advanced layers are applied
    :param kernel_init: initialization technique (e.g. "he_normal" or "glorot_uniform")
    :param l_batch_normalization: flag if batch normalization should be applied
    """
    x = keras.layers.Conv2D(num_filters, kernel, strides=strides, padding=padding, kernel_initializer=kernel_init)(inputs)
    if l_batch_normalization:
        x = keras.layers.BatchNormalization()(x)

    try:
        x = keras.layers.Activation(activation)(x)
    except ValueError:
        ac_layer = keras.layers.advanced_activation(activation, *activation_args)
        x = ac_layer(x)

    return x


def conv_block_n(inputs, num_filters: int, n: int = 2, **kwargs):
    """
    Sequential application of two convolutional layers (using conv_block).
    :param inputs: the input data with dimensions nx, ny and nc
    :param num_filters: number of filters (output channel dimension)
    :param n: number of convolutional blocks
    :param kwargs: keyword arguments for conv_block
    """
    x = conv_block(inputs, num_filters, **kwargs)
    for _ in np.arange(n - 1):
        x = conv_block(x, num_filters, **kwargs)

    return x


def encoder_block(inputs, num_filters, l_large: bool = True, kernel_pool: tuple = (2, 2), l_avgpool: bool = False, **kwargs):
    """
    One complete encoder-block used in U-net.
    :param inputs: input to encoder block
    :param num_filters: number of filters/channel to be used in convolutional blocks
    :param l_large: flag for large encoder block (two consecutive convolutional blocks)
    :param kernel_maxpool: kernel used in max-pooling
    :param l_avgpool: flag if average pooling is used instead of max pooling
    :param kwargs: keyword arguments for conv_block
    """
    if l_large:
        x = conv_block_n(inputs, num_filters, n=2, **kwargs)
    else:
        x = conv_block(inputs, num_filters, **kwargs)

    if l_avgpool:
        p = keras.layers.AveragePooling2D(kernel_pool)(x)
    else:
        p = keras.layers.MaxPool2D(kernel_pool)(x)

    return x, p

def subpixel_block(inputs, num_filters, kernel: tuple = (3,3), upscale_fac: int = 2,
                   padding: str = "same", activation: str = "swish", activation_args: dict = {},
                   kernel_init: str = "he_normal"):


    x = keras.layers.Conv2D(num_filters * (upscale_fac ** 2), kernel, padding=padding, kernel_initializer=kernel_init,
               activation=None)(inputs)

    try:
        x = keras.layers.Activation(activation)(x)
    except ValueError:
        ac_layer = advanced_activation(activation, *activation_args)
        x = ac_layer(x)

    
    x = tf.nn.depth_to_space(x, upscale_fac)

    return x



def decoder_block(inputs, skip_features, num_filters, strides_up: int = 2, l_subpixel: bool = False, **kwargs_conv_block):
    """
    One complete decoder block used in U-net (reverting the encoder)
    """
    if l_subpixel:
        kwargs_subpixel = kwargs_conv_block.copy()
        for ex_key in ["strides", "l_batch_normalization"]:
            kwargs_subpixel.pop(ex_key, None)
        x = subpixel_block(inputs, num_filters, upscale_fac=strides_up, **kwargs_subpixel)
    else:
        x = keras.layers.Conv2DTranspose(num_filters, (strides_up, strides_up), strides=strides_up, padding="same")(inputs)
        
        activation = kwargs_conv_block.get("activation", "relu")
        activation_args = kwargs_conv_block.get("activation_args", {})

        try:
            x = keras.layers.Activation(activation)(x)
        except ValueError:
            ac_layer = advanced_activation(activation, *activation_args)
            x = ac_layer(x)

    x = keras.layers.Concatenate()([x, skip_features])
    x = conv_block_n(x, num_filters, 2, **kwargs_conv_block)

    return x


# The U-net model architecture
def sha_unet(input_shape: tuple, hparams_unet: dict, ntargets: int, concat_out: bool = False) -> keras.Model:
    """
    Builds up U-net model architecture adapted from Sha et al., 2020 (see https://doi.org/10.1175/JAMC-D-20-0057.1).
    :param input_shape: shape of input-data
    :param channels_start: number of channels to use as start in encoder blocks
    :param ntargets: number of target variables (dynamic output variables)
    :param z_branch: flag if z-branch is used.
    :param advanced_unet: flag if advanced U-net is used (LeakyReLU instead of ReLU, average pooling instead of max pooling and subpixel-layer)
    :param concat_out: boolean if output layers will be concatenated (disables named target channels!)
    :param tar_channels: name of output/target channels (needed for associating losses during compilation)
    :return:
    """
    # basic configuration of U-Net 
    ntargets_dyn = ntargets - 1 if hparams_unet["z_branch"] else ntargets

    channels_start = hparams_unet.get("ngf", 56)
    z_branch = hparams_unet.get("z_branch", False)
    kernel_pool = hparams_unet.get("kernel_pool", (3, 3))
    l_avgpool = hparams_unet.get("l_avgpool", True)
    l_subpixel = hparams_unet.get("l_subpixel", True)

    config_conv = {"kernel": hparams_unet["kernel"], "strides": hparams_unet["strides"], "padding": hparams_unet["padding"], 
                   "activation": hparams_unet["activation"], "activation_args": hparams_unet["activation_args"], 
                   "kernel_init": hparams_unet["kernel_init"], "l_batch_normalization": hparams_unet["l_batch_normalization"]}

    # build U-Net
    inputs = keras.layers.Input(input_shape)

    """ encoder """
    s1, e1 = encoder_block(inputs, channels_start, l_large=True, kernel_pool=kernel_pool, l_avgpool=l_avgpool,**config_conv)
    s2, e2 = encoder_block(e1, channels_start * 2, l_large=False, kernel_pool=kernel_pool, l_avgpool=l_avgpool,**config_conv)
    s3, e3 = encoder_block(e2, channels_start * 4, l_large=False, kernel_pool=kernel_pool, l_avgpool=l_avgpool,**config_conv)

    """ bridge encoder <-> decoder """
    b1 = conv_block(e3, channels_start * 8, **config_conv)

    """ decoder """
    d1 = decoder_block(b1, s3, channels_start * 4, l_subpixel=l_subpixel, **config_conv)
    d2 = decoder_block(d1, s2, channels_start * 2, l_subpixel=l_subpixel, **config_conv)
    d3 = decoder_block(d2, s1, channels_start, l_subpixel=l_subpixel, **config_conv)

    output_dyn = keras.layers.Conv2D(ntargets_dyn, (1, 1), kernel_initializer=config_conv["kernel_init"], name=f"dyn_out")(d3)
    if z_branch:
        print("Use z_branch...")
        output_static = keras.layers.Conv2D(1, (1, 1), kernel_initializer=config_conv["kernel_init"], name=f"z_out")(d3)

        if concat_out:
            model = keras.Model(inputs, tf.concat([output_dyn, output_static], axis=-1), name="downscaling_unet_with_z")
        else:
            model = keras.Model(inputs, [output_dyn, output_static], name="downscaling_unet_with_z")
    else:
        model = keras.Model(inputs, output_dyn, name="downscaling_unet")

    return model

# the critic-model architecture
def critic(input_shape: tuple, hparams_critic: dict):
    """
    Set-up convolutional discriminator model that is followed by two dense-layers
    :param input_shape: input shape of data (either real or generated data)
    :param hparams_critic: hyperparameters for critic-model
    :return: critic-model
    """
    critic_in = keras.Input(shape=input_shape)
    x = critic_in
    
    channels_start = hparams_critic.get("channels_start", 56)
    channels = channels_start.copy()
    num_conv = int(hparams_critic.get("num_conv", 4))
    
    assert num_conv > 1, f"Number of convolutional layers is {num_conv:d}, but must be at minimum 2."
    

    for _ in range(num_conv):
        x = conv_block(x, channels, hparams_critic.get("kernel", (3, 3)), hparams_critic.get("stride", (2, 2)),
                       activation=hparams_critic.get("activation", "relu"), l_batch_normalization=hparams_critic.get("lbatch_norm", True))
        channels *= 2
        
    # finally perform global average pooling and finalize by fully connected layers
    x = keras.layers.GlobalAveragePooling2D()(x)
    try:
        x = keras.layers.Dense(channels_start, activation=hparams_critic.get("activation", "relu"))(x)
    except ValueError as _:
        ac = advanced_activation(hparams_critic.get("activation")) 
        x = keras.layers.Dense(channels_start, activation=ac)(x)
    # ... and end with linear output layer
    out = keras.layers.Dense(1, activation="linear")(x)

    return keras.Model(inputs=critic_in, outputs=out)  


"""
Simplified WGAN-implementation, i.e. without learning rate schedule, model checkpointing and other details.
"""
class WGAN(keras.Model):
    """
    WAGN model class
    """
    def __init__(self, generator, critic, hparams_dict: dict):
        """
        :param generator: generator model
        :param critic: critic model
        :param hparams_dict: dictionary with hyperparameters (batch_size must be provided)
        """
        super().__init__()
        self.critic = critic
        self.generator = generator
        self.batch_size = hparams_dict["batch_size"]
        self.d_steps = hparams_dict.get("d_steps", 5)
        self.gp_weight = hparams_dict.get("gp_weight", 10.)
        self.recon_weight = hparams_dict.get("recon_weight", 1000.)
        self.trainable_weights = self.generator.trainable_weights + self.critic.trainable_weights

    def compile(self, optimizer_dict, loss_dict, **kwargs):
        """
        Set the optimizer as well as the adversarial loss functions for the generator and critic.
        Furthermore, the model gets compiled.
        """
        self.c_optimizer = optimizer_dict["c_optimizer"]
        self.g_optimizer = optimizer_dict["g_optimizer"]

        self.critic_loss = loss_dict["critic_loss"]
        self.critic_gen_loss = loss_dict["critic_gen_loss"]
        self.recon_loss = loss_dict["recon_loss"]

        compile_opts = kwargs.copy()
        # drop loss dict and optimizer dict from compile_opts...
        compile_opts.pop("loss_dict", None)
        compile_opts.pop("optimizer_dict", None)
        # ... compile model
        super().compile(**compile_opts) 


    def train_step(self, data_iter: tf.data.Dataset) -> OrderedDict:
        """
        Training step for Wasserstein GAN.
        :param data_iter: Tensorflow Dataset providing training data
        :return: Ordered dictionary with several losses of generator and critic
        """
        predictors, predictands = data_iter

        # train the critic d_steps-times
        for i in range(self.d_steps):
            with tf.GradientTape() as tape_critic:
                ist, ie = i * self.batch_size, (i + 1) * self.batch_size
                # critic only operates on first channel
                predictands_critic = tf.expand_dims(predictands[ist:ie, :, :, 0], axis=-1)
                # generate (downscaled) data
                gen_data = self.generator(predictors[ist:ie, :, :, :], training=True)
                # calculate critics for both, the real and the generated data
                critic_gen = self.critic(gen_data[..., 0], training=True)
                critic_gt = self.critic(predictands_critic, training=True)
                # calculate the loss (incl. gradient penalty)
                c_loss = self.critic_loss(critic_gt, critic_gen)                   
                gp = self.gradient_penalty(predictands_critic, gen_data[..., 0:1])
                d_loss = c_loss + self.gp_weight * gp

            # calculate gradients and update discrimintor
            d_gradient = tape_critic.gradient(d_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(d_gradient, self.critic.trainable_variables))

        # train generator
        with tf.GradientTape() as tape_generator:
            # generate (downscaled) data
            gen_data = self.generator(predictors[-self.batch_size:, :, :, :], training=True)
            # get the critic and calculate corresponding generator losses (critic and reconstruction loss)
            critic_gen = self.critic(gen_data[..., 0], training=True)
            cg_loss = self.critic_gen_loss(critic_gen)
            rloss = self.recon_loss(predictands[-self.batch_size:, :, :, :], gen_data)
            g_loss = cg_loss + self.recon_weight * rloss

        g_gradient = tape_generator.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_variables))

        return OrderedDict(
            [("c_loss", c_loss), ("gp_loss", self.gp_weight * gp), ("d_loss", d_loss), ("cg_loss", cg_loss),
             ("recon_loss", rloss * self.recon_weight), ("g_loss", g_loss)])

    
    def test_step(self, val_iter: tf.data.Dataset) -> OrderedDict:
        """
        Implement step to test trained generator on validation data
        :param val_iter: Tensorflow Dataset with validation data
        :return: dictionary with reconstruction loss on validation data
        """
        predictors, predictands = val_iter

        gen_data = self.generator(predictors, training=False)
        rloss = self.recon_loss(predictands, gen_data)

        return OrderedDict([("recon_loss", rloss)])
    
    def gradient_penalty(self, real_data, gen_data) -> tf.Tensor:
        """
        Calculates gradient penalty based on 'mixture' of generated and ground truth data
        :param real_data: the ground truth data
        :param gen_data: the generated/predicted data
        :return: gradient penalty
        """
        # get mixture of generated and ground truth data
        alpha = tf.random.normal([self.batch_size, 1, 1, 1], 0., 1.)
        mix_data = real_data + alpha * (gen_data - real_data)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(mix_data)
            critic_mix = self.critic(mix_data, training=True)

        # calculate the gradient on the mixture data...
        grads_mix = gp_tape.gradient(critic_mix, [mix_data])[0]
        # ... and norm it
        norm = tf.sqrt(tf.reduce_mean(tf.square(grads_mix), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.) ** 2)
        return gp
    


