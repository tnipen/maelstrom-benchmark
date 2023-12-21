from tensorflow import keras
import tensorflow as tf
from tensorflow import nn
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    ZeroPadding1D,
)
from tensorflow.keras.layers import (
    Multiply,
    Dense,
    GRU,
    LSTM,
)
from tensorflow.keras.models import Model

"""
ML models AP1
""" 

class Unet(keras.Model):
    def __init__(
        self,
        input_shape,
        num_outputs,
        features=16,
        layers=6,
        pool_size=2,
        conv_size=1,
        upsampling_type="upsampling",
        separable=False,
        with_leadtime=False,
        batch_normalization=True,
        downsampling_type="max",
        activation="relu",
        skipcon=True,
        feature_ratio=2,
        bn_momentum=0.99,
        padding="same",
        leadtime_index=None,
        bias_indices=None,

    ):
        """U-net

        Args:
            features (int): Number of features in the first layer
            layers (int): Depth of the U-net
            pool_size (int): Pooling ratio (> 0)
            upsampling_type (str): One of "upsampling" or "conv_transpose"
            conv_size (int): Convolution size (> 0)
            with_leadtime (bool): Should the last layer be leadtime dependent?
        """
        if upsampling_type not in ["upsampling", "conv_transpose", "upsampling_nearest"]:
            raise ValueError(f"Unknown upsampling type {upsampling_type}")

        self._features = features
        self._layers = layers
        self._pool_size = pool_size
        self._conv_size = conv_size
        self._with_leadtime = with_leadtime
        self._upsampling_type = upsampling_type
        self._separable = separable
        self._batch_normalization = batch_normalization
        self._downsampling_type = downsampling_type
        self._activation = activation
        self._skipcon = skipcon
        self._feature_ratio = feature_ratio
        self._bn_momentum = bn_momentum
        self._padding = padding
        self._leadtime_index = leadtime_index
        self._bias_indices = bias_indices
        self._num_outputs = num_outputs

        if downsampling_type not in ["max", "mean"]:
            raise ValuerError(f"Unknown downsampling type {downsampling_type}")

        # Build the model
        inputs = keras.layers.Input(input_shape)
        outputs = self.get_outputs(inputs)

        super().__init__(inputs, outputs)

    def get_outputs(self, inputs):
        outputs = inputs
        layers = list()

        features = self._features
        padding = self._padding

        if self._separable:
            Conv = maelstrom.layers.DepthwiseConv2D
            Conv = maelstrom.layers.SeparableConv2D
            pool_size = [1, self._pool_size, self._pool_size]
            conv_size = [self._conv_size, self._conv_size]
            up_pool_size = [self._pool_size, self._pool_size]
            up_conv_size = [1, self._conv_size, self._conv_size]
            def Conv(output, features, conv_size, activation_name, batch_normalization):
                for i in range(2):
                    output = maelstrom.layers.SeparableConv3D(features, conv_size, padding=padding)(output)
                    if batch_normalization:
                        output = keras.layers.BatchNormalization(momentum=self._bn_momentum, scale=False, center=False)(output)
                    activation_layer = get_activation(activation_name)
                    output = activation_layer(output)
                return output
        else:
            def Conv(output, features, conv_size, activation_name, batch_normalization):
                for i in range(2):
                    output = keras.layers.Conv3D(features, conv_size, padding=padding)(output)
                    if batch_normalization:
                        output = keras.layers.BatchNormalization(momentum=self._bn_momentum, scale=False, center=False)(output)
                        # Activation should be after batch normalization
                    activation_layer = get_activation(activation_name)
                    output = activation_layer(output)
                return output

            pool_size = [1, self._pool_size, self._pool_size]
            conv_size = [1, self._conv_size, self._conv_size]
            up_pool_size = pool_size
            up_conv_size = conv_size

        # Downsampling
        # conv -> conv -> max_pool
        for i in range(self._layers - 1):
            outputs = Conv(outputs, features, conv_size, self._activation, self._batch_normalization)
            layers += [outputs]
            # print(i, outputs.shape)

            name = f"L{i + 1}_pool"
            if self._downsampling_type == "max":
                outputs = keras.layers.MaxPooling3D(pool_size=pool_size, name=name)(outputs)
            elif self._downsampling_type == "min":
                outputs = keras.layers.MinPooling3D(pool_size=pool_size, name=name)(outputs)
            elif self._downsampling_type == "mean":
                outputs = keras.layers.AveragePooling3D(pool_size=pool_size, name=name)(outputs)
            features *= self._feature_ratio

        # conv -> conv
        outputs = Conv(outputs, features, conv_size, self._activation, self._batch_normalization)

        # upconv -> concat -> conv -> conv
        for i in range(self._layers - 2, -1, -1):
            features /= self._feature_ratio
            activation_layer = get_activation(self._activation)
            # Upsampling
            if self._upsampling_type == "upsampling":
                # The original paper used this kind of upsampling
                outputs = keras.layers.Conv3D(features, conv_size,
                        activation=activation_layer, padding=padding)(
                    outputs
                )
                UpConv = keras.layers.UpSampling3D
                outputs = UpConv(pool_size, name=f"L{i + 2}_up")(outputs)
                activation_layer = get_activation(self._activation)
                # Do a 2x2 convolution to simulate "learnable" bilinear interpolation
                outputs = keras.layers.Conv3D(features, [1, 2, 2], activation=activation_layer,
                        padding=padding)(outputs)
            elif self._upsampling_type == "upsampling_nearest":
                outputs = keras.layers.Conv3D(features, conv_size,
                        activation=activation_layer, padding="same")(
                    outputs
                )
                UpConv = keras.layers.UpSampling3D
                outputs = UpConv(pool_size)(outputs)
                # Don't do a 2x2 convolution
            elif self._upsampling_type == "conv_transpose":
                # Some use this kind of upsampling. This seems to create a checkered pattern in the
                # output, at least for me.
                UpConv = keras.layers.Conv3DTranspose
                outputs = UpConv(features, up_conv_size, strides=pool_size, padding=padding)(outputs)
                outputs = keras.layers.Conv3D(features, [1, 2, 2], activation=activation_layer,
                        padding=padding)(outputs)

            # if i == 0 or self._skipcon:
            if self._skipcon:
                # collapse = tf.keras.layers.reshape(layers[i], [outputs)
                # crop = tf.keras.layers.CenterCrop(outputs.shape[2], outputs.shape[3])(layers[i])
                if self._padding == "valid":
                    # Center and crop the skip-connection tensor, since it is larger than the
                    # tensor passed from the lower level
                    d00 = (layers[i].shape[2] - outputs.shape[2])
                    d10 = (layers[i].shape[2] - outputs.shape[2]) - d00
                    d01 = (layers[i].shape[3] - outputs.shape[3])
                    d11 = (layers[i].shape[3] - outputs.shape[3]) - d01
                    # print(d00, d10, d01, d11)
                    # Would be nice to use tf.keras.layers.CenterCrop, but this doesn't work for 5D
                    # tensors.
                    crop = tf.keras.layers.Cropping3D(((0, 0), (d00, d10), (d01, d11)))(layers[i])
                    outputs = keras.layers.concatenate((crop, outputs), axis=-1)
                elif self._padding == "reflect":
                    d00 = (layers[i].shape[2] - outputs.shape[2]) // 2
                    d10 = (layers[i].shape[2] - outputs.shape[2]) - d00
                    d01 = (layers[i].shape[3] - outputs.shape[3]) // 2
                    d11 = (layers[i].shape[3] - outputs.shape[3]) - d01
                    paddings = tf.constant([[0, 0], [0, 0], [d00, d10], [d01, d11], [0, 0]])
                    expanded = tf.pad(outputs, paddings, "REFLECT")
                    # print(paddings, layers[i].shape, outputs.shape, expanded.shape)
                    outputs = keras.layers.concatenate((layers[i], expanded), axis=-1)
                else:
                    outputs = keras.layers.concatenate((layers[i], outputs), axis=-1, name=f"L{i + 1}_concat")
            outputs = Conv(outputs, features, conv_size, self._activation, self._batch_normalization)

        # Create a separate branch with f(leadtime) multiplied by each bias field
        if self._leadtime_index is not None and len(self._bias_indices) > 0:
            leadtime_input = inputs[..., self._leadtime_index]
            leadtime_input = tf.expand_dims(leadtime_input, -1)
            bias_inputs = [tf.expand_dims(inputs[..., i], -1) for i in self._bias_indices]

            leadtime_mult = list()
            for i in range(len(bias_inputs)):
                curr_leadtime_input = leadtime_input
                # Create a flexible function for leadtime
                for j in range(4):
                    activation_layer = "tanh"
                    curr_leadtime_input = keras.layers.Dense(5, activation=activation_layer)(curr_leadtime_input)
                curr_leadtime_input = keras.layers.Dense(1, name=f"leadtime_bias_{i}")(curr_leadtime_input)

                # Multiply the leadtime function by the bias
                curr = tf.multiply(curr_leadtime_input, bias_inputs[i])
                leadtime_mult += [curr]
            outputs = keras.layers.concatenate(leadtime_mult + [outputs], axis=-1)

        # Dense layer at the end
        if self._with_leadtime:
            layer = keras.layers.Dense(self._num_outputs, activation="linear")
            outputs = maelstrom.layers.LeadtimeLayer(layer, "dependent")(outputs)
        else:
            outputs = keras.layers.Dense(self._num_outputs, activation="linear")(
                outputs
            )

        return outputs

def get_activation(name, *args, **kwargs):
    """Get an activation layer corresponding to the name

    Args:
        name (str): Name of layer
        args (list): List of arguments to layer
        kwargs (dict): Named arguments to layer

    Returns:
        keras.layer.Layer: An initialized layer
    """

    if name.lower() == "leakyrelu":
        return keras.layers.LeakyReLU(*args, **kwargs)
        # return keras.layers.LeakyReLU(alpha=0.05)
    else:
        return keras.layers.Activation(name)

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
ML models AP3
""" 

@tf.keras.utils.register_keras_serializable()
class TopFlux(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(TopFlux, self).__init__(name=name, **kwargs)
        self.g_cp = tf.constant(9.80665 / 1004 * 24 * 3600)

    def build(self, input_shape):
        pass

    def call(self, inputs):
        fluxes = inputs[0]
        hr = tf.squeeze(inputs[1])
        hlpress = inputs[2]
        # Net surface flux = down - up
        netflux = fluxes[..., 0] - fluxes[..., 1]
        # Pressure difference between the half-levels
        net_press = hlpress[..., 1:, 0] - hlpress[..., :-1, 0]
        # Integrate the heating rate through the atmosphere
        hr_sum = tf.math.reduce_sum(tf.math.multiply(hr, net_press), axis=-1)
        # Stack the outputs
        # TOA net flux, Surface down, #Surface up
        # upwards TOA flux can be deduced as down flux is prescribed
        # either by solar radiation for SW (known) or 0 for LW.
        return tf.stack(
            [netflux + hr_sum / self.g_cp, fluxes[..., 0], fluxes[..., 1]], axis=-1
        )


class rnncolumns(tf.keras.layers.Layer):
    def __init__(self, include_constants=False, name=None, **kwargs):
        super(rnncolumns, self).__init__(name=name, **kwargs)
        colnorms = tf.constant(
            [
                4.61617715e01,
                5.98355832e04,
                2.36960248e03,
                3.01348603e06,
                4.92351671e05,
                4.77463763e00,
                1.16648264e09,
                2.01012275e09,
                1.0,
                1.0,  # <-- Two zero inputs
                1.00000000e00,
                4.12109712e08,
                4.82166968e06,
                3.96867640e06,
                1.97749625e07,
                7.20587302e06,
                7.82937119e06,
                1.66701023e07,
                2.03854471e07,
                2.43620336e08,
                1.37198036e08,
                4.13003711e07,
                2.10871729e09,
                6.47918275e02,
                1.10262260e02,
                3.33333342e04,
                9.93289347e03,
            ],
            dtype=tf.float32,
        )
        if not include_constants:  # remove indices 5 (O2), 8 (hcfc22), 9 (ccl4_vmr)
            colnorms = tf.concat([colnorms[0:5], colnorms[6:8], colnorms[10:]], axis=0)

        self.colnorms = tf.expand_dims(tf.expand_dims(colnorms, axis=0), axis=0)
        self.nlay = tf.constant(137)

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def build(self, input_shape):
        pass

    def call(self, inputs):

        fl_inputs, hl_inputs, cos_sza = inputs 
        cos_sza_lay = tf.repeat(
            tf.expand_dims(cos_sza, axis=1), repeats=self.nlay, axis=1
        )
        cos_sza_lay = tf.expand_dims(cos_sza_lay, axis=2)

        fl_inputs2 = tf.concat(
            [fl_inputs[:, :, 0:5], fl_inputs[:, :, 6:8], fl_inputs[:, :, 10:]], axis=2
        )
        fl_inputs2 = tf.math.multiply(fl_inputs2, self.colnorms)

        hl_p = hl_inputs[..., :1]
        temp_hl = hl_inputs[..., 1:]

        # Add pressure to layer inputs
        # Pressure at layers / full levels (137)
        pres_fl = tf.math.multiply(
            tf.constant(0.5), tf.add(hl_p[:, :-1, :], hl_p[:, 1:, :])
        )
        # First normalize
        pres_fl_norm = tf.math.log(pres_fl)
        pres_fl_norm = tf.math.multiply(
            pres_fl_norm, tf.constant(0.086698161)
        )  # scale roughly to 0-1

        temp_fl_norm = tf.multiply(
            tf.constant(0.0031765 * 0.5), tf.add(temp_hl[:, :-1, :], temp_hl[:, 1:, :])
        )

        deltap = tf.math.multiply(
            tf.math.subtract(hl_p[:, 1:, :], hl_p[:, :-1, :]), tf.constant(0.0004561)
        )

        return tf.concat(
            [fl_inputs2, cos_sza_lay, pres_fl_norm, temp_fl_norm, deltap], axis=-1
        )

@tf.keras.utils.register_keras_serializable()
class HRLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        name=None,
        hr_units="K d-1",
        **kwargs,
    ):
        super(HRLayer, self).__init__(name=name, **kwargs)
        time_scale = {"K s-1": 1, "K d-1": 3600 * 24}[hr_units]
        self.g_cp = tf.constant(9.80665 / 1004 * time_scale, dtype=tf.float32)

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def build(self, input_shape):
        pass

    def call(self, inputs):
        fluxes = inputs[0]
        hlpress = inputs[1]
        netflux = fluxes[..., 0] - fluxes[..., 1]
        flux_diff = netflux[..., 1:] - netflux[..., :-1]
        net_press = hlpress[..., 1:, 0] - hlpress[..., :-1, 0]
        return -self.g_cp * tf.math.divide(flux_diff, net_press)
    
    
    
def build_rnn(
    inp_spec,
    outp_spec,
    nneur=64,
    hr_loss=True,
    activ_last="sigmoid",
    activ0="linear",
    dl_test=False,
    lstm=True,
):
    #Assume inputs have the order
    #scalar, column, hl, inter, pressure_hl
    kw = 5
    all_inp = []
    for k in inp_spec.keys():
        all_inp.append(Input(inp_spec[k].shape[1:],name=k))

    scalar_inp = all_inp[0]
    lay_inp = all_inp[1]
    hl_inp = all_inp[2]
    inter_inp = all_inp[3]
    hl_p = all_inp[-1]

    # inputs we need:
    #  - layer inputs ("lay_inp"), which are the main RNN sequential input
    #     -- includes repeated mu0, t_lay, and log(p_lay)
    #  - albedos, fed to a dense layer whose output is concatenated with the initial
    #             RNN output sequence (137) to get to half-level outputs (138)

    # extract scalar variables we need
    cos_sza = scalar_inp[:,1]
    albedos = scalar_inp[:,2:14]
    solar_irrad = scalar_inp[:,-1]   # not needed as input when predicting scaled flux

    overlap_param = ZeroPadding1D(padding=(1,1))(inter_inp)

    lay_inp = rnncolumns(name='procCol')([lay_inp,hl_inp,cos_sza])

    # 2. OUTPUTS
    # Outputs are the raw fluxes scaled by incoming flux
    ny = 2
    # incoming flux from inputs
    incflux = Multiply()([cos_sza, solar_irrad])
    incflux = tf.expand_dims(tf.expand_dims(incflux,axis=1),axis=2)

    # hidden0,last_state = layers.SimpleRNN(nneur,return_sequences=True,return_state=True)(inputs)
    if lstm:
        rnnlayer = LSTM
        hidden0,last_state,last_memory = rnnlayer(nneur,return_sequences=True,
                                      return_state=True,name="RNN1")(lay_inp)
        last_state = tf.concat([last_state,last_memory],axis=-1)
    else:
        rnnlayer = GRU
        hidden0,last_state = rnnlayer(nneur,return_sequences=True,
                                      return_state=True,name="RNN1")(lay_inp)

    last_state_plus_albedo =  tf.concat([last_state,albedos],axis=1)

    mlp_surface_outp = Dense(nneur, activation=activ0,name='dense_surface')(last_state_plus_albedo)

    hidden0_lev = tf.concat([hidden0,tf.reshape(mlp_surface_outp,[-1,1,nneur])],axis=1)

    # !! OVERLAP PARAMETER !! added here as an additional feature to the whole sequence
    hidden0_lev = tf.concat([hidden0_lev, overlap_param],axis=2)

    if lstm:
        mlp_surface_outp2 = Dense(nneur, activation=activ0,name='dense_surface2')(last_state_plus_albedo)
        init_state = [mlp_surface_outp,mlp_surface_outp2]
    else:
        init_state = mlp_surface_outp
        
    hidden1 = rnnlayer(nneur,return_sequences=True,
                       go_backwards=True)(hidden0_lev, initial_state=init_state)

    hidden1 = tf.reverse(hidden1,axis=[1])

    # at least before I had better results when concatinating hidden0 and hidden1
    hidden_concat  = tf.concat([hidden0_lev,hidden1],axis=2)

    # Third and final RNN layer
    hidden2 = rnnlayer(nneur,return_sequences=True)(hidden_concat)#,#
    # flux_sw = TimeDistributed(Dense(ny, activation=activ_last))(hidden2)
    flux_sw = Conv1D(ny, kernel_size = 1, activation=activ_last,
                     name='sw_denorm'
    )(hidden2)

    flux_sw = Multiply(name='sw')([flux_sw, incflux])
    hr_sw = HRLayer(name='hr_sw')([flux_sw, hl_p])

    outputs = {'sw':flux_sw, 'hr_sw':hr_sw}

    if dl_test:
        from .dummy_model import DummyModel as Model
    else:
        from tensorflow.keras import Model
    model = Model(all_inp, outputs)
    return model
