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