from tensorflow import keras
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

