import tensorflow.keras.backend as K
import tensorflow as tf


def quantile_score(y_true, y_pred):
    quantiles = [0.1, 0.5, 0.9]
    qtloss = 0
    for i, quantile in enumerate(quantiles):
        err = y_true[..., 0] - y_pred[..., i]
        qtloss += (quantile - tf.cast((err < 0), tf.float32)) * err
    return K.mean(qtloss) / len(quantiles)
