import tensorflow as tf
import numpy as np


def regression_loss(y_true, y_pred):
    return 0.5 * tf.math.square(tf.norm(y_true-y_pred, 2))

