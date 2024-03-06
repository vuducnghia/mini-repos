import numpy as np
import tensorflow as tf

# a = np.array([[[2, 5]]], dtype=float)
#
# x = tf.keras.layers.Dense(units=12)(a)

a = np.array([[[1, 2], [0, 4]]])

c = tf.math.equal(a, 0)
# b = tf.reduce_sum(a, axis=1)
e = tf.logical_not(c)
e = tf.cast(e, dtype=a.dtype)