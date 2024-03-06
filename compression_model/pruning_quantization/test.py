# https://stackoverflow.com/questions/33727935/how-to-use-stop-gradient-in-tensorflow
# https://mlfromscratch.com/tensorflow-2/#/
import tensorflow as tf
import numpy as np
import math
def gelu(x):
    return x*x

def get_gradient(x, activation_function):
    with tf.GradientTape() as gt:
        y = activation_function(x)

    gradient = gt.gradient(y,x).numpy()
    return gradient

x = tf.Variable(0.5)
gradient = get_gradient(x, gelu)

print(gradient)

# import timeit
# conv_layer = tf.keras.layers.Conv2D(100, 3)
# print(conv_layer)
# @tf.function
# def conv_fn(image):
#   return conv_layer(image)
#
# image = tf.zeros([1, 200, 200, 100])
# # print(image.numpy())
# # warm up
# conv_layer(image); conv_fn(image)
#
# no_tf_fn = timeit.timeit(lambda: conv_layer(image), number=10)
# with_tf_fn = timeit.timeit(lambda: conv_fn(image), number=10)
# difference = no_tf_fn - with_tf_fn
#
# print("Without tf.function: ", no_tf_fn)
# print("With tf.function: ", with_tf_fn)
# print("The difference: ", difference)
#
# print("\nJust imagine when we have to do millions/billions of these calculations," \
#       " then the difference will be HUGE!")
# print("Difference times a billion: ", difference*1000000000)
#
# print(tf.newaxis)