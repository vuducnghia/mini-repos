# https://medium.com/analytics-vidhya/tf-gradienttape-explained-for-keras-users-cc3f06276f22
import tensorflow as tf

# x = tf.Variable(3.0)
#
# with tf.GradientTape() as tape:
#     # tape.watch(x)
#     y = x**3
# print(tape.gradient(y, x))

a = tf.Variable(6.0, trainable=True)
b = tf.Variable(2.0, trainable=True)
with tf.GradientTape(persistent=True) as tape:
    y1 = a ** 2
    y2 = b ** 3
    #  in long functions, it is more readable to use stop_recording blocks multiple times to calculate gradients in the middle of a function,
    #  than to calculate all the gradients at the end of a function.
    with tape.stop_recording():
        print(tape.gradient(y1, a).numpy())

    with tape.stop_recording():
        print(tape.gradient(y2, b).numpy())
