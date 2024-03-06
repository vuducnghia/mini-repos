import tensorflow as tf

x = tf.Variable(1.0)

with tf.GradientTape() as t:
    with tf.GradientTape() as t2:
        y = x * x * x
    dy_dx = t2.gradient(y, x)
d2y_d2x = t.gradient(dy_dx, x)

print(dy_dx.numpy())
print(d2y_d2x.numpy())
