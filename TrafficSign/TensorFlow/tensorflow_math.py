import tensorflow as tf

x = tf.constant(10)
y = tf.constant(2)
z = tf.subtract(tf.divide(x, y), tf.cast(1, tf.float64))

with tf.Session() as sess:
    output = sess.run(z)
    print(output)
