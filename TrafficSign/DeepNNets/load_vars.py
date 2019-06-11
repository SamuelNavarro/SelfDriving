import tensorflow as tf

tf.reset_default_graph()

weights = tf.Variable( tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, './model.ckpt')

    print('Weight')
    print(sess.run(weights))
    print('Biases')
    print(sess.run(bias))
