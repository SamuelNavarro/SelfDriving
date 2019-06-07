import numpy as np
import tensorflow as tf


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


logits = [3.0, 1.0, 0.2]
# logits = np.array([
    # [1, 2, 3, 6],
    # [2, 4, 5, 6],
    # [3, 8, 7, 6]])


# Tensorflow implementation
def run():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)

    softmax = tf.nn.softmax(logits)

    with tf.Session() as sess:
        output = sess.run(softmax, feed_dict={logits: logit_data})
    return output

print(run())
