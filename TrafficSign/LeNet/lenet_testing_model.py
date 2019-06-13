import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from lenet import evaluate

saver = tf.train.Saver()


mnist = input_data.read_data_sets("/home/samuel/Data/MNIST_data", reshape=False)
X_test, y_test = mnist.test.images, mnist.test.labels

X_test = np.pad(X_test, ((0,0), (2,2), (2,2), (0,0)), 'constant')

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
