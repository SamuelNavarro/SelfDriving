import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math


tf.reset_default_graph()


lr = 0.01
n_input = 784
n_classes = 10

mnist = input_data.read_data_sets('.', one_hot=True)

features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

logits = tf.add(tf.matmul(features, weights), bias)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

optimize = tf.train.GradientDescentOptimizer(lr).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

save_file = './train_model.ckpt'
batch_size = 128
n_epochs = 100

saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epochs):
        total_batch = math.ceil(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_features, batch_labels = mnist.train.next_batch(batch_size)
            sess.run(
                optimize,
                feed_dict={features: batch_features, labels: batch_labels}
            )

        if epoch & 10 == 0:
            valid_accuracy = sess.run(
                accuracy,
                feed_dict = {features: mnist.validation.images,
                             labels: mnist.validation.labels}
            )
            print('Epoch {:<3} - Validation Accuracy: {}'.format(
                epoch,
                valid_accuracy))

    saver.save(sess, save_file)
    print("Trained model saved")
