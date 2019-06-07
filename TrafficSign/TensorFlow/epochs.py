from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np


def print_epoch_stats(epoch_i, last_features, last_labels):
    current_cost = sess.run(
        cost,
        feed_dict={features: last_features,
                   labels: last_labels}
    )
    valid_accuracy = sess.run(
        accuracy,
        feed_dict={features: valid_features,
                   labels: valid_labels}
    )
    print("Epoch: {:<4} - Cost: {:<8.3} Valid Accuracy: {:<5.3}".format(
        epoch_i,
        current_cost,
        valid_accuracy)
    )


def get_batches(batch_size, features, labels):
    batches = []

    all_data = len(features)
    for i in range(0, all_data, batch_size):
        end_i = i + batch_size
        batch = [features[i:end_i], labels[i:end_i]]
        batches.append(batch)

    return batches


n_input = 784
n_classes = 10
lr = 0.01
batch_size = 128
epochs = 5


mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)

train_features = mnist.train.images
valid_features = mnist.validation.images
test_features = mnist.test.images

train_labels = mnist.train.labels.astype(np.float32)
valid_labels = mnist.validation.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)


features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

weights = tf.Variable(tf.truncated_normal([n_input, n_classes]))
biases = tf.Variable(tf.zeros(n_classes,))

logits = tf.matmul(tf.add(features, weights), biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))


optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_batches = get_batches(batch_size, features, labels)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        for batch_features, batch_labels in train_batches:
            sess.run(optimizer, feed_dict={
                features: batch_features,
                labels: batch_labels,
                learning_rate: lr
            })

        print_epoch_stats(epoch, sess, batch_features, batch_labels)

    test_accuracy = sess.run(
        accuracy,
        feed_dict={
            features: test_features,
            labels: test_labels
        })

print("Test Accuracy: {}".format(test_accuracy))
