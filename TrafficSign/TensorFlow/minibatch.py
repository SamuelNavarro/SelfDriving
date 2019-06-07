from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

lr = 0.001
n_input = 784
n_classes = 10

mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)

train_features = mnist.train.images
test_features = mnist.test.images

train_labels = mnist.train.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)

features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

weights = tf.Variable(tf.random_normal([n_input, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))

logits = tf.add(tf.matmul(features, weights), biases)


# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def get_batches(batch_size, features, labels):
    batches = []

    all_data = len(features)
    for i in range(0, all_data, batch_size):
        end_i = i + batch_size
        batch = [features[i:end_i], labels[i:end_i]]
        batches.append(batch)

    return batches


batch_size = 128
assert batch_size is not None, "You must set the batch size"


init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    for batch_features, batch_labels in get_batches(batch_size=batch_size,
                                                    train_features,
                                                    train_labels):
        sess.run(optimizer, feed_dict={features:batch_features, labels:batch_labels})

    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: test_features, labels: test_labels})


print('Test Accuracy: {}'.format(test_accuracy))
