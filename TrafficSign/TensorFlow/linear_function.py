import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def get_weights(n_features, n_labels):
    """
    Return Tensorflow weights
    :param n_features: Number of features
    :param n_labels: Number of labels
    :return: Tensorflow weights
    """
    weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
    return weights


def get_biases(n_labels):
    biases = tf.Variable(tf.zeros(n_labels))
    return biases


def linear(input, w, b):
    """
    Return linear funciton in Tensorflow
    :param input: Tensorflow input
    :param w: Tensorflow weights
    :param b: Tensorflow biases
    :return: Tensorflow linear function
    """
    output = tf.add(tf.matmul(input, w), b)
    return output


def mnist_features_labels(n_labels):
    """
    Gets the first <n> labels from the MNIST dataset
    :param n_labels: Number of labels to use
    :return: Tuple of feature list and label list
    """
    mnist_features = []
    mnist_labels = []

    mnist = input_data.read_data_sets('datasets/ud730/mnist', one_hot=True)

    for mnist_feature, mnist_label in zip(*mnist.train.next_batch(10000)):
        if mnist_label[:n_labels].any():
            mnist_features.append(mnist_feature)
            mnist_labels.append(mnist_label[:n_labels])

    return mnist_features, mnist_labels

# 28*28 image sizes is 784 features.
n_features = 784
n_labels = 3

features = tf.placeholder(tf.float32)
labels = tf.placeholder(tf.float32)

w = get_weights(n_features, n_labels)
b = get_biases(n_labels)

logits = linear(features, w, b)

train_features, train_labels = mnist_features_labels(n_labels)

with tf.Session as sess:
    sess.run(tf.global_variables_initializer())
    prediction = tf.nn.softmax(logits)

    # It seems that reduction_indices is deprecated. Is an alias for axis.
    cross_entropy = - tf.reduce_sum(labels * tf.log(prediction), axis=1)
    loss = tf.reduce_mean(cross_entropy)
    lr = 0.08

    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    _, l = sess.run(
        [optimizer, loss],
        feed_dict={features: train_features, labels: train_labels})

    print("Loss: {}".format(l))
