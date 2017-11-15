import tensorflow as tf


class SDQA:

    def __init__(self, activation_fn, is_training, input_shape=48536,
                 num_filter=32, num_hidden_fc=128, dropout=0.5, kernel_size=3):

        self.activation_fn = activation_fn
        self.is_training = is_training

        self.num_filter = num_filter
        self.num_hidden_fc = num_hidden_fc
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.input_shape = input_shape

    def define_network(self, features):

        with tf.variable_scope(None, default_name="ConvNet", reuse=True) as scope:
            input_layer = tf.reshape(features, [-1, self.input_shape])

            conv1 = tf.layers.conv1d(
                inputs=input_layer,
                filters=self.num_filter,
                kernel_size=self.kernel_size
            )

            pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)

            relu = tf.nn.relu(pool1)

            dropout = tf.layers.dropout(relu, rate=self.dropout, training=self.is_training)

            logits = tf.layers.dense(inputs=dropout, units=self.num_hidden_fc)

            return logits
