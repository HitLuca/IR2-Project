import tensorflow as tf


class SDQA:
    def __init__(self, activation_fn, is_training, input_shape=48536,
                 num_filter=32, num_hidden_fc=128, dropout=0.5, kernel_size=3,
                 learning_rate=0.01, batch_size=128):

        self.activation_fn = activation_fn
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.num_filter = num_filter
        self.num_hidden_fc = num_hidden_fc
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.input_shape = input_shape

        self.input1 = tf.placeholder(tf.float32, shape=[None, input_shape, 1])
        self.input2 = tf.placeholder(tf.float32, shape=[None, input_shape, 1])

        with tf.variable_scope("SiameseNet", reuse=False):
            self.logits1 = self._define_network(self.input1)
        with tf.variable_scope("SiameseNet", reuse=True):
            self.logits2 = self._define_network(self.input2)

    def _define_network(self, features):
        conv1 = tf.layers.conv1d(
            inputs=features,
            filters=self.num_filter,
            kernel_size=self.kernel_size,
            activation=self.activation_fn
        )

        pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)

        dropout = tf.layers.dropout(pool1, rate=self.dropout, training=self.is_training)

        logits = tf.layers.dense(inputs=dropout, units=self.num_hidden_fc)

        return logits

    def inference(self):
        return tf.losses.cosine_distance(self.logits1, self.logits2, dim=0)

    def loss(self, label, margin=1):

        cosine_dist = tf.losses.cosine_distance(self.logits1, self.logits2, dim=0)

        true_mask = tf.cast(tf.equal(label, tf.ones(shape=tf.shape(label))), tf.float32)
        false_mask = 1 - true_mask

        loss = true_mask * (1 - cosine_dist) + false_mask * (tf.maximum(0, cosine_dist - margin))

        return tf.reduce_sum(loss)

    def train_step(self, loss):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

    def accuracy(self, label, cosine_dist, threshold):

        related_pred = tf.cast(tf.greater_equal(cosine_dist, threshold), dtype=tf.float32)
        related_label = tf.cast(tf.equal(label, 1.0), dtype=tf.float32)

        return tf.reduce_mean(tf.cast(tf.equal(related_pred, related_label), dtype=tf.float32))
