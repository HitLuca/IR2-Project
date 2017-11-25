import tensorflow as tf


class SDQA:
    def __init__(self, is_training, input_shape, activation_fn=tf.nn.relu,
                 num_filter=64, num_hidden_fc=128, dropout=0.5, kernel_size=10,
                 learning_rate=0.01):

        self.activation_fn = activation_fn
        self.is_training = is_training
        self.learning_rate = learning_rate

        self.num_filter = num_filter
        self.num_hidden_fc = num_hidden_fc
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.input_shape = input_shape

        self.input1 = tf.placeholder(tf.float32, shape=[None, input_shape, 1])
        self.input2 = tf.placeholder(tf.float32, shape=[None, input_shape, 1])

        # with tf.variable_scope("SiameseNet", reuse=tf.AUTO_REUSE) as scope:
        #     self.logits1 = self._define_network(self.input1)
        #     self.logits2 = self._define_network(self.input2)

        with tf.variable_scope("SiameseNet", reuse=False):
            self.logits1 = self._define_network(self.input1)
        with tf.variable_scope("SiameseNet", reuse=True):
            self.logits2 = self._define_network(self.input2)

    def _define_network(self, features):
        # with tf.variable_scope(None, default_name="PreFC") as scope:
        #     pre_fc = tf.layers.dense(inputs=features, units=128)

        with tf.variable_scope(None, default_name="Conv1") as scope:
            conv1 = tf.layers.conv1d(
                inputs=features,
                filters=self.num_filter,
                kernel_size=self.kernel_size,
                activation=self.activation_fn
            )
            conv1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=100, strides=5)
            conv1 = tf.layers.dropout(conv1, rate=self.dropout, training=self.is_training)

        with tf.variable_scope(None, default_name="Conv2") as scope:
            conv2 = tf.layers.conv1d(
                inputs=conv1,
                filters=self.num_filter,
                kernel_size=self.kernel_size,
                activation=self.activation_fn
            )
            conv2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=100, strides=5)
            conv2 = tf.layers.dropout(conv2, rate=self.dropout, training=self.is_training)

        with tf.variable_scope(None, default_name="Conv3") as scope:
            conv3 = tf.layers.conv1d(
                inputs=conv2,
                filters=self.num_filter,
                kernel_size=self.kernel_size,
                activation=self.activation_fn
            )
            conv3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=100, strides=5)
            conv3 = tf.layers.dropout(conv3, rate=self.dropout, training=self.is_training)

        with tf.variable_scope(None, default_name="Final_FC_Layers") as scope:
            flattened = tf.contrib.layers.flatten(conv3)
            logits = tf.layers.dense(inputs=flattened, units=self.num_hidden_fc)

        return logits

    def inference(self):
        return tf.map_fn(lambda x: self.find_cosine_dist(x[0], x[1]),
                         (self.logits1, self.logits2), dtype=tf.float32)

    @staticmethod
    def find_cosine_dist(x1, x2):
        normalized_x1 = tf.nn.l2_normalize(x1, dim=0)
        normalized_x2 = tf.nn.l2_normalize(x2, dim=0)
        return tf.losses.cosine_distance(normalized_x1, normalized_x2, dim=0)

    def loss(self, label, margin=1.0):

        # for those y = 1
        true_mask = tf.cast(tf.equal(label, tf.ones(shape=tf.shape(label))), tf.float32)
        # for those y = -1
        false_mask = 1.0 - true_mask

        cosine_dist = tf.map_fn(lambda x: self.find_cosine_dist(x[0], x[1]),
                                (self.logits1, self.logits2), dtype=tf.float32)

        loss = true_mask * (1.0 - cosine_dist) + false_mask * (tf.maximum(0.0, cosine_dist - margin))

        return tf.reduce_sum(loss), cosine_dist

    def train_step(self, loss):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

    @staticmethod
    def accuracy(label, cosine_dist, threshold):

        related_pred = tf.cast(tf.greater_equal(cosine_dist, threshold), dtype=tf.float32)
        related_label = tf.cast(tf.equal(label, 1.0), dtype=tf.float32)

        return tf.reduce_mean(tf.cast(tf.equal(related_pred, related_label), dtype=tf.float32))
