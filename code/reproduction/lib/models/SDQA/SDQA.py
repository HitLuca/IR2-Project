import tensorflow as tf
from scipy.constants.codata import unit


class SDQA:
    def __init__(self, is_training, vocabulary_size, activation_fn=tf.nn.relu,
                 num_filter=32, num_hidden_fc=128, dropout_rate=0.0, kernel_size=10):
        self.activation_fn = activation_fn
        self.is_training = is_training
        self.num_filter = num_filter
        self.num_hidden_fc = num_hidden_fc
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.vocabulary_size = vocabulary_size

    def _define_network(self, input):
        with tf.variable_scope("Conv1"):
            conv1 = tf.layers.conv1d(
                inputs=input,
                filters=self.num_filter,
                kernel_size=self.kernel_size,
                activation=self.activation_fn
            )
            conv1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=100, strides=10)
            conv1 = tf.layers.dropout(conv1, rate=self.dropout_rate, training=self.is_training)

        with tf.variable_scope("Conv2"):
            conv2 = tf.layers.conv1d(
                inputs=conv1,
                filters=self.num_filter,
                kernel_size=self.kernel_size,
                activation=self.activation_fn
            )
            conv2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=100, strides=5)
            conv2 = tf.layers.dropout(conv2, rate=self.dropout_rate, training=self.is_training)

        with tf.variable_scope("Conv3"):
            conv3 = tf.layers.conv1d(
                inputs=conv2,
                filters=self.num_filter,
                kernel_size=self.kernel_size,
                activation=self.activation_fn
            )
            conv3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=100, strides=5)
            conv3 = tf.layers.dropout(conv3, rate=self.dropout_rate, training=self.is_training)

        with tf.variable_scope("FC"):
            flattened = tf.contrib.layers.flatten(conv3)
            logits = tf.layers.dense(inputs=flattened, units=self.num_hidden_fc)
        return logits

    def inference(self, inputs1, inputs2):
        inputs1 = tf.expand_dims(inputs1, axis=-1)
        inputs2 = tf.expand_dims(inputs2, axis=-1)

        with tf.variable_scope("SiameseNet", reuse=False):
            logits1 = self._define_network(inputs1)
        with tf.variable_scope("SiameseNet", reuse=True):
            logits2 = self._define_network(inputs2)
        return tf.map_fn(lambda logits: self._cosine_similarity(logits[0], logits[1]),
                         (logits1, logits2), dtype=tf.float32)

    @staticmethod
    def _cosine_similarity(logit_1, logit_2):
        normalized_logit_1 = tf.nn.l2_normalize(logit_1, dim=0)
        normalized_logit_2 = tf.nn.l2_normalize(logit_2, dim=0)
        return 1.0 - tf.losses.cosine_distance(normalized_logit_1, normalized_logit_2, dim=0)

    @staticmethod
    def loss(labels, cosine_similarity, margin=0.3):
        true_mask = tf.cast(tf.equal(labels, tf.ones(shape=tf.shape(labels))), tf.float32)
        loss = true_mask * (1.0 - cosine_similarity) + (1.0 - true_mask) * (tf.maximum(0.0, cosine_similarity - margin))
        return tf.reduce_mean(loss)

    @staticmethod
    def train_step(loss, learning_rate):
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)

    @staticmethod
    def accuracy(label, cosine_similarity):
        related_pred = tf.cast(tf.greater_equal(cosine_similarity, 0), dtype=tf.float32)
        related_label = tf.cast(tf.equal(label, 1.0), dtype=tf.float32)

        return tf.reduce_mean(tf.cast(tf.equal(related_label, related_pred), dtype=tf.float32))
