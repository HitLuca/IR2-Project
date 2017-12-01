import tensorflow as tf


class LSTM:
    def __init__(self, batch_size=64, lstm_num_layers=2, lstm_num_hidden=128,
                 num_hidden_fc=128, padding_value=0):

        self._lstm_num_hidden = lstm_num_hidden
        self._lstm_num_layers = lstm_num_layers
        self._batch_size = batch_size
        self._num_hidden_fc = num_hidden_fc
        self._padding_value = padding_value

    def inference(self, inputs1, inputs2):
        with tf.variable_scope("SiameseNet", reuse=False):
            logits1 = self._define_network(inputs1)
        with tf.variable_scope("SiameseNet", reuse=True):
            logits2 = self._define_network(inputs2)
        return tf.map_fn(lambda logits: self._cosine_similarity(logits[0], logits[1]),
                         (logits1, logits2), dtype=tf.float32)

    def _define_network(self, inputs):
        inputs = tf.expand_dims(inputs, axis=-1)

        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [self._lstm_cell(self._lstm_num_hidden) for _ in range(self._lstm_num_layers)])

        sequence_length = self._length(inputs, self._padding_value)
        outputs, state = tf.nn.dynamic_rnn(cell=stacked_lstm,
                                           inputs=inputs,
                                           sequence_length=sequence_length,
                                           dtype=tf.float32)

        batch_size = tf.shape(outputs)[0]
        max_length = tf.shape(outputs)[1]
        out_size = int(outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (sequence_length - 1)
        flat = tf.reshape(outputs, [-1, out_size])
        relevant = tf.gather(flat, index)
        logits = tf.layers.dense(relevant, self._num_hidden_fc)
        return logits

    @staticmethod
    def _length(data, val):
        tmp_indices = tf.where(tf.equal(data, val))
        result = tf.segment_min(tmp_indices[:, 1], tmp_indices[:, 0])
        return tf.cast(result, tf.int32)

    @staticmethod
    def _lstm_cell(lstm_num_hidden):
        return tf.contrib.rnn.BasicLSTMCell(lstm_num_hidden)

    @staticmethod
    def train_step(loss, learning_rate):
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)

    @staticmethod
    def accuracy(label, cosine_similarity):
        related_pred = tf.cast(tf.greater_equal(cosine_similarity, 0), dtype=tf.float32)
        related_label = tf.cast(tf.equal(label, 1.0), dtype=tf.float32)

        return tf.reduce_mean(tf.cast(tf.equal(related_label, related_pred), dtype=tf.float32))

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

    # @staticmethod
    # def probabilities(logits):
    #     return tf.nn.softmax(logits)
    #
    # @staticmethod
    # def predictions(probabilities):
    #     return tf.argmax(probabilities, axis=-1)
