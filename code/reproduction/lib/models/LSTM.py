import tensorflow as tf
import numpy as np


class LSTM:
    def __init__(self, is_training, batch_length, vocabulary_filepath, embeddings_filepath,
                 batch_size=64, lstm_num_layers=2, lstm_num_hidden=128,
                 num_hidden_fc=128):
        self._is_training = is_training
        self._lstm_num_hidden = lstm_num_hidden
        self._lstm_num_layers = lstm_num_layers
        self._batch_size = batch_size
        self._num_hidden_fc = num_hidden_fc
        self._batch_length = batch_length

        self._load_embeddings(vocabulary_filepath)

        self.embedding_matrix = tf.get_variable(name='embeddings',
                                                shape=[self.vocab_length, 300])

    def assign_embedding_matrix(self, embedding_matrix):
        self.embedding_matrix.assign(embedding_matrix)

    def inference(self, inputs1, inputs2):
        with tf.variable_scope("SiameseNet", reuse=False):
            logits1 = self._define_network(inputs1)
        with tf.variable_scope("SiameseNet", reuse=True):
            logits2 = self._define_network(inputs2)
        return tf.map_fn(lambda logits: self._cosine_similarity(logits[0], logits[1]),
                         (logits1, logits2), dtype=tf.float32)

    def _define_network(self, inputs):
        print(inputs)
        dense_inputs = tf.map_fn(lambda x: self._string_to_dense(x, 300), inputs, dtype=tf.string)
        dense_inputs = tf.squeeze(dense_inputs, axis=1)
        padded_inputs = self.lookup_table.lookup(dense_inputs)
        embedded_inputs = tf.cast(tf.nn.embedding_lookup(self.embedding_matrix, padded_inputs), dtype=tf.float32)
        sequence_length = self._padded_length(padded_inputs, self.vocab_length - 2)

        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [self._lstm_cell(self._lstm_num_hidden) for _ in range(self._lstm_num_layers)])

        outputs, state = tf.nn.dynamic_rnn(cell=stacked_lstm,
                                           inputs=embedded_inputs,
                                           sequence_length=sequence_length,
                                           dtype=tf.float32)

        print(outputs)
        batch_size = tf.shape(outputs)[0]
        max_length = tf.shape(outputs)[1]
        out_size = int(outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (sequence_length - 1)
        flat = tf.reshape(outputs, [-1, out_size])
        relevant = tf.gather(flat, index)
        relevant = tf.layers.batch_normalization(relevant, training=self._is_training)
        logits = tf.layers.dense(relevant, self._num_hidden_fc, activation=None)
        return logits

    @staticmethod
    def _padded_length(data, val):
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
        loss = true_mask * (1.0 - cosine_similarity) + 5 * (1.0 - true_mask) * (
            tf.maximum(0.0, cosine_similarity - margin))
        return tf.reduce_mean(loss)

    def _load_embeddings(self, vocabulary_filepath):
        file = open(vocabulary_filepath)
        self.vocab_length = len(file.readlines())
        self.lookup_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=vocabulary_filepath,
                                                                    default_value=self.vocab_length + 1)

    @staticmethod
    def _string_to_dense(string, length):
        words = tf.string_split(string, delimiter=' ')
        dense = tf.sparse_to_dense(words.indices[0: length], [1, length],
                                   words.values[0: length], default_value='<PAD>',
                                   validate_indices=False)
        return dense

        # @staticmethod
        # def probabilities(logits):
        #     return tf.nn.softmax(logits)
        #
        # @staticmethod
        # def predictions(probabilities):
        #     return tf.argmax(probabilities, axis=-1)
