import tensorflow as tf


class LSTM:
    def __init__(self, is_training, vocabulary_filepath, train_embedding=False,
                 batch_size=64, lstm_num_layers=2, lstm_num_hidden=128,
                 num_hidden_fc=128, use_tfrecord=True):
        self._is_training = is_training
        self._lstm_num_hidden = lstm_num_hidden
        self._lstm_num_layers = lstm_num_layers
        self._batch_size = batch_size
        self._num_hidden_fc = num_hidden_fc
        self._tfrecord = use_tfrecord

        self.global_step = tf.Variable(0, trainable=False)

        self._load_embeddings(vocabulary_filepath)

        if train_embedding:             # train our own embedding matrix
            self.e_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
            self.embedding_matrix = tf.get_variable(shape=[self.vocab_length, 300],
                                                    name='embeddings', initializer=self.e_initializer)
        else:
            self.embedding_matrix = tf.get_variable(name='embeddings',
                                                    shape=[self.vocab_length, 300],
                                                    trainable=False)

    def assign_embedding_matrix(self, embedding_matrix):
        self.embedding_matrix.assign(embedding_matrix)

    def inference(self, inputs1, inputs2):
        with tf.variable_scope("SiameseNet", reuse=False):
            logits1 = self._define_network(inputs1)
        with tf.variable_scope("SiameseNet", reuse=True):
            logits2 = self._define_network(inputs2)

        # concatenate the two hidden states
        combined = tf.concat([logits1, logits2], axis=1)

        output_fc = tf.layers.dense(combined, 1)

        return output_fc, tf.nn.sigmoid(output_fc)

        # original loss function
        # # TODO: For the old loss function, calculate cos similarity
        # return tf.map_fn(lambda logits: self._cosine_similarity(logits[0], logits[1]),
        #                  (logits1, logits2), dtype=tf.float32)

    def _define_network(self, inputs):
        if self._tfrecord is False:
            inputs = tf.expand_dims(inputs, dim=-1)
            tf.reduce_max(tf.size(inputs))
            dense_inputs = tf.map_fn(lambda x: self._string_to_dense(x, 200), inputs, dtype=tf.string)
            dense_inputs = tf.squeeze(dense_inputs, axis=1)
            padded_inputs = self.lookup_table.lookup(dense_inputs)

            embedded_inputs = tf.cast(tf.nn.embedding_lookup(self.embedding_matrix, padded_inputs), dtype=tf.float32)
            sequence_length = self._padded_length(padded_inputs, self.vocab_length - 2)
        else:
            embedded_inputs = tf.cast(tf.nn.embedding_lookup(self.embedding_matrix, inputs), dtype=tf.float32)
            sequence_length = self._padded_length(inputs, self.vocab_length - 2)

        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [self._lstm_cell(self._lstm_num_hidden) for _ in range(self._lstm_num_layers)])

        outputs, state = tf.nn.dynamic_rnn(cell=stacked_lstm,
                                           inputs=embedded_inputs,
                                           sequence_length=sequence_length,
                                           dtype=tf.float32)

        logits = state[0][1]
        return logits

    @staticmethod
    def _padded_length(data, val):
        tmp_indices = tf.where(tf.equal(data, val))
        result = tf.segment_min(tmp_indices[:, 1], tmp_indices[:, 0])
        return tf.cast(result, tf.int32)

    @staticmethod
    def _lstm_cell(lstm_num_hidden):
        return tf.contrib.rnn.LSTMCell(num_units=lstm_num_hidden, initializer=tf.orthogonal_initializer)

    def train_step(self, loss, learning_rate):
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #
        # with tf.control_dependencies(update_ops):
        #     train_op = optimizer.minimize(loss)

        # return train_op
        return tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=self.global_step)

    @staticmethod
    def predict(cosine_similarity):
        cosine_similarity = tf.nn.sigmoid(cosine_similarity)
        cosine_similarity = tf.round(cosine_similarity)

        return cosine_similarity

    @staticmethod
    def accuracy(label, cosine_similarity):

        cosine_similarity = tf.nn.sigmoid(cosine_similarity)
        cosine_similarity = tf.round(cosine_similarity)

        acc = tf.cast(tf.equal(label, cosine_similarity), tf.float32)

        return tf.reduce_mean(acc)

    @staticmethod
    def _cosine_similarity(logit_1, logit_2):
        normalized_logit_1 = tf.nn.l2_normalize(logit_1, dim=0)
        normalized_logit_2 = tf.nn.l2_normalize(logit_2, dim=0)
        return 1.0 - tf.losses.cosine_distance(normalized_logit_1, normalized_logit_2, dim=0)

    @staticmethod
    def loss(labels, cosine_similarity):
        # # TODO: Old loss, to be removed, or place into a different script
        # true_mask = tf.cast(tf.equal(labels, tf.ones(shape=tf.shape(labels))), tf.float32)
        # loss = true_mask * (1.0 - cosine_similarity) + (1.0 - true_mask) * (
        #     tf.maximum(0.0, cosine_similarity - 0.3))

        # NEW LOSS
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=cosine_similarity)
        # loss = tf.squared_difference(labels, cosine_similarity)

        return tf.reduce_mean(loss)

    def _load_embeddings(self, vocabulary_filepath):
        file = open(vocabulary_filepath)
        vocabulary = list(file.readlines())
        self.vocab_length = len(vocabulary)
        if self._tfrecord is False:
            self.lookup_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=vocabulary_filepath,
                                                                        default_value=self.vocab_length-1)

    @staticmethod
    def _string_to_dense(string, length):
        words = tf.string_split(string, delimiter=' ')
        dense = tf.sparse_to_dense(words.indices[0: length], [1, length],
                                   words.values[0: length], default_value='<PAD>',
                                   validate_indices=False)
        return dense
