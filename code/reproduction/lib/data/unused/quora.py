import tensorflow as tf
from tensorflow.contrib.data import TFRecordDataset


class QuoraLstm:
    def __init__(self, files, batch_size, num_epochs, max_length, train):
        self.max_length = max_length
        self.dataset = TFRecordDataset(files)
        self.batch_size = batch_size

        if train:
            self.dataset = self.dataset.map(
                self._parser_train, num_parallel_calls=4,
                output_buffer_size=batch_size * 4).shuffle(buffer_size=10000).repeat(num_epochs).batch(batch_size)
        else:
            self.dataset = self.dataset.map(
                self._parser_train, num_parallel_calls=4,
                output_buffer_size=batch_size * 4).shuffle(buffer_size=10000).repeat(num_epochs).batch(batch_size)

        self.iterator = self.dataset.make_initializable_iterator()

    def __call__(self):
        return self.iterator

    def _parser_train(self, example):

        features = tf.parse_single_example(example, features={
            'id': tf.FixedLenFeature([], tf.int64),
            'qid1': tf.FixedLenFeature([], tf.int64),
            'qid2': tf.FixedLenFeature([], tf.int64),
            'question1': tf.FixedLenFeature([], tf.string),
            'question2': tf.FixedLenFeature([], tf.string),
            'is_duplicate': tf.FixedLenFeature([], tf.int64),
        })

        id = features['id']
        qid1 = features['qid1']
        qid2 = features['qid2']
        # q1_words = features['question1']
        # q2_words = features['question2']
        is_duplicate = features['is_duplicate']

        # id = tf.reshape(features['id'], [1])
        # qid1 = tf.reshape(features['qid1'], [1])
        # qid2 = tf.reshape(features['qid2'], [1])

        question1 = tf.string_split([features['question1']])
        print(question1)

        print(question1.values)
        sparse_val2 = tf.string_to_number(question1.values[0:self.max_length], out_type=tf.int64)
        print(sparse_val2)
        # sparse_ind = tf.string_to_number(question1.values[0:self.max_length])
        print(question1.indices[0: self.max_length])

        sparse_val = tf.expand_dims(sparse_val2, axis=1)
        tmp = tf.zeros(shape=tf.shape(sparse_val), dtype=tf.int64)
        sparse_idx = tf.concat([tmp, sparse_val], axis=1)
        print(sparse_idx)
        # mlen = tf.cast(sparse_idx.shape[0], dtype=tf.int64)

        q1_words = tf.sparse_to_dense(sparse_indices=sparse_idx,
                                      output_shape=[1, self.max_length],
                                      # sparse_values=question1.values[0: self.max_length],
                                      sparse_values=1,
                                      default_value=0,
                                      validate_indices=False)

        # q1_words = tf.sparse_tensor_to_dense(sp_input=question1, default_value='0', validate_indices=False)

        # q1_words = features['question1']
        q2_words = features['question1']

        return id, qid1, qid2, q1_words, q2_words, is_duplicate, sparse_idx
