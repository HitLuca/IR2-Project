import tensorflow as tf
from tensorflow.contrib.data import TFRecordDataset


class LSTMDataset_TFRC:
    def __init__(self, files, batch_size, num_epochs, max_length, train):
        self.max_length = max_length
        self.dataset = TFRecordDataset(files)

        if train:
            self.dataset = self.dataset.map(
                self._parser, num_parallel_calls=4,
                output_buffer_size=batch_size * 4).shuffle(buffer_size=10000).repeat(num_epochs).batch(batch_size)

        else:
            self.dataset = self.dataset.map(
                self._parser, num_parallel_calls=4,
                output_buffer_size=batch_size * 4).shuffle(buffer_size=10000).repeat(1).batch(batch_size)

        self.iterator = self.dataset.make_initializable_iterator()

    def __call__(self):
        return self.iterator

    def _parser(self, example):

        features = tf.parse_single_example(example, features={
            'subject': tf.FixedLenFeature([], tf.string),
            'bestanswer': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })

        subject = tf.string_split([features['subject']])
        subject = tf.sparse_to_dense(subject.indices[0: self.max_length], [1, self.max_length],
                                     subject.values[0: self.max_length], default_value='<PAD>',
                                     validate_indices=False)
        subject_words = tf.reshape(subject, [self.max_length])
        # subject_words = tf.concat([subject_words, ['<EOS>']], 0)

        answer = tf.string_split([features['subject']])
        answer = tf.sparse_to_dense(answer.indices[0: self.max_length], [1, self.max_length],
                                    answer.values[0: self.max_length], default_value='<PAD>',
                                    validate_indices=False)
        answer_words = tf.reshape(answer, [self.max_length])
        # answer_words = tf.concat([answer_words, ['<EOS>']], 0)

        label = tf.reshape(features['label'], [1])
        label = tf.cast(label, dtype=tf.float32)

        return subject_words, answer_words, label
