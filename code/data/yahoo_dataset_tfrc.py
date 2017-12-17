import tensorflow as tf
from tensorflow.contrib.data import TFRecordDataset


class LSTMDataset_TFRC:
    def __init__(self, files, batch_size, num_epochs, max_length, train,
                 vocabulary_filepath='./lib/data/vocabulary.txt'):

        self.max_length = max_length - 1        # -1 to include a padding for EVERY sentence
        self.dataset = TFRecordDataset(files)

        self._load_lookup_table(vocabulary_filepath)

        if train:
            self.dataset = self.dataset.map(
                self._parser, num_parallel_calls=16,
                output_buffer_size=batch_size * 8).shuffle(buffer_size=10000).repeat(num_epochs).batch(batch_size)
        else:
            self.dataset = self.dataset.map(
                self._parser, num_parallel_calls=16,
                output_buffer_size=batch_size * 8).shuffle(buffer_size=10000).repeat(1).batch(batch_size)

        self.iterator = self.dataset.make_initializable_iterator()

    def _load_lookup_table(self, vocabulary_filepath):
        file = open(vocabulary_filepath)
        vocabulary = list(file.readlines())
        self.vocab_length = len(vocabulary)
        self.lookup_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=vocabulary_filepath,
                                                                    default_value=self.vocab_length-1)

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
        subject_words = tf.concat([subject_words, ['<PAD>']], 0)

        answer = tf.string_split([features['bestanswer']])
        answer = tf.sparse_to_dense(answer.indices[0: self.max_length], [1, self.max_length],
                                    answer.values[0: self.max_length], default_value='<PAD>',
                                    validate_indices=False)
        answer_words = tf.reshape(answer, [self.max_length])
        answer_words = tf.concat([answer_words, ['<PAD>']], 0)

        subject_words = self.lookup_table.lookup(subject_words)
        answer_words = self.lookup_table.lookup(answer_words)

        label = tf.reshape(features['label'], [1])
        label = tf.cast(label, dtype=tf.float32)

        return subject_words, answer_words, label
