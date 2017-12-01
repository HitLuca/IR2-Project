import glob

import pandas as pd
import numpy as np
import pickle
import os


# np.random.seed(42)


class TrigramsDataset(object):
    def __init__(self, batch_size, filename, data_dir):
        self._batch_size = batch_size

        self._total_steps = 0
        self._step_in_epoch = 0
        self._epoch = 0

        self._subject_encoded = None
        self._bestanswer_encoded = None
        self.dataset_size = 0
        self._init_data(os.path.join(data_dir, filename))

        self._indices_neg = np.random.permutation(self.dataset_size)

        self._vocabulary = pickle.load(open(os.path.join(data_dir, 'vocab_dict.p'), 'rb'))
        self.vocabulary_size = len(self._vocabulary)

    def next_batch(self, is_one_hot_encoding=False):
        if int(((self._total_steps + 1) * self._batch_size) / self.dataset_size) > self._epoch:
            self._shuffle_subject_bestanswer_lists()
            self._indices_neg = np.random.permutation(self.dataset_size)
            self._epoch += 1
            self._step_in_epoch = 0

        slice_pos = range(self._step_in_epoch * self._batch_size,
                          int(self._step_in_epoch * self._batch_size + 0.5 * self._batch_size))
        X1_pos = self._subject_encoded[slice_pos]
        X2_pos = self._bestanswer_encoded[slice_pos]
        y_pos = np.ones(len(X1_pos), dtype=int)

        slice_neg = slice_pos
        X1_neg = self._subject_encoded[self._indices_neg[slice_neg]]
        X2_neg = self._bestanswer_encoded[self._indices_neg[slice_neg]]
        y_neg = np.zeros(len(X1_neg), dtype=int)

        X1 = [*X1_pos, *X1_neg]
        X2 = [*X2_pos, *X2_neg]
        y = [*y_pos, *y_neg]

        temp = list(zip(X1, X2, y))
        np.random.shuffle(temp)
        Q, A, y = zip(*temp)

        self._step_in_epoch += 1
        self._total_steps += 1

        if is_one_hot_encoding:
            Q = self._one_hot_encoding(Q)
            A = self._one_hot_encoding(A)
        return np.array(Q), np.array(A), np.array(y)

    def dataset_statistics(self):
        print('batch_size:', self._batch_size)
        print('total_steps:', self._total_steps)
        print('steps_in_epoch:', self._step_in_epoch)
        print('epoch:', self._epoch)
        print('dataset_size:', self.dataset_size)
        print('vocabulary_size:', self.vocabulary_size)

    def _init_data(self, filename):
        if 'minimal_sparse' in filename:
            f_names = glob.glob(filename + '*.p')
            df = pd.concat([pd.read_pickle(f_name) for f_name in f_names], ignore_index=True)
        else:
            df = pd.read_pickle(filename)

        df = self._shuffle_df(df)
        df = df.loc[:, ['subject_encoded', 'bestanswer_encoded']]
        self._subject_encoded = df.subject_encoded.values
        self._bestanswer_encoded = df.bestanswer_encoded.values
        self.dataset_size = len(self._subject_encoded)

    def _one_hot_encoding(self, batch):
        zero_matrix = np.zeros([self._batch_size, self.vocabulary_size], dtype=np.int64)
        for i in range(len(batch)):
            zero_matrix[i, batch[i]] = 1
        return zero_matrix

    def _shuffle_subject_bestanswer_lists(self):
        indices = np.random.permutation(self.dataset_size)
        self._subject_encoded = self._subject_encoded[indices]
        self._bestanswer_encoded = self._bestanswer_encoded[indices]

    @staticmethod
    def _shuffle_df(df):
        indices = np.random.permutation(len(df))
        df = df.iloc[indices]
        return df


class LSTMDataset(object):
    def __init__(self, batch_size, filename, data_dir):
        self._batch_size = batch_size
        self._total_steps = 0
        self._step_in_epoch = 0
        self._epoch = 0

        self._subject_preprocessed = None
        self._bestanswer_preprocessed = None
        self.dataset_size = 0

        self._init_data(os.path.join(data_dir, filename))
        self._indices_neg = np.random.permutation(self.dataset_size)

    def next_batch(self):
        if int(((self._total_steps + 1) * self._batch_size) / self.dataset_size) > self._epoch:
            self._shuffle_subject_bestanswer_lists()
            self._indices_neg = np.random.permutation(self.dataset_size)
            self._epoch += 1
            self._step_in_epoch = 0

        slice_pos = range(self._step_in_epoch * self._batch_size,
                          int(self._step_in_epoch * self._batch_size + 0.5 * self._batch_size))
        X1_pos = self._subject_preprocessed[slice_pos]
        X2_pos = self._bestanswer_preprocessed[slice_pos]
        y_pos = np.ones(len(X1_pos), dtype=int)

        slice_neg = slice_pos
        X1_neg = self._subject_preprocessed[self._indices_neg[slice_neg]]
        X2_neg = self._bestanswer_preprocessed[self._indices_neg[slice_neg]]
        y_neg = np.zeros(len(X1_neg), dtype=int)

        X1 = [*X1_pos, *X1_neg]
        X2 = [*X2_pos, *X2_neg]
        y = [*y_pos, *y_neg]

        temp = list(zip(X1, X2, y))
        np.random.shuffle(temp)
        Q, A, y = zip(*temp)

        self._step_in_epoch += 1
        self._total_steps += 1

        return np.array(Q), np.array(A), np.array(y)

    def dataset_statistics(self):
        print('batch_size:', self._batch_size)
        print('total_steps:', self._total_steps)
        print('steps_in_epoch:', self._step_in_epoch)
        print('epoch:', self._epoch)
        print('dataset_size:', self.dataset_size)

    def _init_data(self, filename):
        df = pd.read_pickle(filename)
        df = self._shuffle_df(df)
        df = df.loc[:, ['subject_preprocessed', 'bestanswer_preprocessed']]
        self._subject_preprocessed = df.subject_preprocessed.values
        self._bestanswer_preprocessed = df.bestanswer_preprocessed.values
        self.dataset_size = len(self._subject_preprocessed)

    def _shuffle_subject_bestanswer_lists(self):
        indices = np.random.permutation(self.dataset_size)
        self._subject_preprocessed = self._subject_preprocessed[indices]
        self._bestanswer_preprocessed = self._bestanswer_preprocessed[indices]

    @staticmethod
    def _shuffle_df(df):
        indices = np.random.permutation(len(df))
        df = df.iloc[indices]
        return df
