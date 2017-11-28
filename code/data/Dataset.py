import glob

import pandas as pd
import numpy as np
import pickle
import os
np.random.seed(42)

class Dataset(object):
    def __init__(self, batch_size, f_name, data_dir = ''):
        f_name = os.path.join(data_dir, f_name)
        self.batch_size = batch_size
        self.total_steps = 0
        self.step_in_epoch = 0
        self.epoch = 0
        self.subject_encoded = None
        self.bestanswer_encoded = None
        self.init_data(f_name)
        self.dataset_size = len(self.subject_encoded)
        self.indices_neg = np.random.permutation(self.dataset_size)
        self.vocabulary = pickle.load(open(os.path.join(data_dir,'vocab_dict.p'), 'rb'))
        self.vocabulary_size = len(self.vocabulary)

    def init_data(self, f_name):
        df = None
        if 'minimal_sparse' in f_name:
            f_names = glob.glob(f_name+'*.p')
            df = pd.concat([pd.read_pickle(f_name) for f_name in f_names], ignore_index=True)
        else:
            df = pd.read_pickle(f_name)
        df = self.shuffle_df(df)
        df = df.loc[:, ['subject_encoded', 'bestanswer_encoded']]
        self.subject_encoded = df.subject_encoded.values
        self.bestanswer_encoded = df.bestanswer_encoded.values

    def one_hot_encoding(self, list_of_lists):
        zero_matrix = np.zeros([self.batch_size, self.vocabulary_size], dtype=np.int64)
        for i in range(len(list_of_lists)):
            zero_matrix[i, list_of_lists[i]] = 1
        return zero_matrix

    def shuffle_df(self, df):
        indices = np.random.permutation(len(df))
        df = df.iloc[indices]
        return df

    def shuffle_subject_bestanswer_lists(self):
        indices = np.random.permutation(self.dataset_size)
        self.subject_encoded = self.subject_encoded[indices]
        self.bestanswer_encoded = self.bestanswer_encoded[indices]

    def next_batch(self, is_one_hot_encoding=False):
        if int(((self.total_steps + 1) * self.batch_size) / self.dataset_size) > self.epoch:
            self.shuffle_subject_bestanswer_lists()
            self.indices_neg = np.random.permutation(self.dataset_size)
            self.epoch += 1
            self.step_in_epoch = 0

        slice_pos = range(self.step_in_epoch * self.batch_size,
                          int(self.step_in_epoch * self.batch_size + 0.5 * self.batch_size))
        X1_pos = self.subject_encoded[slice_pos]
        X2_pos = self.bestanswer_encoded[slice_pos]
        y_pos = np.ones(len(X1_pos), dtype=int)

        slice_neg = slice_pos
        X1_neg = self.subject_encoded[self.indices_neg[slice_neg]]
        X2_neg = self.bestanswer_encoded[self.indices_neg[slice_neg]]
        y_neg = np.zeros(len(X1_neg), dtype=int)

        X1 = [*X1_pos, *X1_neg]
        X2 = [*X2_pos, *X2_neg]
        y = [*y_pos, *y_neg]

        temp = list(zip(X1, X2, y))
        np.random.shuffle(temp)
        Q, A, y = zip(*temp)

        self.step_in_epoch += 1
        self.total_steps += 1

        if is_one_hot_encoding:
            Q = self.one_hot_encoding(Q)
            A = self.one_hot_encoding(A)
        return np.array(Q), np.array(A), np.array(y)

    def print_internal_state(self):
        print('batch_size: ', self.batch_size)
        print('total_steps: ', self.total_steps)
        print('steps_in_epoch: ', self.step_in_epoch)
        print('epoch: ', self.epoch)
        print('dataset_size: ', self.dataset_size)
