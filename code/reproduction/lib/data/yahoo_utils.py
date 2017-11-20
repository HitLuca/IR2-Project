import pandas as pd
import pickle
import numpy as np
import random

pd.options.mode.chained_assignment = None


class YahooDataset:
    def __init__(self):
        self.dataset_filepath = './../../../data/Yahoo/part_1.p'
        self.vocabulary_filepath = './../../../data/Yahoo/vocabulary.p'
        self.df = None
        self.batch_index = 0

    def init_dataset(self, shuffle):
        self.load_dataset()
        self.df = self.df.T.to_dict().values()
        if shuffle:
            random.shuffle(self.df)

    def get_next_batch(self, batch_size):
        self.batch_index += batch_size
        return self.df[self.batch_index - batch_size:self.batch_index]

    def load_dataset(self):
        self.df = pd.read_pickle(self.dataset_filepath)

    def generate_trigrams(self):
        self.load_dataset()

        df = self.df
        sel_1 = (df['content'] == '')
        sel_2 = df['subject'] == df['content']
        sel_3 = ~(df['bestanswer'] == '')
        sel_4 = (df["bestanswer"].str.len() >= 3) & (df["subject"].str.len() >= 3)
        sel = (sel_1 | sel_2) & sel_3 & sel_4
        df_sel = df[sel]

        all_strings = list(df_sel['subject'].values) + list(df_sel['bestanswer'].values)
        len(all_strings)
        trigrams_vocabulary = set()
        for string in all_strings:
            for i in range(len(string) - 2):
                trigram = string[i:i + 3]
                trigrams_vocabulary.add(trigram)
        trigrams_vocabulary = list(trigrams_vocabulary)
        self._save_to_binary(trigrams_vocabulary, self.vocabulary_filepath)
        trigrams_vocabulary = np.array(trigrams_vocabulary)

        df_sel.loc["tri_subject"] = None
        df_sel["tri_bestanswer"] = None
        df_sel.head(1)

        sort_idx = np.array(trigrams_vocabulary).argsort()

        subject_list = list(df_sel['subject'].values)
        tri_subject_list = subject_list.copy()
        bestanswer_list = list(df_sel['bestanswer'].values)
        tri_bestanswer_list = bestanswer_list.copy()

        for i in range(len(subject_list)):
            subject = str(subject_list[i])
            trigrams = self._extract_trigrams(subject, trigrams_vocabulary, sort_idx)
            tri_subject_list[i] = trigrams

            bestanswer = str(bestanswer_list[i])
            trigrams = self._extract_trigrams(bestanswer, trigrams_vocabulary, sort_idx)
            tri_bestanswer_list[i] = trigrams

        df_sel['tri_subject'] = pd.Series(tri_subject_list)
        df_sel['tri_bestanswer'] = pd.Series(tri_bestanswer_list)

        df_sel.to_pickle(self.dataset_filepath)

    @staticmethod
    def _save_to_binary(data, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def _extract_trigrams(self, string, trigrams_vocabulary, sort_idx):
        trigrams = np.unique([string[j:j + 3] for j in range(len(string) - 2)])
        if len(trigrams) == 0:
            print("no trigrams extracted!", string)
            raise NotImplementedError

        return np.unique(sort_idx[np.searchsorted(trigrams_vocabulary, trigrams, sorter=sort_idx)])


