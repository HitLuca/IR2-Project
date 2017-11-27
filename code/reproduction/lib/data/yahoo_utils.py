import pandas as pd
import pickle
import numpy as np
import random
import os

pd.options.mode.chained_assignment = None


class YahooDataset:
    class MissingVocabularyException(Exception):
        pass

    def __init__(self, dataset_folder, dataset_filename):
        self.dataset_folder = dataset_folder + '/'
        self.dataset_filepath = self.dataset_folder + dataset_filename
        self.vocabulary_filepath = self.dataset_folder + 'vocabulary_' + dataset_filename
        self.df = None
        self.vocabulary = None
        self.batch_index = 0
        self.batch_indexes = None


        '''
        init:
        1. unpickle the pickle named self.dataset_filepath
        2. load the pickle
        3. get or make the trigrams

        '''
        #1
        print('loading_dataset...')
        self._load_dataset()
        print('loaded')
        #2
        if self._vocabulary_exists():
            print('loading vocabulary...')
            self._load_vocabulary()
            print('loaded')
        else: #3
            if not self._check_for_trigrams():
                print('no trigrams in the dataset!')
                print('generating trigrams...')
                self._add_trigrams_to_dataset()
                print('generated')

        print(np.shape(self.df))
        # negative_indexes = [random.choice(list(range(N)).remove(i) for i in range(N)]

        

    def init_dataset(self, shuffle):
        self.batch_indexes = np.arange(len(self.df.index))
        if shuffle:
            print('shuffling...')
            np.random.shuffle(self.batch_indexes)
            print('shuffled')

    def get_next_batch(self, batch_size):
        batch_size = int(batch_size*0.5)
        self.batch_index += batch_size
        selected_data = self.df.iloc[self.batch_indexes[self.batch_index - batch_size:self.batch_index], :]
        # selected_data = self.df.ix[self.batch_indexes]
        selected_data['relevance'] = 1.0
        negative_samples = self.df.sample(n=batch_size)
        negative_samples['relevance'] = 0.0
        sample = pd.concat([selected_data, negative_samples], ignore_index=True)
        sample = sample.reindex(np.random.permutation(sample.index))
        # TODO: Temporary solution to avoid NaNs
        return self._convert_pandas_to_list(sample.dropna())

    def get_vocabulary_size(self):
        if self.vocabulary is not None:
            return len(self.vocabulary)
        else:
            return self.MissingVocabularyException

    def _check_for_trigrams(self):
        if 'tri_subject' in self.df.columns and 'tri_bestanswer' in self.df.columns:
            return True
        else:
            return False

    def _vocabulary_exists(self):
        return os.path.exists(self.vocabulary_filepath)

    def _load_dataset(self):
        self.df = pd.read_pickle(self.dataset_filepath)


    def _load_vocabulary(self):
        self.vocabulary = np.array(self._load_from_binary(self.vocabulary_filepath))

    def _shuffle_dataset(self):
        random.shuffle(self.df)

    def _generate_vocabulary(self, strings):
        self.vocabulary = set()
        for string in strings:
            for i in range(len(string) - 2):
                trigram = string[i:i + 3]
                self.vocabulary.add(trigram)
        self.vocabulary = list(self.vocabulary)
        self._save_to_binary(self.vocabulary, self.vocabulary_filepath)
        self.vocabulary = np.array(self.vocabulary)

    def _add_trigrams_to_dataset(self):
        df = self.df

        if 'subject_content' in df.columns:
            df = df.rename(columns={'subject_content': 'subject'})
            print('renamed column subject_content to subject')

        if 'content' in df.columns:
            sel_1 = (df['content'] == '')
            sel_2 = df['subject'] == df['content']
            sel_3 = ~(df['bestanswer'] == '')
            sel_1 = (sel_1 | sel_2) & sel_3
        else:
            sel_1 = ~(df['bestanswer'] == '')

        sel_2 = (df["bestanswer"].str.len() >= 3) & (df["subject"].str.len() >= 3)
        sel = sel_1 & sel_2
        df_sel = df[sel]

        all_strings = list(df_sel['subject'].values) + list(df_sel['bestanswer'].values)

        print('generating vocabulary...')
        self._generate_vocabulary(all_strings)
        print('generated')

        df_sel["tri_subject"] = 'not_added_yet'
        df_sel["tri_bestanswer"] = 'not_added_yet'

        sort_idx = self.vocabulary.argsort()

        subject_list = list(df_sel['subject'].values)
        tri_subject_list = subject_list.copy()
        bestanswer_list = list(df_sel['bestanswer'].values)
        tri_bestanswer_list = bestanswer_list.copy()

        print('generating trigrams for subject...')
        for i in range(len(subject_list)):
            if i % 10000 == 0:
                print(i, len(subject_list))
            subject = str(subject_list[i])
            trigrams = self._extract_trigrams(subject, self.vocabulary, sort_idx)
            tri_subject_list[i] = trigrams
        print(len(subject_list), len(subject_list))

        print('generating trigrams for bestanswer...')
        for i in range(len(bestanswer_list)):
            if i % 10000 == 0:
                print(i, len(bestanswer_list))
            bestanswer = str(bestanswer_list[i])
            trigrams = self._extract_trigrams(bestanswer, self.vocabulary, sort_idx)
            tri_bestanswer_list[i] = trigrams
        print(len(bestanswer_list), len(bestanswer_list))

        df_sel['tri_subject'] = pd.Series(tri_subject_list)
        df_sel['tri_bestanswer'] = pd.Series(tri_bestanswer_list)

        self._save_pandas(df_sel, self.dataset_filepath)
        self.df = df_sel

    def _extract_trigrams(self, string, trigrams_vocabulary, sort_idx):
        trigrams = np.unique([string[j:j + 3] for j in range(len(string) - 2)])
        if len(trigrams) == 0:
            print("no trigrams extracted!", string)
            raise NotImplementedError

        return np.unique(sort_idx[np.searchsorted(trigrams_vocabulary, trigrams, sorter=sort_idx)])

    def _sample_negative_examples():
        print('hoi')
        [np.random.choice(np.delete(np.arange(N), i)) for i in range(N)]


    @staticmethod
    def _convert_pandas_to_list(data):
        return list(data.T.to_dict().values())

    @staticmethod
    def _save_to_binary(data, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load_from_binary(filepath):
        with open(filepath, 'rb') as f:
            p = pickle.load(f)
            return p

    @staticmethod
    def _save_pandas(data, filepath):
        data.to_pickle(filepath)

#
# dataset_filenames = [
#                      'yahoo_df_subject+content.p',
#                      'yahoo_df_subject.p',
#                      'yahoo_df_subject_no_content.p'
#                     ]
#
# dataset_folder = './../../../data/Yahoo'
# dataset_filename = dataset_filenames[1]
#
# shuffle_dataset = False
# batch_size = 20
#
# yahoo = YahooDataset(dataset_folder, dataset_filename)
# yahoo.init_dataset(shuffle_dataset)
# print('vocabulary size:', yahoo.get_vocabulary_size())
# print(yahoo.get_next_batch(batch_size))
