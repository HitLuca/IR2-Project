import os
import pickle
import numpy as np
import random
import string
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

DATASET_VERSIONS = {0: 'default',
                    1: 'trigrams',
                    2: 'default_sanitized',
                    3: 'trigrams_sanitized'}


class QuoraDataset:
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder + '/'

        self.dataset_filename = 'quora_duplicate_questions.tsv'
        self.dataset_filepath = self.dataset_folder + self.dataset_filename

        self.sanitized_dataset_filename = 'quora_sanitized.pk1'
        self.sanitized_dataset_filepath = self.dataset_folder + self.sanitized_dataset_filename

        self.trigrams_vocabulary_filename = 'quora_trigrams_vocabulary.pk1'
        self.trigrams_vocabulary_filepath = self.dataset_folder + self.trigrams_vocabulary_filename

        self.sanitized_trigrams_vocabulary_filename = 'quora_trigrams_sanitized_vocabulary.pk1'
        self.sanitized_trigrams_vocabulary_filepath = self.dataset_folder + self.sanitized_trigrams_vocabulary_filename

        self.trigrams_dataset_filename = 'quora_trigrams_dataset.pk1'
        self.trigrams_dataset_filepath = self.dataset_folder + self.trigrams_dataset_filename

        self.sanitized_trigrams_dataset_filename = 'quora_trigrams_sanitized_dataset.pk1'
        self.sanitized_trigrams_dataset_filepath = self.dataset_folder + self.sanitized_trigrams_dataset_filename

        self.dataset = None
        self.vocabulary = None
        self.batch_index = 0
        self.dataset_version = None

    def fix_dataset(self):
        fixed_dataset_filepath = self.dataset_filepath + '.fixed'

        clean_line = True
        incomplete_lines = 0

        with open(self.dataset_filepath, 'r') as f1:
            with open(fixed_dataset_filepath, 'w') as f2:
                for line in f1.readlines():
                    line = line.replace('\n', '')
                    if clean_line:
                        splitted = line.split('\t')
                        if len(splitted) == 6:
                            print(line, file=f2)
                        else:
                            clean_line = False
                            incomplete_lines += 1
                    else:
                        clean_line = True
                        pass
        os.rename(fixed_dataset_filepath, self.dataset_filepath)

    def load_trigrams_vocabulary(self, dataset, filepath):
        if not os.path.exists(filepath):
            trigrams_vocabulary = set()
            for entry in dataset:
                question1 = entry['question1']
                question2 = entry['question2']
                for i in range(len(question1) - 2):
                    trigram = question1[i:i + 3]
                    trigrams_vocabulary.add(trigram)
                for i in range(len(question2) - 2):
                    trigram = question2[i:i + 3]
                    trigrams_vocabulary.add(trigram)
            trigrams_vocabulary = list(trigrams_vocabulary)
            self.save_to_binary(trigrams_vocabulary, filepath)
            return np.array(trigrams_vocabulary)
        else:
            return np.array(self.load_from_binary(filepath))

    def load_quora_dataset(self):
        try:
            original_dataset = []
            with open(self.dataset_filepath, 'r') as f:
                for line in f.readlines():
                    line = line.split('\t')
                    if line[0] == 'id':
                        pass
                    else:
                        id = int(line[0])
                        qid1 = int(line[1])
                        qid2 = int(line[2])
                        question1 = line[3].replace('\n', '')
                        question2 = line[4].replace('\n', '')
                        is_duplicate = int(line[5])
                        entry = {'id': id,
                                 'qid1': qid1,
                                 'qid2': qid2,
                                 'question1': question1,
                                 'question2': question2,
                                 'is_duplicate': is_duplicate
                                 }
                        original_dataset.append(entry)
            return original_dataset
        except IndexError:
            self.fix_dataset()
            return self.load_quora_dataset()

    @staticmethod
    def extract_trigrams(question, trigrams_vocabulary, sort_idx):
        trigrams = np.unique([question[j:j + 3] for j in range(len(question) - 2)])
        if len(trigrams) == 0:
            return None

        return np.unique(sort_idx[np.searchsorted(trigrams_vocabulary, trigrams, sorter=sort_idx)])

    @staticmethod
    def sanitize_question(question):
        sanitized_question = question.lower()
        sanitized_question = sanitized_question.replace("'s", ' is').replace("can't", 'cannot').replace("'ve", ' have')
        stop = stopwords.words('english') + list(string.punctuation)
        sanitized_question = " ".join([i for i in word_tokenize(sanitized_question) if i not in stop])
        return sanitized_question

    def load_quora_dataset_sanitized(self):
        if not os.path.exists(self.sanitized_dataset_filepath):
            sanitized_dataset = []
            original_dataset = self.load_quora_dataset()
            for i in range(len(original_dataset)):
                entry = original_dataset[i]
                new_entry = entry
                new_entry['question1'] = self.sanitize_question(entry['question1'])
                new_entry['question2'] = self.sanitize_question(entry['question2'])
                sanitized_dataset.append(new_entry)
            self.save_to_binary(sanitized_dataset, self.sanitized_dataset_filepath)
            return sanitized_dataset
        else:
            return self.load_from_binary(self.sanitized_dataset_filepath)

    def load_dataset_trigrams_sanitized(self):
        if not os.path.exists(self.sanitized_trigrams_dataset_filepath):
            sanitized_dataset = []
            original_dataset = self.load_quora_dataset_sanitized()
            self.vocabulary = self.load_trigrams_vocabulary(original_dataset,
                                                            self.sanitized_trigrams_vocabulary_filepath)
            sort_idx = self.vocabulary.argsort()
            for i in range(len(original_dataset)):
                entry = original_dataset[i]
                new_entry = entry
                new_entry['question1'] = self.extract_trigrams(entry['question1'], self.vocabulary, sort_idx)
                new_entry['question2'] = self.extract_trigrams(entry['question2'], self.vocabulary, sort_idx)
                if new_entry['question1'] is None or new_entry['question2'] is None:
                    continue
                sanitized_dataset.append(new_entry)
            self.save_to_binary(sanitized_dataset, self.sanitized_trigrams_dataset_filepath)
            return sanitized_dataset
        else:
            return self.load_from_binary(self.sanitized_trigrams_dataset_filepath)

    def load_dataset_trigrams(self):
        if not os.path.exists(self.trigrams_dataset_filepath):
            original_dataset = self.load_quora_dataset()
            self.vocabulary = self.load_trigrams_vocabulary(original_dataset, self.trigrams_vocabulary_filepath)

            trigrams_dataset = []
            sort_idx = self.vocabulary.argsort()
            for i in range(len(original_dataset)):
                entry = original_dataset[i]
                new_entry = entry
                question1 = entry['question1']
                question2 = entry['question2']

                new_entry['question1'] = self.extract_trigrams(question1, self.vocabulary, sort_idx)
                new_entry['question2'] = self.extract_trigrams(question2, self.vocabulary, sort_idx)
                if new_entry['question1'] is None or new_entry['question2'] is None:
                    continue
                trigrams_dataset.append(new_entry)
            self.save_to_binary(trigrams_dataset, self.trigrams_dataset_filepath)
            return trigrams_dataset
        else:
            return self.load_from_binary(self.trigrams_dataset_filepath)

    def init_dataset(self, shuffle, dataset_version):
        self.dataset_version = dataset_version
        if dataset_version == 'default':
            self.dataset = self.load_quora_dataset()
        if dataset_version == 'trigrams':
            self.dataset = self.load_dataset_trigrams()
            self.vocabulary = self.load_from_binary(self.trigrams_vocabulary_filepath)
        if dataset_version == 'default_sanitized':
            self.dataset = self.load_quora_dataset_sanitized()
        if dataset_version == 'trigrams_sanitized':
            self.dataset = self.load_dataset_trigrams_sanitized()
            self.vocabulary = self.load_from_binary(self.sanitized_trigrams_vocabulary_filepath)

        if shuffle:
            random.shuffle(self.dataset)

    def get_vocabulary_size(self):
        if self.vocabulary is not None:
            return len(self.vocabulary)
        else:
            return NotImplementedError

    def get_next_batch(self, batch_size):
        self.batch_index += batch_size
        return self.dataset[self.batch_index - batch_size:self.batch_index]

    @staticmethod
    def load_from_binary(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def save_to_binary(data, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


# dataset_folder = './../data/Quora'
# shuffle_dataset = True
# batch_size = 1
#
# quora = QuoraDataset(dataset_folder)
# quora.init_dataset(shuffle_dataset, 'trigrams_sanitized')
# print(quora.get_vocabulary_size())
# print(quora.get_next_batch(batch_size))
