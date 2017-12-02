import collections
import json
import scipy.misc
import sys
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import time

'''
this is a script to convert the dataset in pandas to tfrecord
'''

class Example:
    def __init__(self, subject, bestanswer, label):
        self.label = label
        self.subject = subject
        self.bestanswer = bestanswer

    @staticmethod
    def int64(value):
        list = tf.train.Int64List(value = [ value ])
        return tf.train.Feature(int64_list = list)

    @staticmethod
    def bytes(value):
        list = tf.train.BytesList(value = [ value ])
        return tf.train.Feature(bytes_list = list)

    def feature(self):
        assert(self.label is not None)

        return {
            "subject":    Example.bytes(tf.compat.as_bytes(self.subject)),
            "bestanswer": Example.bytes(tf.compat.as_bytes(self.bestanswer)),
            "label":    Example.int64(int(self.label))
        }


output_file_name = "data_lstm.tfrecord"
input_file_name = "data_LSTM.p"

##### READING DATA FROM DF

df = pd.read_pickle(input_file_name)

df['label'] = 1.0
df = df.loc[:, ['subject_preprocessed', 'bestanswer_preprocessed', 'label']]

df_sp = df.loc[:, ['subject_preprocessed']]
df_bp = df.loc[:, ['bestanswer_preprocessed']]
df_bp = df_bp.sample(frac=1.0)      # shuffle the answer only

df_sp.reset_index(drop=True, inplace=True)
df_bp.reset_index(drop=True, inplace=True)

df_neg = pd.concat([df_sp, df_bp], axis=1)  # combine the originally ordered subject with shuffled answer
df_neg['label'] = 0.0                       # add label as not relevant

df_all = pd.concat([df, df_neg], axis=0)

df_all = df_all.sample(frac=1.0)            # shuffle the dataset
df_all = df_all.dropna()

subject = df_all.subject_preprocessed.values
bestanswer = df_all.bestanswer_preprocessed.values
labels = df_all.label.values

num_sample = subject.shape[0]
print("Total number of samples to be written in TFRecord: {}".format(num_sample))


# Open a new writer.
writer = tf.python_io.TFRecordWriter(output_file_name)

start = time.time()
for j in range(num_sample):

    example = Example(subject[j], bestanswer[j], labels[j])

    feature = example.feature()
    packed = tf.train.Example(features=tf.train.Features(feature=feature))

    writer.write(packed.SerializeToString())
    if j % 10000 == 0:
        print('#sample processed: {}, time elapsed: {}'.format(j, time.time() - start))
        start = time.time()

writer.close()
