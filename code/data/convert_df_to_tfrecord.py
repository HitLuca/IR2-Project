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
this script also create negative samples so that +ve to -ve sample are mixed at ratio of 50/50
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


mode = "val"

q1_key = "subject"
q2_key = "bestanswer"

df_paths = {"train":     "./Yahoo/train_df_reduced_no_empty.p",
            "train_sim": "./Yahoo/train_df_sim_len.p",
            "val":       "./Yahoo/val_df_reduced_no_empty.p",
            "val_sim":   "./Yahoo/val_df_sim_len.p",
            "pruned":    "./Yahoo/train_pruned_alot.p",
            "test":      "./Yahoo/test_set_1018_preprocessed.p"}

output_paths = {"train":     "./Yahoo/train_lstm.tfrecord",
                "train_sim": "./Yahoo/train_lstm_sim_len.tfrecord",
                "val":       "./Yahoo/val_lstm.tfrecord",
                "val_sim":   "./Yahoo/val_lstm_sim_len.tfrecord",
                "pruned":    "./Yahoo/train_pruned.tfrecord",
                "test":      "./Yahoo/test_lstm_1018.tfrecord"}

output_file_name = output_paths[mode]

# READING DATA FROM DF
df = pd.read_pickle(df_paths[mode])

if mode == "test":
    df_all = df.loc[:, [q1_key, q2_key, 'label']]
else:
    # set the label of +ve samples to 1.0
    df['label'] = 1.0

    # select interested columns
    df = df.loc[:, [q1_key, q2_key, 'label']]

    # select the subject and answer, then ONLY shuffle the answer
    df_sp = df.loc[:, [q1_key]]
    df_bp = df.loc[:, [q2_key]]
    df_bp = df_bp.sample(frac=1.0)      # shuffle the answer only

    df_sp.reset_index(drop=True, inplace=True)
    df_bp.reset_index(drop=True, inplace=True)

    df_neg = pd.concat([df_sp, df_bp], axis=1)  # combine the originally ordered subject with shuffled answer
    df_neg['label'] = 0.0                       # add label as not relevant for -ve sample

    df_all = pd.concat([df, df_neg], axis=0)    # concatenate the original data with the -ve sample

    df_all = df_all.sample(frac=1.0)            # shuffle the dataset
    df_all = df_all.dropna()                    # remove nan, just to be sure

subject = df_all[q1_key].as_matrix()
bestanswer = df_all[q2_key].as_matrix()
labels = df_all['label'].as_matrix()

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
