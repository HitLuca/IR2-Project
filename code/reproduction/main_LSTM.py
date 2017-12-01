import tensorflow as tf
import os
import sys
import numpy as np

path = os.path.dirname(os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))) + '/data/'
sys.path.append(path)  # to correctly import Dataset

from lib.models.LSTM import LSTM
from yahoo_datasets import LSTMDataset

dataset_folder = './../data/Yahoo'
dataset_filename = 'mini_data_LSTM.p'
vocabulary_filepath = './lib/data/vocabulary.txt'
embeddings_filepath = './lib/data/partial_embedding_matrix.npy'

batch_size = 64
learning_rate = 1e-6
max_steps = 10000
lstm_num_layers = 2
lstm_num_hidden = 128
accuracy_threshold = 0.5

data = LSTMDataset(batch_size,
                   dataset_filename,
                   data_dir=dataset_folder)

# initialize the network
is_training = tf.placeholder(tf.bool)
input1 = tf.placeholder(name='input1', dtype=tf.string, shape=[None])
input2 = tf.placeholder(name='input2', dtype=tf.string, shape=[None])
labels = tf.placeholder(name='labels', dtype=tf.float32, shape=[None])
batch_length = tf.placeholder(name='batch_length', dtype=tf.int32)
embedding_matrix = tf.placeholder(name='embedding_matrix', dtype=tf.float32, shape=[None, None])

nn = LSTM(is_training,
          vocabulary_filepath,
          batch_size=batch_size,
          lstm_num_layers=lstm_num_layers,
          lstm_num_hidden=lstm_num_hidden)

nn.assign_embedding_matrix(embedding_matrix)

cosine_similarity = nn.inference(input1, input2)
loss = nn.loss(labels, cosine_similarity)
train_step = nn.train_step(loss, learning_rate)
accuracy = nn.accuracy(labels, cosine_similarity, accuracy_threshold)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
sess.run(tf.tables_initializer())

np_embedding_matrix = np.load(embeddings_filepath)

for i in range(max_steps):
    question1, question2, y = data.next_batch()
    result = sess.run([loss, accuracy, cosine_similarity, train_step],
                      feed_dict={input1: question1,
                                 input2: question2,
                                 labels: y,
                                 is_training: True,
                                 embedding_matrix: np_embedding_matrix})

    print("step: %3d, loss: %.6f, acc: %.3f" % (i, result[0], result[1]))
