import tensorflow as tf
import os
import sys
import numpy as np
import time

path = os.path.dirname(os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))) + '/data/'
sys.path.append(path)  # to correctly import Dataset

from lib.models.LSTM import LSTM
# from yahoo_datasets import LSTMDataset
from yahoo_dataset_tfrc import LSTMDataset_TFRC

dataset_folder = './../data/Yahoo'
dataset_filename = 'data_lstm.tfrecord'
vocabulary_filepath = './lib/data/vocabulary.txt'
yahoo_vocabulary_filepath = './lib/data/yahoo_vocabulary.txt'       # only use the most frequent words
embeddings_filepath = './lib/data/partial_embedding_matrix.npy'

batch_size = 64
learning_rate = 0.0001
max_steps = 10000
lstm_num_layers = 1
lstm_num_hidden = 128
train_embedding = True

dataset_path = os.path.join(dataset_folder, dataset_filename)

# create a dataset iterator
dataset = LSTMDataset_TFRC(dataset_path, batch_size, 10, 50, True, vocabulary_filepath)
iterator = dataset()

# load the next batch
question1, question2, labels = iterator.get_next()

# initialize the network
is_training = tf.placeholder(tf.bool)
batch_length = tf.placeholder(name='batch_length', dtype=tf.int32)
embedding_matrix = tf.placeholder(name='embedding_matrix', dtype=tf.float32, shape=[None, None])

nn = LSTM(is_training,
          vocabulary_filepath,
          batch_size=batch_size,
          lstm_num_layers=lstm_num_layers,
          lstm_num_hidden=lstm_num_hidden,
          use_tfrecord=True)

if train_embedding is False:
    nn.assign_embedding_matrix(embedding_matrix)
    np_embedding_matrix = np.load(embeddings_filepath)
else:
    np_embedding_matrix = [[None]]

cosine_similarity = nn.inference(question1, question2)
loss = nn.loss(labels, cosine_similarity)
train_step = nn.train_step(loss, learning_rate)
accuracy = nn.accuracy(labels, cosine_similarity)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
sess.run(tf.tables_initializer())
sess.run(iterator.initializer)

start = time.time()
for i in range(1000):
    result = sess.run([loss, accuracy, train_step],
                      feed_dict={
                          is_training: True,
                          embedding_matrix: np_embedding_matrix
                      })
    print("step: %3d, loss: %.6f, acc: %.3f" % (i, result[0], result[1]))
    # result = sess.run([question1, question2, labels])
    if i % 100 == 0:
        print("Time elapsed: {}".format(time.time() - start))
        start = time.time()

sess.close()
