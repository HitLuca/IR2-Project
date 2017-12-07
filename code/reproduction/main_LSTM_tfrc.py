import tensorflow as tf
import os
import sys
import numpy as np
import time

path = os.path.dirname(os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))) + '/data/'
sys.path.append(path)  # to correctly import Dataset

from lib.models.LSTM import LSTM
from yahoo_dataset_tfrc import LSTMDataset_TFRC

dataset_folder = './../data/Yahoo'
dataset_filename = 'data_lstm_test.tfrecord'
vocabulary_filepath = './lib/data/testset_vocabulary.txt'
embeddings_filepath = './lib/data/testset_partial_embedding_matrix.npy'

batch_size = 64
learning_rate = 0.001
max_steps = 10000
lstm_num_layers = 1
lstm_num_hidden = 128
train_embedding = False

# TODO: Add a tag to allow fine tuning the embedding

dataset_path = os.path.join(dataset_folder, dataset_filename)

# create a dataset iterator
dataset = LSTMDataset_TFRC(dataset_path, batch_size, 10, 100, True, vocabulary_filepath)
dataset_test = LSTMDataset_TFRC(dataset_path, batch_size, 10, 100, False, vocabulary_filepath)

iterator = dataset()
iterator_test = dataset_test()

# load the next batch
question1, question2, labels = iterator.get_next()
q1_test, q2_test, l_test = iterator_test.get_next()

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

# perform inference on the training set
with tf.variable_scope("inference"):
    cosine_similarity = nn.inference(question1, question2)
    loss = nn.loss(labels, cosine_similarity)
    train_step = nn.train_step(loss, learning_rate)
    accuracy = nn.accuracy(labels, cosine_similarity)

# reuse the network for testing on the whole dataset
with tf.variable_scope("inference", reuse=True):
    cosine_similarity_test = nn.inference(q1_test, q2_test)
    loss_test = nn.loss(l_test, cosine_similarity_test)

# log the training behavior
tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)
summary_op = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
sess.run(tf.tables_initializer())

# initializer data iterators
sess.run(iterator.initializer)
sess.run(iterator_test.initializer)

# handling multi-threading
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

train_writer = tf.summary.FileWriter('./log_dir/lstm_train', sess.graph)

for i in range(100000):
    result = sess.run([loss, accuracy, train_step, summary_op],
                      feed_dict={
                          is_training: True,
                          embedding_matrix: np_embedding_matrix
                      })

    if i % 10 == 0:
        print("step: %3d, loss: %.6f, acc: %.3f" % (i, result[0], result[1]))
        train_writer.add_summary(result[-1], i)

    # Run through the test set
    if i % 100 == 0:
        print("Iterating through the test set...")
        total_loss = list()
        while True:
            try:
                test_loss = sess.run(loss_test)
                total_loss.append(test_loss)
            except tf.errors.OutOfRangeError:
                print("Testing loss at step: {}, loss: {}".format(i, np.array(total_loss).mean()))
                break
        # reset the test set iterator
        sess.run(iterator_test.initializer)


coord.request_stop()
coord.join(threads)

sess.close()
