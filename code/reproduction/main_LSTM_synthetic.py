#region imports
import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))) + '/data/'
sys.path.append(path)# to correctly import Dataset

import tensorflow as tf
import numpy as np
from lib.models.LSTM import LSTM
from lib.utils.synthetic_datasets import SyntheticDatasetLSTM
#endregion

batch_size = 64
loss_margin = 0.0
learning_rate = 0.001
max_steps = 10000
lstm_num_layers = 2
lstm_num_hidden = 128
seq_max_length = 1000

# initialize the network
is_training = tf.placeholder(tf.bool)
input1 = tf.placeholder(dtype=tf.float32, shape=[None, None])
input2 = tf.placeholder(dtype=tf.float32, shape=[None, None])
labels = tf.placeholder(dtype=tf.float32, shape=[None])

nn = LSTM(batch_size=batch_size,
          lstm_num_layers=lstm_num_layers,
          lstm_num_hidden=lstm_num_hidden)

cosine_similarity = nn.inference(input1, input2)
loss = nn.loss(labels, cosine_similarity, margin=loss_margin)
train_step = nn.train_step(loss, learning_rate)
accuracy = nn.accuracy(labels, cosine_similarity)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

data = SyntheticDatasetLSTM()

for i in range(max_steps):
    timeseries1, timeseries2, y = data.generate_batch(batch_size, seq_max_length)
    result = sess.run([cosine_similarity, loss, accuracy, train_step],
                      feed_dict={input1: timeseries1,
                                 input2: timeseries2,
                                 labels: y})

    print("step: %3d, loss: %.6f, acc: %.3f, min_cosine_sim: %.4f" % (i, result[1], result[2], np.min(result[0])))
