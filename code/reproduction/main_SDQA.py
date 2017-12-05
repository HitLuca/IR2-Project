# region imports
import os
import sys

path = os.path.dirname(os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))) + '/data/'
sys.path.append(path)  # to correctly import Dataset

import tensorflow as tf
import numpy as np
from yahoo_datasets import TrigramsDataset
from lib.models.SDQA import SDQA

# endregion

checkpoint_path = './ckpt/'
checkpoint_prefix = 'ckpt_sdqa'

# dataset_folder = './../data/Yahoo'
dataset_folder = './../data/batch_creation'
dataset_filename = 'minimal_sparse'
shuffle_dataset = True
batch_size = 64
loss_margin = 0.3
learning_rate = 0.001
max_steps = 10000

data = TrigramsDataset(batch_size,
                       dataset_filename,
                       dataset_folder)

# initialize the network
is_training = tf.placeholder(tf.bool)
input1 = tf.placeholder(dtype=tf.float32, shape=[None, data.vocabulary_size])
input2 = tf.placeholder(dtype=tf.float32, shape=[None, data.vocabulary_size])
labels = tf.placeholder(dtype=tf.float32, shape=[None])

nn = SDQA(is_training=is_training,
          vocabulary_size=data.vocabulary_size)

cosine_similarity = nn.inference(input1, input2)
loss = nn.loss(labels, cosine_similarity, margin=loss_margin)
train_step = nn.train_step(loss, learning_rate)
accuracy = nn.accuracy(labels, cosine_similarity)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

for i in range(max_steps):
    question1, question2, y = data.next_batch(is_one_hot_encoding=True)
    result = sess.run([cosine_similarity, loss, accuracy, train_step],
                      feed_dict={input1: question1,
                                 input2: question2,
                                 labels: y,
                                 is_training: True})

    print("step: %3d, loss: %.6f, acc: %.3f, min_cosine_sim: %.4f" % (i, result[1], result[2], np.min(result[0])))
