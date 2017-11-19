import tensorflow as tf
import numpy as np
import time

from code.reproduction.lib.data.quora_utils import QuoraDataset

# define the parameters
from code.reproduction.lib.models.SDQA import SDQA
from code.reproduction.lib.utils import cosine_sim_loss

restore = False

# TODO: To be defined
checkpoint_path = './ckpt/'
checkpoint_prefix = 'ckpt_sdqa'


dataset_folder = './../data/Quora'
shuffle_dataset = True
batch_size = 20

quora = QuoraDataset(dataset_folder)
quora.init_dataset(shuffle_dataset, 'trigrams_sanitized')
vocabulary_size = quora.get_vocabulary_size()

# initialize the network
nn = SDQA(is_training=True, input_shape=vocabulary_size)

input1 = nn.input1
input2 = nn.input2
labels = tf.placeholder(dtype=tf.float32, shape=[batch_size, 1])

logits1 = nn.logits1
logits2 = nn.logits2

inference = nn.inference()
loss = nn.loss(labels)
# train_step = nn.train_step(loss)
# accuracy = nn.accuracy() #TODO: check what to use here


def sparse2dense(batch):
    labels = np.array([sample['is_duplicate'] for sample in batch])
    qid1 = np.array([sample['qid1'] for sample in batch])
    qid2 = np.array([sample['qid2'] for sample in batch])
    ids = np.array([sample['id'] for sample in batch])

    # turn them to sparse representation
    q1_vec = np.zeros(shape=[batch_size, vocabulary_size])
    q2_vec = np.zeros(shape=[batch_size, vocabulary_size])

    for i, sample in enumerate(batch):
        q1_vec[i, sample['question1']] = 1
    for i, sample in enumerate(batch):
        q2_vec[i, sample['question2']] = 1

    q1_vec = np.expand_dims(q1_vec, axis=2)
    q2_vec = np.expand_dims(q2_vec, axis=2)
    labels = np.expand_dims(labels, axis=1)

    return ids, qid1, qid2, q1_vec, q2_vec, labels


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for i in range(1):
        print("--- iteration " + str(i) + " ---")
        batch = quora.get_next_batch(batch_size)

        ids, qid1, qid2, q1_vec, q2_vec, y = sparse2dense(batch)

        result = sess.run([logits1, logits2, inference, loss],
                          feed_dict={input1: q1_vec, input2: q2_vec, labels: y})
        print(result[0].shape)
