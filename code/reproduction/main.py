import tensorflow as tf
import numpy as np
import time

# from lib.data.quora_utils import QuoraDataset
from lib.data.yahoo_utils import YahooDataset

# define the parameters
from lib.models.SDQA import SDQA


# TODO: To be defined
checkpoint_path = './ckpt/'
checkpoint_prefix = 'ckpt_sdqa'


dataset_folder = './../data/Yahoo'
dataset_filename = 'yahoo_df_subject+content.p'
shuffle_dataset = True
batch_size = 32
acc_threshold = 0.7     # TODO: This has to be verified
loss_margin = 0.5

# loading data
# quora = QuoraDataset(dataset_folder)
# quora.init_dataset(shuffle_dataset, 'trigrams_sanitized')
# vocabulary_size = quora.get_vocabulary_size()

yahoo = YahooDataset(dataset_folder, dataset_filename)
yahoo.init_dataset(shuffle_dataset)
vocabulary_size = yahoo.get_vocabulary_size()

is_training = tf.placeholder(tf.bool, shape=())

# initialize the network
nn = SDQA(is_training=is_training, input_shape=vocabulary_size)

input1 = nn.input1
input2 = nn.input2
labels = tf.placeholder(dtype=tf.float32, shape=[None, 1])

logits1 = nn.logits1
logits2 = nn.logits2

inference = nn.inference()
loss, cosine_dist = nn.loss(label=labels, margin=loss_margin)
train_step = nn.train_step(loss)
accuracy = nn.accuracy(labels, cosine_dist, acc_threshold)  # TODO: check what to use here


# TODO: Move this function to utils
def sparse2dense(batch):
    labels = np.array([sample['relevance'] for sample in batch])

    # TODO: IDs not required for yahoo dataset
    # qid1 = np.array([sample['qid1'] for sample in batch])
    # qid2 = np.array([sample['qid2'] for sample in batch])
    # ids = np.array([sample['id'] for sample in batch])

    # turn them to sparse representation
    q1_vec = np.zeros(shape=[batch_size, vocabulary_size])
    q2_vec = np.zeros(shape=[batch_size, vocabulary_size])

    # TODO: Make keys more generic
    for i, sample in enumerate(batch):
        q1_vec[i, sample['tri_subject']] = 1
    for i, sample in enumerate(batch):
        q2_vec[i, sample['tri_bestanswer']] = 1

    q1_vec = np.expand_dims(q1_vec, axis=2)
    q2_vec = np.expand_dims(q2_vec, axis=2)
    labels = np.expand_dims(labels, axis=1)

    # return ids, qid1, qid2, q1_vec, q2_vec, labels
    return q1_vec, q2_vec, labels


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for i in range(1000):
        batch = yahoo.get_next_batch(batch_size)

        # ids, qid1, qid2, q1_vec, q2_vec, y = sparse2dense(batch)
        q1_vec, q2_vec, y = sparse2dense(batch)

        result = sess.run([logits1, logits2, inference, loss, accuracy, train_step, cosine_dist],
                          feed_dict={input1: q1_vec,
                                     input2: q2_vec,
                                     labels: y,
                                     is_training: True})

        print("step: %3d, loss: %2.3f, acc: %2.3f" % (i, result[3], result[4]))
        print(result[6])
