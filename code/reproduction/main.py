# from .lib.models.SDQA import SDQA
# from .lib.utils.losses import cosine_sim_loss
import tensorflow as tf
import numpy as np

from lib import data


# define the parameters
batch_size = 128
learning_rate = 0.001
vocabulary_size = 47000
activation_fn = tf.nn.relu

data_file = '/home/henglin/uva17/ir2/ir2/code/data/Quora/trigrams_quora_duplicate_questions.tfrecords'

##########################
##########################

dataset = data.QuoraLstm(files=data_file, batch_size=batch_size,
                         num_epochs=1, max_length=100000, train=True)

iterator = dataset()

PairID, qid1, qid2, q1_words, q2_words, is_duplicate, sv = iterator.get_next()

##########################
##########################

# # initialize the network
# nn = SDQA(activation_fn=activation_fn, is_training=True)
#
# # calculate loss
# # TODO: get the labels
# labels = tf.placeholder(dtype=tf.float32, shape=[batch_size])
#
#
# global_step = tf.Variable(0, name='global_step', trainable=False)
# saver = tf.train.Saver(max_to_keep=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # sess.run(tf.tables_initializer())
    sess.run(iterator.initializer)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(1):
        print("--- iteration " + str(i) + " ---")

        result = sess.run([q1_words, q2_words, sv])
        print(result[0][:2])
        print(result[1][:2])
        print(result[2])

    coord.request_stop()
    coord.join(threads)
