from .lib.models.SDQA import SDQA
from .lib.utils.losses import cosine_sim_loss

import tensorflow as tf


# define the parameters
batch_size = 128
learning_rate = 0.001
activation_fn = tf.nn.relu
restore = False

# TODO: To be defined
checkpoint_path = './ckpt/'
checkpoint_prefix = 'ckpt_sdqa'


##########################
##########################
'''
NEED TO LOAD THE DATA
'''
##########################
##########################

# initialize the network
nn = SDQA(activation_fn=activation_fn, is_training=True)

# features for 2 diff inputs
# TODO: Better way to define the shape
features1 = tf.placeholder(dtype=tf.float32, shape=[1, 48536])
features2 = tf.placeholder(dtype=tf.float32, shape=[1, 48536])

# logits for 2 diff inputs
logit1 = nn.define_network(features1)
logit2 = nn.define_network(features2)

# calculate loss
# TODO: get the labels
labels = tf.placeholder(dtype=tf.float32, shape=[batch_size])

loss = cosine_sim_loss(logit1, logit2, labels, m=1)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

global_step = tf.Variable(0, name='global_step', trainable=False)
saver = tf.train.Saver(max_to_keep=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # sess.run(tf.tables_initializer())
    # sess.run(iterator.initializer)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if restore:
        # TODO: Not sure why tf.train.latest_checkpoint returns None???
        saver.restore(sess, checkpoint_path)
        last_global_step = global_step.eval()
    else:
        last_global_step = 0

    print("Start training from:", last_global_step)

    for i in range(100000):
        print("--- iteration " + str(i + 1 + last_global_step) + " ---")

        # result = sess.run([], feed_dict={})

        if i % 500 == 0:
            save_path = saver.save(sess, checkpoint_prefix, global_step=global_step)

    coord.request_stop()
    coord.join(threads)
