import tensorflow as tf

from code.reproduction.lib.data.quora_utils import QuoraDataset

# define the parameters
from code.reproduction.lib.models.SDQA import SDQA
from code.reproduction.lib.utils import cosine_sim_loss

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

dataset_folder = './../data/Quora'
shuffle_dataset = True
batch_size = 1

quora = QuoraDataset(dataset_folder)
quora.init_dataset(shuffle_dataset, 'trigrams_sanitized')
vocabulary_size = quora.get_vocabulary_size()

# initialize the network
nn = SDQA(activation_fn=activation_fn, is_training=True, input_shape=vocabulary_size)

input1 = nn.input1
input2 = nn.input2
labels = tf.placeholder(dtype=tf.float32, shape=[batch_size])

logits1 = nn.logits1
logits2 = nn.logits2

inference = nn.inference()
loss = nn.loss(labels)
train_step = nn.train_step(loss)
# accuracy = nn.accuracy() #TODO: check what to use here

global_step = tf.Variable(0, name='global_step', trainable=False)
saver = tf.train.Saver(max_to_keep=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

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
        batch = quora.get_next_batch(batch_size)
        # result = sess.run([], feed_dict={})

        if i % 500 == 0:
            save_path = saver.save(sess, checkpoint_prefix, global_step=global_step)

    coord.request_stop()
    coord.join(threads)
