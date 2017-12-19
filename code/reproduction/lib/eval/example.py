# region imports
import os
import sys

path = os.path.dirname(os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))) + '/data/'
sys.path.append(path)  # to correctly import Dataset

path = os.path.dirname(os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))) + '/ir2_data/'
sys.path.append(path)  # to correctly import Dataset

import tensorflow as tf
import numpy as np
# from yahoo_datasets TrigramsDataset import
from Dataset import Dataset
from lib.models.SDQA_simply_fc import SDQA
import utils_joop as uj
import evaluation_bench

batch_size = 32
loss_margin = 0.3
learning_rate = 0.001
max_steps = 3e6

f_train = '../ir2_data/train_val/trigram_dense_train.p'
f_test = '../ir2_data/train_val/trigram_dense_test.p'
f_val = '../ir2_data/train_val/trigram_dense_val.p'
data = Dataset(batch_size, f_train, f_val, f_test)

# initialize the network
is_training = tf.placeholder(tf.bool)
input1 = tf.placeholder(dtype=tf.float32, shape=[None, data.vocabulary_size])
input2 = tf.placeholder(dtype=tf.float32, shape=[None, data.vocabulary_size])
labels = tf.placeholder(dtype=tf.float32, shape=[None])

nn = SDQA(is_training=is_training,
		  vocabulary_size=data.vocabulary_size)

cosine_similarity = nn.inference(input1, input2)
loss = nn.loss(labels, cosine_similarity, margin=loss_margin)
tf.summary.scalar("loss", loss)
train_step = nn.train_step(loss, learning_rate)
summaries = tf.summary.merge_all()
# accuracy = nn.accuracy(labels, cosine_similarity)
# tf.summary.scalar("accuracy", accuracy)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	model_name = 'trigram_fc'
	saver = uj.get_saver(model_name, sess, is_continue_training=False)
	train_writer, test_writer = uj.get_writers(sess, model_name)

	model = nn
	evaluation_interval = 250
	save_interval = int(data.train_set_size/float(data.batch_size)) # = epoch
	while(data.epoch_train < 100):
		# get random initialization score on the test-set
		if data.total_steps_train == 0:
			predictions = []
			while(data.epoch_test < 1):
				question1, question2, y = data.next_batch(data_set='test', is_one_hot_encoding=True)
				pred = sess.run(cosine_similarity, feed_dict={input1: question1,
										 input2: question2,
										 labels: y,
										 is_training: False})
				predictions += list(pred)
			print('epoch: ', str(data.epoch_train), evaluation_bench.calculate_metrics(predictions))
			
			# set back to original value, as I use this mechanic later
			data.total_steps_test = 0
			data.step_in_epoch_test = 0 
			data.epoch_test = 0
		iter = sess.run(model.global_step)
		uj.save_model(model_name, saver, sess, i=iter, save_frequency=save_interval, is_saving=True)
		question1, question2, y = data.next_batch(data_set='train', is_one_hot_encoding=True)
		result = sess.run(train_step, feed_dict={input1: question1,
									 input2: question2,
									 labels: y,
									 is_training: True})

		if data.epoch_train > data.epoch_test:
			predictions = []
			while(data.epoch_train > data.epoch_test):
				question1, question2, y = data.next_batch(data_set='test', is_one_hot_encoding=True)
				pred = sess.run(cosine_similarity, feed_dict={input1: question1,
										 input2: question2,
										 labels: y,
										 is_training: False})
				predictions += list(pred)
			print('epoch: ', str(data.epoch_train), evaluation_bench.calculate_metrics(predictions))

		if int(data.total_steps_train) % evaluation_interval == 0:
			c, l, summary = sess.run([cosine_similarity, loss, summaries], feed_dict={input1: question1,
									 input2: question2,
									 labels: y,
									 is_training: False})
			print("train step: %3d, loss: %.6f, min_cosine_sim: %.4f" % (iter, l, np.min(c)))
			train_writer.add_summary(summary, global_step=iter)

			# notice that the validation set result is written to the test.. this is on purpose
			question1, question2, y = data.next_batch(data_set='val', is_one_hot_encoding=True)
			c, l, summary = sess.run([cosine_similarity, loss, summaries], feed_dict={input1: question1,
									 input2: question2,
									 labels: y,
									 is_training: False})
			test_writer.add_summary(summary, global_step=iter)
			print("val step: %3d, loss: %.6f, min_cosine_sim: %.4f" % (iter, l, np.min(c)))
			train_writer.flush()
			test_writer.flush()
	train_writer.close()
	test_writer.close()
	# uj.plot_summary(model_name, uj.max_log(), 'accuracy')
	uj.plot_summary(model_name, uj.max_log(), 'loss')
