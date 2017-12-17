import lib.eval.utils_joop as uj
import lib.eval.evaluation_bench

import tensorflow as tf
import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))) + '/data/'
sys.path.append(path)  # to correctly import Dataset

from lib.models.LSTM import LSTM
from yahoo_dataset_tfrc import LSTMDataset_TFRC

dataset_folder = './../data/Yahoo'

# non pruned ds, 5k
training_set_filename = 'train_lstm_5000.tfrecord'
testing_set_filename = 'train_lstm_5000.tfrecord'
validating_set_filename = 'train_lstm_5000.tfrecord'

# test set, 20k
# training_set_filename = 'data_lstm_test.tfrecord'
# testing_set_filename = 'data_lstm_test.tfrecord'
# validating_set_filename = 'data_lstm_test.tfrecord'

### New embedding (includes vocab from train/val/test split
vocabulary_filepath = './../data/Embedding/vocabulary.txt'
embeddings_filepath = './../data/Embedding/partial_embedding_matrix.npy'

batch_size = 256
learning_rate = 0.01
max_steps = 10000
lstm_num_layers = 1
lstm_num_hidden = 128
train_embedding = False
training_epochs = 20
max_length = 40

training_set_path = os.path.join(dataset_folder, training_set_filename)
testing_set_path = os.path.join(dataset_folder, testing_set_filename)
validating_set_path = os.path.join(dataset_folder, validating_set_filename)

# create a dataset processing module
dataset_train = LSTMDataset_TFRC(training_set_path, batch_size, training_epochs, max_length, True, vocabulary_filepath)
dataset_test = LSTMDataset_TFRC(testing_set_path, batch_size, 1, max_length, False, vocabulary_filepath)
dataset_val = LSTMDataset_TFRC(validating_set_path, batch_size, training_epochs, max_length, True, vocabulary_filepath)

# create dataset iterator
iterator_train = dataset_train()
iterator_test = dataset_test()
iterator_val = dataset_val()

# load the next batch for both training/testing
q1_train, q2_train, label_train = iterator_train.get_next()
q1_test, q2_test, label_test = iterator_test.get_next()
q1_val, q2_val, label_val = iterator_val.get_next()

# define the placeholders
is_training = tf.placeholder(tf.bool)
batch_length = tf.placeholder(name='batch_length', dtype=tf.int32)
embedding_matrix = tf.placeholder(name='embedding_matrix', dtype=tf.float32, shape=[None, None])

# initialize the network
nn = LSTM(is_training,
          vocabulary_filepath,
          batch_size=batch_size,
          lstm_num_layers=lstm_num_layers,
          lstm_num_hidden=lstm_num_hidden,
          use_tfrecord=True)

# embedding method (train/pre-trained)
if train_embedding is False:
    nn.assign_embedding_matrix(embedding_matrix)
    np_embedding_matrix = np.load(embeddings_filepath)
else:
    np_embedding_matrix = [[None]]

# perform inference on the training set
with tf.variable_scope("inference"):
    inference = nn.inference(q1_train, q2_train)
    loss = nn.loss(label_train, inference)
    train_step = nn.train_step(loss, learning_rate)
    accuracy = nn.accuracy(label_train, inference)
    predicted = nn.predict(inference)

# reuse the network for testing on the whole dataset
with tf.variable_scope("inference", reuse=True):
    inference_test = nn.inference(q1_test, q2_test)
    loss_test = nn.loss(label_test, inference_test)
    accuracy_test = nn.accuracy(label_test, inference_test)

with tf.variable_scope("inference", reuse=True):
    inference_val = nn.inference(q1_val, q2_val)
    loss_val = nn.loss(label_val, inference_val)
    accuracy_val = nn.accuracy(label_val, inference_val)


# log the training behavior
tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)
summary_op = tf.summary.merge_all()

# start a session and initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
sess.run(tf.tables_initializer())

# initializer data iterators
sess.run(iterator_train.initializer)
sess.run(iterator_test.initializer)
sess.run(iterator_val.initializer)

# # evaluation
# evaluation_interval = 250     # evaluation interval is ARBITRARY
# save_interval = 1000          # Every Epoch
# model_name = 'trigram_fc'
# saver = uj.get_saver(model_name, sess, is_continue_training=False)
# train_writer, test_writer = uj.get_writers(sess, model_name)


# handling multi-threading
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

train_writer = tf.summary.FileWriter('./log_dir/lstm_train', sess.graph)
# test_writer = tf.summary.FileWriter('./log_dir/lstm_test')

f = open("training_stat.txt", "w")
f.write("step\tloss\n")
f.flush()

_loss_train, _acc_train = list(), list()
_loss_test, _acc_test = list(), list()
_step_train = list()
_step_test = list()

for i in range(100000):
    try:
        result = sess.run([loss, accuracy, train_step, label_train, predicted, inference, summary_op],
                          feed_dict={
                              is_training: True,
                              embedding_matrix: np_embedding_matrix
                          })

        # TODO: Change 10 to evaluation interval
        if i % 10 == 0:
            print("step: %3d, loss: %.6f, acc: %.6f" % (i, result[0], result[1]))
            train_writer.add_summary(result[-1], global_step=i)
            # print(np.min(result[-2]), np.max(result[-2]))

            _loss_train.append(result[0])
            _acc_train.append(result[1])
            _step_train.append(i)

            # TODO: Run through the evaluation set
            # TODO: Write validation summary to test summary
            # validation_result = sess.run([])
            # test_writer.add_summary(summary, global_step=i)

    except tf.errors.OutOfRangeError:
        print("End of training!")
        break

    # Run through the test set
    # TODO: Set 100 to the number of steps for an epoch
    if i % 100 == 0:
        print("Iterating through the test set...")
        total_loss, total_acc = list(), list()
        # TODO: Initialize a list to store the predictions
        predictions = list()

        while True:
            try:
                # TODO: Get the prediction of relevance score
                # test_loss, test_acc = sess.run([loss_test, accuracy_test])
                pred, test_loss, test_acc = sess.run([inference_test, loss_test, accuracy_test])
                total_loss.append(test_loss)
                total_acc.append(test_acc)

                # TODO: Append the prediction to a list
                # predictions += list(pred)

            except tf.errors.OutOfRangeError:
                mean_loss = np.array(total_loss).mean()
                mean_acc = np.array(total_acc).mean()

                _loss_test.append(mean_loss)
                _acc_test.append(mean_acc)
                _step_test.append(i)

                print("Testing at step: {}, loss: {}, acc: {}".format(i, mean_loss, mean_acc))

                # TODO: Call the evaluation_bench to calculate the scores based on predictions
                # TODO: Save the result of the evaluation bench

                f.write(str(i) + "\t" + str(mean_loss) + "\n")
                break
        # reset the test set iterator
        sess.run(iterator_test.initializer)

    # TODO: Get the global_step
    # TODO: save model
    '''
    iter = sess.run(model.global_step)
    uj.save_model(model_name, saver, sess, i=iter, save_frequency=save_interval, is_saving=True)
    '''


# # plot the train/test loss/acc
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.title("Loss")
# plt.plot(np.array(_step_train), np.array(_loss_train), label="Training Loss")
# plt.plot(np.array(_step_test), np.array(_loss_test), label="Testing Loss")
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.title("Accuracy")
# plt.plot(np.array(_step_train), np.array(_acc_train), label="Training Acc")
# plt.plot(np.array(_step_test), np.array(_acc_test), label="Testing Acc")
# plt.legend()
# plt.show()


f.flush()
f.close()

coord.request_stop()
coord.join(threads)

sess.close()
