import lib.eval.utils_joop as uj
import lib.eval.evaluation_bench as evaluation_bench

import tensorflow as tf
import os
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle

path = os.path.dirname(os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))) + '/data/'
sys.path.append(path)  # to correctly import Dataset

from lib.models.LSTM import LSTM
from yahoo_dataset_tfrc import LSTMDataset_TFRC

dataset_folder = './../data/Yahoo'

# non pruned ds, 5k
# training_set_filename = 'train_lstm_20000.tfrecord'
# testing_set_filename = 'train_lstm_20000.tfrecord'
# validating_set_filename = 'train_lstm_20000.tfrecord'

# test set, 20k
training_set_filename = 'test_lstm_1018.tfrecord'
testing_set_filename = 'test_lstm_1018.tfrecord'
validating_set_filename = 'test_lstm_1018.tfrecord'

# embedding
vocabulary_filepath = './../data/Embedding/vocabulary.txt'
embeddings_filepath = './../data/Embedding/partial_embedding_matrix.npy'

batch_size = 256
learning_rate = 0.01
max_steps = 1000000
lstm_num_layers = 1
lstm_num_hidden = 128
train_embedding = False
training_epochs = 40
max_length = 40

evaluation_interval = 100     # evaluation interval is ARBITRARY
save_interval = 11718          # Every Epoch
model_name = 'lstm'

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
    inference, _ = nn.inference(q1_train, q2_train)
    loss = nn.loss(label_train, inference)
    train_step = nn.train_step(loss, learning_rate)
    accuracy = nn.accuracy(label_train, inference)
    predicted = nn.predict(inference)

# reuse the network for testing
with tf.variable_scope("inference", reuse=True):
    inference_test, inference_test_sigmoid = nn.inference(q1_test, q2_test)
    loss_test = nn.loss(label_test, inference_test)
    accuracy_test = nn.accuracy(label_test, inference_test)

# reuse the network for validating
with tf.variable_scope("inference", reuse=True):
    inference_val, _ = nn.inference(q1_val, q2_val)
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

saver = uj.get_saver(model_name, sess, is_continue_training=False)

# handling multi-threading
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

train_writer = tf.summary.FileWriter('./log_dir/lstm_train', sess.graph)

f = open("training_stat.txt", "w")
f.write("step\ttrain_loss\ttrain_acc\tval_loss\tval_acc\n")
f.flush()

# store the prediction result and evaluation result
eval_result_list = list()
pred_result_list = list()

for i in range(max_steps):
    # run training session
    try:
        result = sess.run([loss, accuracy, train_step, label_train, predicted, inference, summary_op],
                          feed_dict={
                              is_training: True,
                              embedding_matrix: np_embedding_matrix
                          })
        if i % 10 == 0:
            print("step: %3d, loss: %.6f, acc: %.6f" % (i, result[0], result[1]))
            train_writer.add_summary(result[-1], global_step=i)
    except tf.errors.OutOfRangeError:
        print("End of training!")
        break

    # run validating session
    if i % evaluation_interval == 0:
        try:
            total_loss, total_acc = list(), list()

            _loss_val, _acc_val, _inf_val = sess.run([loss_val, accuracy_val, inference_val],
                                                     feed_dict={is_training: False})

            print("Validating at step: {}, loss: {}, acc: {}".format(i, _loss_val, _acc_val))

            f.write(str(i) + "\t" + str(result[0]) + "\t" + str(result[1]) + "\t"
                    + str(_loss_val) + "\t" + str(_acc_val) + "\t" + "\n")
            f.flush()
        except tf.errors.OutOfRangeError:
            sess.run(iterator_val.initializer)  # reset the test set iterator

    # Run through the test set
    if i % save_interval == 0:
        print("Testing at step: %d..." % i)

        predictions = list()    # Initialize a list to store the predictions

        while True:
            try:
                pred_test, test_loss, test_acc = sess.run([inference_test_sigmoid, loss_test, accuracy_test],
                                                          feed_dict={is_training: False})
                predictions += list(np.squeeze(pred_test))

            except tf.errors.OutOfRangeError:
                print("Running Evaluation Benchmark...")

                eval_result = evaluation_bench.calculate_metrics(predictions)

                # Save the eval_result, predictions
                eval_result_list.append({"epoch": i, "data": eval_result})
                pred_result_list.append({"epoch": i, "predictions": predictions})

                with open("eval_results.p", "wb") as f_eval:
                    pickle.dump(eval_result_list, f_eval)
                with open("pred_results.p", "wb") as f_pred:
                    pickle.dump(pred_result_list, f_pred)

                break
        sess.run(iterator_test.initializer)     # reset the test set iterator

    iter = sess.run(nn.global_step)
    uj.save_model(model_name, saver, sess, i=iter, save_frequency=100, is_saving=True)

f.flush()
f.close()

coord.request_stop()
coord.join(threads)

sess.close()
