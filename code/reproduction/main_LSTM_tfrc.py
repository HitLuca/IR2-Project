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
training_set_filename = 'data_lstm_test.tfrecord'       # TODO: To be modified
testing_set_filename = 'data_lstm_test.tfrecord'

vocabulary_filepath = './lib/data/testset_vocabulary.txt'
embeddings_filepath = './lib/data/testset_partial_embedding_matrix.npy'

batch_size = 64
learning_rate = 0.01
max_steps = 10000
lstm_num_layers = 1
lstm_num_hidden = 128
train_embedding = False

# TODO: Add a tag to allow fine tuning the embedding

training_set_path = os.path.join(dataset_folder, training_set_filename)
testing_set_path = os.path.join(dataset_folder, training_set_filename)

# create a dataset processing module
dataset = LSTMDataset_TFRC(training_set_path, batch_size, 20, 20, True, vocabulary_filepath)
dataset_test = LSTMDataset_TFRC(testing_set_path, batch_size, 10, 20, False, vocabulary_filepath)

# create dataset iterator
iterator = dataset()
iterator_test = dataset_test()

# load the next batch for both training/testing
question1, question2, labels = iterator.get_next()
q1_test, q2_test, labels_test = iterator_test.get_next()

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
    cosine_similarity = nn.inference(question1, question2)
    loss = nn.loss(labels, cosine_similarity)
    train_step = nn.train_step(loss, learning_rate)
    accuracy = nn.accuracy(labels, cosine_similarity)

    predicted = nn.predict(cosine_similarity)

# reuse the network for testing on the whole dataset
with tf.variable_scope("inference", reuse=True):
    cosine_similarity_test = nn.inference(q1_test, q2_test)
    loss_test = nn.loss(labels_test, cosine_similarity_test)
    accuracy_test = nn.accuracy(labels_test, cosine_similarity_test)


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
sess.run(iterator.initializer)
sess.run(iterator_test.initializer)

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
        result = sess.run([loss, accuracy, train_step, labels, predicted, summary_op],
                          feed_dict={
                              is_training: True,
                              embedding_matrix: np_embedding_matrix
                          })
        if i % 10 == 0:
            print("step: %3d, loss: %.6f, acc: %.6f" % (i, result[0], result[1]))
            train_writer.add_summary(result[-1], i)

            # print("***Sanity check***")
            # print("Predicted:    {}".format(np.squeeze(result[5][:5])))
            # print("Cosine Sim:   {}".format(result[3][:5]))
            # print("Ground Truth: {}".format(np.squeeze(result[4][:5])))

            _loss_train.append(result[0])
            _acc_train.append(result[1])
            _step_train.append(i)

    except tf.errors.OutOfRangeError:
        print("End of training!")
        break

    # Run through the test set
    '''
    if i % 100 == 0:
        print("Iterating through the test set...")
        total_loss, total_acc = list(), list()
        while True:
            try:
                test_loss, test_acc = sess.run([loss_test, accuracy_test])
                total_loss.append(test_loss)
                total_acc.append(test_acc)

            except tf.errors.OutOfRangeError:
                mean_loss = np.array(total_loss).mean()
                mean_acc = np.array(total_acc).mean()

                _loss_test.append(mean_loss)
                _acc_test.append(mean_acc)
                _step_test.append(i)

                print("Testing at step: {}, loss: {}, acc: {}".format(i, mean_loss, mean_acc))

                f.write(str(i) + "\t" + str(mean_loss) + "\n")
                break
        # reset the test set iterator
        sess.run(iterator_test.initializer)
    '''


# plot the train/test loss/acc
plt.figure()
plt.subplot(1, 2, 1)
plt.title("Loss")
plt.plot(np.array(_step_train), np.array(_loss_train), label="Training Loss")
plt.plot(np.array(_step_test), np.array(_loss_test), label="Testing Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Accuracy")
plt.plot(np.array(_step_train), np.array(_acc_train), label="Training Acc")
plt.plot(np.array(_step_test), np.array(_acc_test), label="Testing Acc")
plt.legend()
plt.show()


f.flush()
f.close()

coord.request_stop()
coord.join(threads)

sess.close()
