import tensorflow as tf
from lib.models.LSTM import LSTM
from lib.utils.synthetic_datasets import SyntheticDatasetLSTM

batch_size = 64
learning_rate = 0.001
max_steps = 10000
lstm_num_layers = 2
lstm_num_hidden = 128
seq_max_length = 500
padding_value = 0

# initialize the network
is_training = tf.placeholder(tf.bool)
input1 = tf.placeholder(dtype=tf.float32, shape=[None, None])
input2 = tf.placeholder(dtype=tf.float32, shape=[None, None])
labels = tf.placeholder(dtype=tf.float32, shape=[None])

nn = LSTM(batch_size=batch_size,
          lstm_num_layers=lstm_num_layers,
          lstm_num_hidden=lstm_num_hidden,
          padding_value=padding_value)

cosine_similarity = nn.inference(input1, input2)
loss = nn.loss(labels, cosine_similarity)
train_step = nn.train_step(loss, learning_rate)
accuracy = nn.accuracy(labels, cosine_similarity)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

data = SyntheticDatasetLSTM()

for i in range(max_steps):
    timeseries1, timeseries2, y = data.generate_batch(batch_size, seq_max_length, padding_value)
    result = sess.run([loss, accuracy, train_step],
                      feed_dict={input1: timeseries1,
                                 input2: timeseries2,
                                 labels: y})
    print("step: %3d, loss: %.6f, acc: %.3f" % (i, result[0], result[1]))
