#region imports
import tensorflow as tf
import numpy as np
from lib.models.SDQA.SDQA import SDQA
#endregion


def generate_batch(batch_size, vocabulary_size):
    points1 = np.zeros((batch_size, vocabulary_size))
    points1[np.random.choice(a=[False, True], size=points1.shape)] = 1
    points1 /= np.linalg.norm(points1, ord=2, axis=1)[:, None]
    points2 = np.zeros((batch_size, vocabulary_size))
    points2[np.random.choice(a=[False, True], size=points2.shape)] = 1
    points2 /= np.linalg.norm(points2, ord=2, axis=1)[:, None]

    labels = np.ones(batch_size) * -1
    for row in range(points1.shape[0]):
        dist = np.linalg.norm(points1[row] - points2[row], ord=2)
        if dist <= 0.999:
            labels[row] = 1
    return points1, points2, labels


batch_size = 256
learning_rate = 0.001
max_steps = 1000
loss_margin = 0
vocabulary_size = 10000

# initialize the network
is_training = tf.placeholder(tf.bool)
nn = SDQA(is_training=is_training,
          vocabulary_size=vocabulary_size)

input1 = tf.placeholder(dtype=tf.float32, shape=[None, vocabulary_size])
input2 = tf.placeholder(dtype=tf.float32, shape=[None, vocabulary_size])
labels = tf.placeholder(dtype=tf.float32, shape=[None])

cosine_similarity = nn.inference(input1, input2)
loss = nn.loss(labels, cosine_similarity, margin=loss_margin)
train_step = nn.train_step(loss, learning_rate)
accuracy = nn.accuracy(labels, cosine_similarity)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

for i in range(max_steps):
    points1, points2, label = generate_batch(batch_size, vocabulary_size)
    result = sess.run([cosine_similarity, loss, accuracy, train_step],
                      feed_dict={input1: points1,
                                 input2: points2,
                                 labels: label,
                                 is_training: True})

    print("step: %3d, loss: %.6f, acc: %.3f, min_cosine_sim: %.4f" % (i, result[1], result[2], np.min(result[0])))
    # print(result[1], result[2], np.min(result[0]))
