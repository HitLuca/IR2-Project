"""
this file defines different type of loss function
computing the losses by taking logits from two network
"""

import tensorflow as tf


def cosine_sim_loss(logit1, logit2, labels, m=1):
    loss = tf.losses.cosine_distance(logit1, logit2)

    return 0
