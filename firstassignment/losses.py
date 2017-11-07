## Students:
# Heng Lin 11392533
# Joop Pascha 10090614
# Luca Simonetto 11413522
# Muriel Hol 10161740

import tensorflow as tf


##################################################
# Loss functions
# Implement these functions for this assignment.
# Each function should return a 1d scalar tensor
# that holds the appropriate loss to be minimized.
# For this task you may assume binary relevance.
##################################################

# doc_scores is a logits tensor with (float32) scores for the documents,
# labels is a (integer) tensor of the same size containing matching relevance labels.
def pointwise_regression_loss(doc_scores, labels):
    return tf.reduce_sum(
        (doc_scores - tf.cast(labels, tf.float32)) ** 2
    )


def pointwise_classification_loss(doc_scores, labels):
    return -tf.reduce_sum(tf.log(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=doc_scores) + 1e-10))


def pairwise_loss(doc_scores, labels):

    # using sigma=1 for the logistic function
    def pairwise_cost(i, j, sigma=1):
        return tf.log(1 + tf.exp(-sigma * (i - j)))

    si = doc_scores[:-1, :]
    sj = doc_scores[1:, :]

    li = labels[:-1, :]
    lj = labels[1:, :]

    # mask for correct ordered entries
    mask0 = tf.cast(tf.greater(li, lj), tf.float32)
    # mask for incorrect ordered entries
    mask1 = tf.cast(tf.less(li, lj), tf.float32)
    # mask for the entries that should be equally ordered
    mask2 = tf.cast(tf.equal(li, lj), tf.float32)

    sigma = 1.0

    S_ij = mask0 + (-1.0 * mask1)
    _loss = 0.5 * (1 - S_ij) * sigma * (si - sj)

    loss = _loss + mask0 * pairwise_cost(si, sj) + mask1 * pairwise_cost(sj, si) + mask2 * pairwise_cost(si, sj)

    return tf.reduce_sum(loss)


def listwise_loss(doc_scores, labels):

    # use the top k entries to calculate the cost
    def scoreK(s, k=6):
        s_exp = tf.exp(s)
        den = tf.cumsum(s_exp, reverse=True)
        return tf.reduce_prod(tf.divide(s_exp[:k, :], den[:k]))

    # calculate the top k score for docs and labels
    p_score = scoreK(doc_scores)
    p_labels = scoreK(tf.cast(labels, tf.float32))

    # return the loss (cross entropy between the score and labels)
    return -tf.multiply(p_labels, tf.log(p_score)) * 100000000
