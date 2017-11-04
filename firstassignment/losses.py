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
    raise NotImplementedError('Pointwise classification is not implemented')


def pairwise_loss(doc_scores, labels):
    raise NotImplementedError('Pairwise is not implemented')


def listwise_loss(doc_scores, labels):
    raise NotImplementedError('Listwise is not implemented')
