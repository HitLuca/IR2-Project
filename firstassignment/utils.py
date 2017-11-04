import six
import tensorflow as tf

from tensorflow.contrib.layers import input_from_feature_columns, fully_connected


# tf.contrib.layers.fully_connected

def spread_out_documents(qid_t, examples):
    if qid_t is not None:
        doc_mask = tf.SparseTensor(indices=qid_t.indices,
                                   values=tf.tile(tf.constant([True]),
                                                  tf.shape(qid_t.values)),
                                   dense_shape=qid_t.dense_shape)
        doc_mask = tf.sparse_tensor_to_dense(doc_mask, default_value=False)
        doc_mask = tf.reshape(doc_mask, [-1])

    result = {}
    for featid in examples:
        sparse_tensor = examples[featid]
        dense_tensor = tf.reshape(tf.sparse_tensor_to_dense(sparse_tensor), [-1])
        if qid_t is not None:
            dense_tensor = tf.boolean_mask(dense_tensor, doc_mask)
        result[featid] = tf.expand_dims(dense_tensor, 1)
    return result


def get_doc_pairs(qid_input, label_input):
    '''
    Returns 2d [?,2] tensor with document pairs
    where [i,0] should be preferred over [i,1].
    '''
    label_t = tf.squeeze(label_input, axis=1)
    less_label = tf.less(*tf.meshgrid(label_t, label_t))
    if qid_input is None:
        return less_label
    else:
        qid_t = tf.squeeze(qid_input, axis=1)
        same_query = tf.equal(*tf.meshgrid(qid_t, qid_t))
        pair_matrix = tf.logical_and(same_query, less_label, 'get_doc_pairs')
        return pair_matrix


def get_all_doc_pairs(qid_input, label_input):
    '''
    Returns 2d [?,2] tensor with document pairs
    where [i,0] should be preferred over [i,1]. and more
    '''
    if qid_input is not None:
        qid_t = tf.squeeze(qid_input, axis=1)
        same_query = tf.equal(*tf.meshgrid(qid_t, qid_t))
    num_docs = tf.shape(label_input)[0]
    same_doc = tf.cast(tf.eye(num_docs), tf.bool)

    label_t = tf.squeeze(label_input, axis=1)
    x, y = tf.meshgrid(label_t, label_t)
    less_label = tf.less(x, y)
    equal_label = tf.logical_and(tf.equal(x, y), tf.logical_not(same_doc))
    more_label = tf.greater(x, y)

    if qid_input is None:
        return less_label, equal_label, more_label
    else:
        less_matrix = tf.logical_and(same_query, less_label)
        equal_matrix = tf.logical_and(same_query, equal_label)
        more_matrix = tf.logical_and(same_query, more_label)
        return less_matrix, equal_matrix, more_matrix


def get_doc_ranks(qid_input, score_input):
    less, equal, more = get_all_doc_pairs(qid_input, score_input)
    uncorrected_rank = tf.reduce_sum(tf.cast(more, tf.int32), axis=1, keep_dims=True)

    def uncorrected(): return uncorrected_rank

    _, _, noise_more = get_all_doc_pairs(qid_input, tf.random_uniform(tf.shape(score_input)))
    rank_corrections = tf.reduce_sum(tf.cast(tf.logical_and(noise_more, equal), tf.int32), axis=1, keep_dims=True)

    def corrected(): return uncorrected_rank + rank_corrections

    correction_needed = tf.reduce_any(equal)
    return tf.cond(correction_needed, true_fn=corrected, false_fn=uncorrected)


def get_dcg_per_doc(labels, doc_ranks, top_k=10):
    nominators = tf.cast(2 ** labels - 1, tf.float32)
    if top_k is not None:
        nominators = nominators * tf.cast(tf.less(doc_ranks, 10), nominators.dtype)
    denominators = tf.log(tf.cast(doc_ranks + 2, tf.float32)) / tf.log(tf.constant(2, dtype=tf.float32))
    return nominators / denominators


def get_dcg_from_ranks(labels, doc_ranks, top_k=10):
    return tf.reduce_sum(get_dcg_per_doc(labels, doc_ranks, top_k))


def get_ndcg_from_scores(labels, doc_scores, top_k=10):
    doc_ranks = get_doc_ranks(None, doc_scores)
    dcg = get_dcg_from_ranks(labels, doc_ranks)
    ideal_doc_ranks = get_doc_ranks(None, labels)
    ideal_dcg = get_dcg_from_ranks(labels, ideal_doc_ranks)

    def ndcg(): return tf.reduce_sum(dcg / ideal_dcg)

    def nolabels(): return tf.constant(0., dtype=tf.float32)

    return tf.cond(tf.greater(ideal_dcg, 0),
                   true_fn=ndcg,
                   false_fn=nolabels)


def create_scoring_model(features, model_params, reuse_variable_scope=False):
    """Creates the model that scores documents.
    Returns a Tensor [n_docs, 1] with document scores.
    """
    feature_columns = model_params['feat_columns']
    hidden_layers = model_params.get('hidden_layers', [20, 10])

    with tf.variable_scope('dnn',
                           values=(tuple(six.itervalues(features))),
                           reuse=reuse_variable_scope
                           ) as dnn_parent_scope:
        with tf.variable_scope('input_from_feature_columns'):
            # Takes the example input and converts into tensors,
            # according to the provided feature_columns.
            net = input_from_feature_columns(features, feature_columns)
        for layer_id, num_hidden_units in enumerate(hidden_layers):
            with tf.variable_scope("hiddenlayer_%d" % layer_id,
                                   values=(net,)
                                   ) as dnn_hidden_layer_scope:
                # Non-linear hidden layer with default activation function (ReLu).
                net = fully_connected(net, num_hidden_units,
                                      variables_collections=[dnn_parent_scope])
        # Final output is linear, allowing for regression.
        return fully_connected(net, 1, activation_fn=None)
