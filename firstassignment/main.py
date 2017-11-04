""" Simple example for performing Learning to Rank."""
import argparse
import utils as bsu
import losses
import tensorflow as tf

from tensorflow.contrib.training import wait_for_new_checkpoint

parser = argparse.ArgumentParser(description='Deep Learning '
                    'to Rank Assignment')
parser.add_argument('--model_dir', type=str, default=None,
                    help='Directory to store/load model.')
parser.add_argument('--input_dir', type=str, required=True,
                    help='Directory where input is found '
                      '(features.txt, [train, vali, test].tfrecord).')
parser.add_argument('--steps', type=int, default=20000,
                    help='Number of gradient steps the model should take.')
parser.add_argument('--eval_steps', type=int, default=50,
                    help='Saves model at every eval_steps for eveluation.')
parser.add_argument('--train', action='store_true',
                    help='Perform training on training set.')
parser.add_argument('--validation', action='store_true',
                    help='Run evaluation on validation set.')
parser.add_argument('--test', action='store_true',
                    help='Run evaluation on test set.')
parser.add_argument('--pointwise_regr', action='store_true',
                    help='Use pointwise regression loss.')
parser.add_argument('--pointwise_class', action='store_true',
                    help='Use pointwise classification loss.')
parser.add_argument('--pairwise', action='store_true',
                    help='Use pairwise loss.')
parser.add_argument('--listwise', action='store_true',
                    help='Use listwise loss.')

args = parser.parse_args()

params = {
  'randomize_input': args.train,
  'input_queue_capacity': 1000,
  'input_num_threads': 1,
  'input_read_batch_size': 1,
  'input_seed': None,
  'hidden_units': [20],
  'learning_rate': 0.01,
  'model_dir': args.model_dir,
  'steps': args.steps,
  'eval_steps': args.eval_steps,
  'vali_size': 30,
  'test_size': 30,
}

#########################################
# Data Input
# These values will appear in Tensorboard
#########################################

assert sum([args.validation, args.train, args.test]) == 1
if args.train:
  params['dataset'] = 'train'
elif args.validation:
  params['dataset'] = 'vali'
else:
  params['dataset'] = 'test'

assert sum([args.pointwise_regr,
            args.pointwise_class,
            args.pairwise,
            args.listwise]
          ) == 1
if args.pointwise_regr:
  params['loss'] = losses.pointwise_regression_loss
elif args.pointwise_class:
  params['loss'] = losses.pointwise_classification_loss
elif args.pairwise:
  params['loss'] = losses.pairwise_loss
else:
  params['loss'] = losses.listwise_loss

def input_fn(params):
  '''
  Outputs batches of all documents beloning to one query.

  Returns:
    A dictionary of tensors from the given dataset.
  '''
  file_pattern = '%s%s.*-of-*.tfrecord' % (args.input_dir,
                                    params['dataset'])
  batched_examples = tf.contrib.learn.read_batch_examples(
    file_pattern=file_pattern,
    batch_size=1,
    reader=tf.TFRecordReader,
    randomize_input=params['randomize_input'],
    num_epochs=None,
    queue_capacity=params['input_queue_capacity'],
    num_threads=params['input_num_threads'],
    read_batch_size=params['input_read_batch_size'],
    parse_fn=None,
    name='%s_input_reader' % params['dataset'],
    seed=params['input_seed']
  )

  # Parses all documents for a query and stores it
  # as a single example, i.e. dimensions [1, ?, ...].
  examples = tf.parse_example(batched_examples,
                              features=features)
  # 'Spreads' the documents per query across the first
  # dimension, i.e. dimensions [?, ...].
  examples = bsu.spread_out_documents(None, examples)

  labels = examples['label']

  return examples, labels

# Features keeps track of the features that should
# be parsed from the TFRecord examples.
features = {'qid': tf.VarLenFeature(dtype=tf.int64),
          'label': tf.VarLenFeature(dtype=tf.int64)}
# All Lerot features will be parsed as scalars,
# the features.txt file is used to keep track of
# their names and numbers.
with open(args.input_dir + '/features.txt', 'r') as f:
  for fid in f:
    # Every TFRecord contains all documents for a single query.
    # These features are of variable lenght, since the number
    # of documents can vary per query.
    features[fid.strip()] = tf.VarLenFeature(dtype=tf.float32)

# feature_columns will be used to parse our features.
# It wil contain FeatureColumn objects that explain
# how the input should be parsed.
# In the Lerot case every feature is just a scalar.
feature_columns = []
for featid in features:
  if featid not in ['qid', 'label']:
    feature_columns.append(
      tf.contrib.layers.real_valued_column(
        featid, dimension=1,
        default_value=0, dtype=tf.float32))

params['feat_columns'] = feature_columns

documents, labels = input_fn(params)

#########################################
# Scoring and loss calculation.
#########################################

# The scores the model assigns the documents,
# these are 'logits'.
doc_scores = bsu.create_scoring_model(documents, params)

# Loss function to minimize.
loss = params['loss'](doc_scores, labels)

# Metrics
ndcg = bsu.get_ndcg_from_scores(labels, doc_scores)

#########################################
# Summary writers
# These values will appear in Tensorboard
#########################################

# Report the loss.
tf.summary.scalar('loss', loss)

# Average scores for relevant and non-relevant docs.
relevant_scores = tf.boolean_mask(doc_scores,
                                  tf.greater(labels,0))
irrelevant_scores = tf.boolean_mask(doc_scores,
                                    tf.equal(labels,0))

if params['dataset'] == 'train':
  # During training input is shuffled and we can't
  # easily take the average of the entire set of
  # queries, thus we only look at the current query.
  ndcg_mean = tf.reduce_mean(ndcg)
  # Keeping track of score ranges.
  doc_scores_mean = tf.reduce_mean(doc_scores)
  rel_mean = tf.reduce_mean(relevant_scores)
  irrel_mean = tf.reduce_mean(irrelevant_scores)
else:
  # In evaluation keeping track of the mean is
  # straightforward.
  def mean_helper(values):
    return tf.metrics.mean(tf.reduce_mean(values))
  ndcg_mean, ndcg_update = mean_helper(ndcg)
  doc_scores_mean, doc_scores_update = mean_helper(doc_scores)
  rel_mean, rel_update = mean_helper(relevant_scores)
  irrel_mean, irrel_update = mean_helper(irrelevant_scores)

  update_op = tf.group(ndcg_update, doc_scores_update,
                       rel_update, irrel_update)

# Report statisctic in Tensorboard.
tf.summary.scalar('ndcg', ndcg_mean)
tf.summary.scalar('scores/average', doc_scores_mean)
tf.summary.scalar('scores/relevant', rel_mean)
tf.summary.scalar('scores/irrelevant', irrel_mean)
merged_summary = tf.summary.merge_all()



# The variable used to keep track of the number of gradient
# steps the model has taken.
global_step = tf.Variable(0, trainable=True,
                             name='global_step')
# We use the Adam optimizer to update our model.
opt = tf.train.AdagradOptimizer(
                  learning_rate=params['learning_rate'])
opt_op = opt.minimize(loss, global_step=global_step)

# Operator to initialize the network and other variables.
init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())

# Folder to store summaries, unique per dataset.
summary_folder = '%s/%s' % (params['model_dir'],
                            params['dataset'])
writer = tf.summary.FileWriter(summary_folder)

# Operator to save checkpoint.
saver = tf.train.Saver()


# Our loop is different depening on whether,
# we are performing training or evaluation.
if params['dataset'] == 'train':

  # Train supervisor to handle starting sessions.
  sv = tf.train.Supervisor(logdir=args.model_dir,
                           summary_writer=writer,
                           save_summaries_secs=10,
                           global_step=global_step,
                           # manual saving since learning is fast
                           save_model_secs=0)

  # At this point the model will be instantiated and actually ran.
  with sv.managed_session() as sess:

    # Continue from a previous saved checkpoint, if it exists.
    checkpoint = tf.train.latest_checkpoint(args.model_dir)
    if checkpoint:
      print 'Loading checkpoint', checkpoint
      saver.restore(sess, checkpoint)
    else:
      print 'No existing checkpoint found.'
      sess.run(init)

    # Check the current state of the network.
    i = sess.run([global_step])[0]
    print 'Running %d steps.' % (params['steps'] - i)
    while i < params['steps']:
      i = sess.run([global_step, opt_op])[0]
      # Evaluation will be performed on saved checkpoints
      # only. Since learning goes very fast, we save often.
      if i % params['eval_steps'] == 0 or i == params['steps']-1:
        saver.save(sess, args.model_dir + 'model.ckpt', global_step=i)
else:
  print 'Evaluating on', params['dataset']
  # For each checkpoint the entire dataset is evaluated.
  steps_per_eval = params['%s_size' % params['dataset']]
  checkpoint = None
  # Basic session since we will only manually save summaries.
  with tf.Session() as sess:
    coord = tf.train.Coordinator()
    # Queue runners will take care of reading data in seperate threads.
    threads = tf.train.start_queue_runners(coord=coord)
    while True:
      checkpoint = wait_for_new_checkpoint(args.model_dir,
                                           checkpoint,
                                           seconds_to_sleep=1)
      # Init for variables that are not part of checkpoint,
      # in this case the ones used for metrics.
      sess.run(init)
      # Restore a checkpoint saved by the training run.
      saver.restore(sess, checkpoint)
      # Update the metrics for every element in the dataset.
      for i in range(steps_per_eval):
        sess.run([update_op])
      # Get the resulting metrics.
      cur_step, ndcg_value, cur_summary = sess.run([global_step,
                                                    ndcg_mean,
                                                    merged_summary])
      # Pass the summary to the writer, which stores it for Tensorboard.
      writer.add_summary(cur_summary, global_step=cur_step)
      print 'Step %d: %.02f' % (cur_step, ndcg_value)
      if cur_step == params['steps']:
        break

    coord.request_stop()
    coord.join(threads)
