import glob
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import shutil
from tensorboard.backend.event_processing import event_accumulator
import time
from matplotlib.colors import ListedColormap
from matplotlib import cm

def is_dir_empty(path):
    return next(os.scandir(path), None) is None

def print_message(msg):
    print('########################### ' + msg)

def get_saver(model_name, sess, is_continue_training=False, max_to_keep=10):
    saver = tf.train.Saver(max_to_keep=max_to_keep)
    create_dir_if_not_excist('./check_points/')
    if is_continue_training:
        if get_latest_model_save(model_name):
            saver = tf.train.import_meta_graph(get_latest_model_save(model_name) + '.meta')
            saver.restore(sess, get_latest_model_save(model_name))
            print_message('Latest checkpoint restored!')
        else:
            print_message('Tried to continue training but no previous file found!')
    return saver

def save_model(model_name, saver, sess, i, save_frequency=100, is_saving=True):
    if is_saving:
        if (i % save_frequency == 0 and not i == 0) or i == -1:
            saver.save(sess, './check_points/'+model_name, global_step=i)
            print_message('Checkpoint Saved!')

def remove_empty_folders(logdir):
    folders = glob.glob(logdir + '*')
    for f_path in folders:

        if os.path.isdir(f_path) and is_dir_empty(f_path):
            to_remove = list(set([f_path.replace('test', 'train'), f_path.replace('train', 'test'),
             f_path.replace('val', 'train'), f_path.replace('val', 'test'),
             f_path.replace('test', 'val'), f_path.replace('train', 'val')]))
            for p in to_remove:
                if os.path.isdir(p):
                    shutil.rmtree(p)

def get_writers(sess, model_name, val=False):
    logdir = './logs/'
    remove_empty_folders(logdir)
    create_dir_if_not_excist(logdir)
    id = get_identifier_of_run(logdir)
    train_writer = tf.summary.FileWriter(logdir + '/' + id + '_train_' + model_name, graph=sess.graph)
    test_writer = tf.summary.FileWriter(logdir + '/' + id + '_test_' + model_name)
    if val:
        val_writer = tf.summary.FileWriter(logdir + '/' + id + '_val_' + model_name)
        return train_writer, test_writer, val_writer
    else:
        return train_writer, test_writer

def create_dir_if_not_excist(logdir):
    if not os.path.exists(logdir):
        os.makedirs(logdir)


def plot_summary(model_name, run_id, metric, x_label='step', y_label=None):
    time.sleep(1)
    create_dir_if_not_excist('./plots/')
    df = get_df('./logs/')
    df = df[(df['name'] == model_name) & (df['run_id'] == str(run_id))]

    if not y_label:
        y_label = metric
    df_sel = df.groupby(['step', 'mode'], as_index=False).mean()
    df_sel = df_sel.set_index('step')
    df_group = df_sel.groupby('mode')[metric]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    df_group.plot(legend=True, grid=True, title=model_name, ax=ax)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    save_fn = str(run_id) + '_' + model_name + '_' + metric
    fig.savefig('./plots/' + save_fn + '.pdf', format='pdf', bbox_inches=None, pad_inches=0)
    plt.close()

def restore_previous_session(is_continue_training, sess, path, max_to_keep=10):
    saver = tf.train.Saver(max_to_keep=max_to_keep)
    if is_continue_training:
        if get_latest_model_save(path):
            saver.restore(sess, get_latest_model_save(path))

def write_to_tensorboard(writers, global_step, summaries):
    for i, wr in enumerate(writers):
        wr.write(summaries[i], global_step)

def print_results(i, dicts):
    """
    assumes dicts are in order [train, val, test]
    :param i: 
    :param dicts: 
    :return: 
    """
    keys = []
    for d in dicts:
        keys += d.keys()
    keys = np.unique(keys)
    string = '[batch] %04d' % i
    sep_key = '/'
    sep_metric = '\t'
    precision = '{:.6f}'
    not_available = -9.99
    for k in keys:
        values = []
        for d in dicts:
            if d[k]:
                values.append(d[k])
            else:
                values.append(not_available)
        string += sep_metric + '[' + k + ']' + ((precision+sep_key)*len(values))[:-1].format(*values)
    print(string)

######################## helper functions
def get_graph_confusion_matrix(logits, y, n_classes):
    confusion_matrix = tf.confusion_matrix(tf.argmax(logits, 1), tf.argmax(y, 1), num_classes=n_classes, dtype=tf.int32)
    return confusion_matrix

def plot_run_confusion_matrix(matrix, save_name, col_names=None, is_annot=True):
    df = pd.DataFrame(matrix[0])
    if not col_names:
        col_names = list(range(len(df)))
    df.index = col_names, df.columns = col_names
    sns.heatmap(df, cmap="YlGnBu", annot=is_annot, fmt="d", square=True)

    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.tight_layout()
    plt.savefig(save_name + '.pdf', format='pdf', bbox_inches='tight', dpi=1000)


def get_identifier_of_run(path):
    dirs = [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
    if not dirs:
        return '0'
    else:
        import ntpath
        return str(np.max([int(ntpath.basename(i).split('_')[0]) for i in dirs]) + 1)

def max_log():
    dirs = [os.path.join('./logs/', o) for o in os.listdir('./logs/') if os.path.isdir(os.path.join('./logs/', o))]
    if not dirs:
        return '0'
    else:
        import ntpath
        return np.max([int(ntpath.basename(i).split('_')[0]) for i in dirs])

def get_latest_model_save(model):
    path = './check_points/' + model
    f_paths = glob.glob(path + '*.meta')
    print(f_paths)
    if f_paths:
        files = [int(p.replace(path + '-', '').replace('.meta', '')) for p in f_paths if '.meta' in p]
        if files:
            max_save_step = np.max(files)
            return path + '-' + str(max_save_step)
        else:
            return None
    else:
        return None


def obtain_df(inpath):
    sg = {event_accumulator.COMPRESSED_HISTOGRAMS: 1,
          event_accumulator.IMAGES: 1,
          event_accumulator.AUDIO: 1,
          event_accumulator.SCALARS: 0,
          event_accumulator.HISTOGRAMS: 1}
    ea = event_accumulator.EventAccumulator(inpath, size_guidance=sg)
    ea.Reload()
    scalar_tags = ea.Tags()['scalars']

    df = pd.DataFrame(columns=scalar_tags + ['wall_time', 'step'])
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        scalars = list(map(lambda x: [x.value, x.wall_time, x.step], events))
        scalars = np.array(scalars)
        cols = [tag, 'wall_time', 'step']
        for i, c in enumerate(cols):
            df.loc[:, c] = scalars[:, i]
    return df


def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))


def get_run_info_from_path(path):
    dir_name = os.path.basename(os.path.dirname(path))
    splits = dir_name.split('_')
    run_id, tr_or_te, name = splits[0], splits[1], splits[2]
    return run_id, tr_or_te, name


def get_filepaths_from_dirs(dirs):
    f_names = []
    for d in dirs:
        f_names.append(d + '/' + os.listdir(d)[0])
    return f_names


def is_substring_in_string(list_or_element, string):
    if type(list_or_element) == type([]):
        for i in list_or_element:
            if i in string:
                return True
    else:
        if list_or_element in string:
            return True
    return False


def translate_columns_to_one_naming_convension(df):
    acc = ['acc']
    cost = ['cost', 'loss']
    wall_time = 'wall_time'
    step = 'step'
    new_columns = []
    for c in df.columns:
        if is_substring_in_string(acc, c):
            new_columns.append('accuracy')
        elif is_substring_in_string(cost, c):
            new_columns.append('loss')
        elif wall_time is c:
            new_columns.append(wall_time)
        elif step is c:
            new_columns.append(step)
        else:
            raise ValueError(
                "Unknown tags: " + c + " please add it in function translate_columns_to_one_naming_convention")
    df.columns = new_columns


def get_df(logdir):
    dirs = listdir_nohidden(os.getcwd() + logdir[1:])
    print(dirs)
    f_paths = get_filepaths_from_dirs(dirs)

    # df = pd.DataFrame(columns=['run_id', 'mode', 'name', 'step', 'wall_time', 'tag', 'score'])
    df_list = []
    for f in f_paths:
        run_id, mode, name = get_run_info_from_path(f)
        df_temp = obtain_df(f)
        translate_columns_to_one_naming_convension(df_temp)
        df_temp['run_id'] = run_id
        df_temp['mode'] = mode
        df_temp['name'] = name
        df_temp['wall_time'] = df_temp['wall_time'] - df_temp['wall_time'].min()
        df_list.append(df_temp)
    return pd.concat(df_list, axis=0, ignore_index=True)

def plot_multiple_summaries(model_names_plus_run_id, metric, train=False, test=True, val=False, x_label='step', y_label=None):
    df = get_df('./logs/')
    if not train:
        df = df[~(df['mode'] == 'train')]
    if not test:
        df = df[~(df['mode'] == 'test')]
    if not val:
        df = df[~(df['mode'] == 'val')]
    df['combined'] = df['name'] + '_' + df['run_id']
    df = df[df['combined'].isin(model_names_plus_run_id)]
    df['combined'] = df['combined'] + '_' + df['mode'].values.astype(str)
    if not y_label:
        y_label = metric
    df_sel = df.groupby(['step', 'combined'], as_index=False).mean()
    df_sel = df_sel.set_index('step')
    df_group = df_sel.groupby('combined')[metric]
    sns.set_style("ticks")
    ax = df_group.plot(legend=True, grid=True, title='Comparison of Runs')
    ax = ax.values
    ax = ax[0]
    colormap = sns.color_palette("hls", len(ax.lines))

    leg = ax.get_legend()
    for i, j in enumerate(ax.lines):
        leg.legendHandles[i].set_color(colormap[i])
        j.set_color(colormap[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    save_fn = '_'.join(model_names_plus_run_id) + '_' + metric
    plt.savefig('./plots/' + save_fn + '.pdf', format='pdf', bbox_inches=None, pad_inches=0)
    plt.close()

def get_possible_summaries():
    df = get_df('./logs/')
    df['combined'] = df['name'] + '_' + df['run_id']
    return np.unique(df['combined'])

# plot_multiple_summaries(get_possible_summaries(),'loss')
# plot_multiple_summaries(get_possible_summaries(),'accuracy')