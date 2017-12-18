
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Qualitative-Analysis" data-toc-modified-id="Qualitative-Analysis-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Qualitative Analysis</a></div><div class="lev2 toc-item"><a href="#MAP" data-toc-modified-id="MAP-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>MAP</a></div><div class="lev2 toc-item"><a href="#MRR" data-toc-modified-id="MRR-12"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>MRR</a></div><div class="lev2 toc-item"><a href="#P@1" data-toc-modified-id="P@1-13"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>P@1</a></div><div class="lev2 toc-item"><a href="#MAP-with-alpha" data-toc-modified-id="MAP-with-alpha-14"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>MAP with alpha</a></div><div class="lev2 toc-item"><a href="#NDCG" data-toc-modified-id="NDCG-15"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>NDCG</a></div><div class="lev2 toc-item"><a href="#Experiments" data-toc-modified-id="Experiments-16"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>Experiments</a></div><div class="lev1 toc-item"><a href="#Quantitative-Analysis" data-toc-modified-id="Quantitative-Analysis-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Quantitative Analysis</a></div>

# In[21]:


# get_ipython().system('jupyter nbconvert --to script 5_evaluation_bench.ipynb')


# In[1]:


import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import seaborn.apionly as sns
import seaborn
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker
import random
import os

# get_ipython().magic('matplotlib inline')


# In[17]:


# def obtain_queries_result_list():
#     df = pd.read_pickle('./evaluation/test_set_1018.p')
#     index_list = []
#     for i in np.unique(df.qid):
#         index_list.append(df.index[df['qid'] == i].values)
#     return index_list

# query_indices = obtain_queries_result_list()
# pickle.dump(query_indices, open('./evaluation/test_query_indices.p', 'wb'))


# # Qualitative Analysis

# ## MAP

# In[2]:


def mean_average_precision(rs):
    """Score is mean average precision

    Relevance is binary (nonzero is relevant).

    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666

    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])

def average_precision(r):
    """Score is average precision (area under PR curve)

    Relevance is binary (nonzero is relevant).

    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


# ## MRR

# In[3]:


def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item

    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).

    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75

    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


# ## P@1

# In[4]:


def precision_at_k(r, k):
    """Score is precision @ k

    Relevance is binary (nonzero is relevant).

    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k


    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Precision @ k

    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def r_precision(r):
    """Score is precision after all relevant documents have been retrieved

    Relevance is binary (nonzero is relevant).

    >>> r = [0, 0, 1]
    >>> r_precision(r)
    0.33333333333333331
    >>> r = [0, 1, 0]
    >>> r_precision(r)
    0.5
    >>> r = [1, 0, 0]
    >>> r_precision(r)
    1.0

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])


# ## MAP with alpha

# In[5]:


# todo


# ## NDCG

# In[6]:


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)

    Relevance is positive real values.  Can use binary
    as the previous methods.

    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)

    Relevance is positive real values.  Can use binary
    as the previous methods.

    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


# ## Experiments

# In[31]:


def calculate_metrics(pred):
    query_indices = pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + '/evaluation/test_query_indices.p', 'rb'))
    df = pd.read_pickle(os.path.dirname(os.path.realpath(__file__)) + '/evaluation/test_set_1018.p')

    df['predictions'] = pred
    df.sort_values(['qid', 'predictions'], inplace=True, ascending=[True, False])
    labels = df.label.values
    r = [list(labels[i]) for i in query_indices]

    # one score
    map_ = mean_average_precision(r)
    # one score
    mrr = mean_reciprocal_rank(r)

    # for each query sepapately
    number_of_ranks = 10
    metrics_at_rank = np.zeros([2,number_of_ranks])
    for k in range(0,number_of_ranks):
        ndcg_at_ks = np.zeros(len(r))
        precision_at_ks = np.zeros(len(r))
        for i, rs in enumerate(r):
            ndcg_at_ = ndcg_at_k(rs, min(k+1, len(rs)), method=1)
            ndcg_at_ks[i] = ndcg_at_
            precision_at_ = precision_at_k(rs, min(k+1, len(rs)))
            precision_at_ks[i] = precision_at_
        metrics_at_rank[0, k] = ndcg_at_ks.mean()
        metrics_at_rank[1, k] = precision_at_ks.mean()
    return map_, mrr, metrics_at_rank

# only for dummy purposes
# df = pd.read_pickle('./evaluation/trigram_test_set_1018.p')
# df
# df = pd.read_pickle('./evaluation/trigram_test_set_1018_BM25_scores.p')
# scores = df.scores.values
# # dummy_p = [random.uniform(0, 1) for i in range(len(df))]
# # print(scores[:15])

# map_, mrr, metrics_at_rank = calculate_metrics(scores)
# pickle.dump(map_, open('bm25_map_.p', 'wb'))
# pickle.dump(mrr, open('mb25_mrr_.p', 'wb'))
# pickle.dump(metrics_at_rank[0, :], open('bm25_ndcg_k.p', 'wb'))
# pickle.dump(metrics_at_rank[1, :], open('bm25_precision_k.p', 'wb'))
# map_, mrr, metrics_at_rank


# In[36]:


def plot_at_k(m, name):
    # %%writefile /Users/jooppascha/Dropbox/projects/pandas_ipython/plot_template.py
    # %load /Users/jooppascha/Dropbox/projects/pandas_ipython/plot_template.py

    # data_types = ['sequential', 'diverging', 'qualitative']
    # data_type = 'diverging'
    # colormap = sns.choose_colorbrewer_palette(data_type=data_type, as_cmap=True)

    # https://seaborn.pydata.org/index.html#
    # plot type
    kinds = ['area', 'bar', 'barh', 'box', 'density', 'hexbin', 'hist', 'kde', 'line', 'pie', 'scatter']
    kind = 'line'

    # style
    style_overall = ['bmh','classic', 'dark_background', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright',
     'seaborn-colorblind','seaborn-dark-palette','seaborn-dark','seaborn-darkgrid','seaborn-deep',
     'seaborn-muted','seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 
     'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'seaborn']
    matplotlib.style.use('seaborn-whitegrid');

    # font
    plt.rcParams['font.family'] = 'Sans-serif'
    plt.rcParams['font.serif'] = 'Palatino'
    # plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    large_size = 12
    small_size = 8
    plt.rcParams['font.size'] = small_size
    plt.rcParams['axes.labelsize'] = large_size
    plt.rcParams['axes.labelweight'] = 'normal' #'bold'
    plt.rcParams['xtick.labelsize'] = large_size
    plt.rcParams['ytick.labelsize'] = large_size
    plt.rcParams['legend.fontsize'] = large_size
    plt.rcParams['figure.titlesize'] = large_size
    font_size = small_size

    fig, axes = plt.subplots(nrows=1, ncols=1)
    ax = axes

    # size
    aspect_ratio = 2/3
    scale = 1
    width, height = plt.figaspect(aspect_ratio)
    width = width*scale
    height = height*scale

    # matlab color palettes
    colormaps_perceptual_uniform = ['viridis', 'plasma', 'inferno', 'magma']
    colormaps_sequantial_1 = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    colormaps_sequantial_2 = ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper']
    diverging = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
    qualitative = ['Pastel1', 'Pastel2', 'Paired', 'Accent','Dark2', 'Set1', 'Set2', 'Set3','tab10', 'tab20', 'tab20b', 'tab20c']
    miscellaneous = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv', 'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

    # seaborn color palettes
    desat = 1
    n_colors = 4
    lightness = 0.3

    sns_categorical = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']
    cicular_color_systems = ['hls', 'husl']
    sns_colors_mixed = ['hls', 'Set1']
    sns_colors_two = ['RdBu']
    sns_colors_base = ['Blues_d']
    palette = 'muted'
    sns_color = sns.color_palette(palette=palette, n_colors=n_colors, desat=desat).as_hex()
    # sns_color = sns.hls_palette(n_colors, l=lightness, s=desat)
    colormap = ListedColormap(sns_color) #sns.cubehelix_palette(8, as_cmap=True)

    rot_xlabel = 0
    is_horizontal = True
    title = None
    xlabel = 'Cut-off Point: k'
    ylabel = 'Metric Score'
    xlims = [1, 10]
    ylims = [0, 1]

    # data settings, take log?
    logx = False
    logy = False

    # subplots
    subplots = False
    sharex = False
    sharey = False
    sort_columns = False
    use_index = False

    # show std or average
    yerr = None # gp3.std() 

    # bars
    space_between_bars = 0.7
    is_stacked = False
    linewidth = 40

    # lines and grids
    line_styles = ['-', '--', '-.', ':']
    line_style = {0:'-'}
    xticks = np.arange(1,11,1)
    yticks = np.arange(0,1.1,0.1)
    grid = True

#     pickle.dump(map_, open('bm25_map_.p', 'wb'))
#     pickle.dump(mrr, open('mb25_mrr_.p', 'wb'))
    ndcg_k_random = pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + '/evaluation/ndcg_k_random.p', 'rb'))
    precision_k_random = pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + '/evaluation/precision_k_random.p', 'rb'))
    m_random = np.array([ndcg_k_random, precision_k_random])
    
    ndcg_k_bm25 = pickle.load(open('bm25_ndcg_k.p', 'rb'))
    precision_k_bm25 = pickle.load(open('bm25_precision_k.p', 'rb'))
    m_bm25 = np.array([ndcg_k_bm25, precision_k_bm25])

    data = np.concatenate([m.T, m_random.T, m_bm25.T], axis=1)
    print(data)
#     data = data[:, [0,2,1,3]]
    df = pd.DataFrame(data, columns=['nDCG@k', 'nDCG@k baseline', 'precision@k', 'precision@k baseline', 'blha', 'blah'])
    df.index = range(1,len(df)+1)
    # actual plot
    ax = ax=df.plot(ax=ax, 
                 cmap=colormap, 
                 figsize=[width, height], 
                 fontsize=font_size, 
                 grid=grid, 
                 kind=kind, 
                 rot=rot_xlabel, 
                 title=title,
                 xlim=xlims, 
                 ylim=ylims, 
                 logx=logx, 
                 logy=logy, 
                 sharex=sharex, 
                 sharey=sharey, 
                 sort_columns=sort_columns, 
                 yerr=yerr, 
    #              style=line_style, 
    #              subplots=subplots, 
                 xticks=xticks, 
                 yticks=yticks, 
    #              use_index=use_index, 
    #              width=space_between_bars, 
    #              stacked=is_stacked, 
    #              linewidth=linewidth
                )
    fig = plt.gcf()
    fig.tight_layout()
    none = plt.xlabel(xlabel, labelpad=None)
    none = plt.ylabel(ylabel, labelpad=None)

    # legend handling
    # handles, labels = ax.get_legend_handles_labels()
    # locs = ['best', 'upper right', 'upper left', 'lower left', 
    #         'lower right', 'right', 'center left', 'center right', 
    #         'lower center', 'upper center', 'center']
    # loc = 'upper right'

    # legend = ax.legend(loc=loc, # bbox_to_anchor=(0.5, 1.05)
    #           ncol=1, fancybox=False, shadow=False, frameon=True, labelspacing=0.1, borderpad=0.25, handlelength=0.5, handletextpad=0.25, columnspacing=1)
    # frame = legend.get_frame()
    # transparant_ratio = 0.95
    # frame.set_alpha(transparant_ratio)
    # legend_border_color = 'white'
    # legend_color = 'white'
    # frame.set_edgecolor(legend_border_color)
    # frame.set_facecolor(legend_color)
    # frame.set_linewidth(1)
    # for legobj in legend.legendHandles:
    #     legobj.set_linewidth(3.0)


    # figure background color
    # background_color = '#EFF0F1'
    # background_color = '#FFFFFF'
    # ax.set_facecolor(background_color)

    # add line
    # plt.axhline(1, linestyle='--', linewidth=1, color='k')
    # plt.axvline(1, linestyle='--', linewidth=1, color='k')

    # arrows
    # markers : https://matplotlib.org/api/markers_api.html
    # arrows : https://matplotlib.org/users/annotations_guide.html
    # arrow_text = 'local_max'
    # arrow_go_to = [0.1, 0.1]
    # arrow_from = [0.5, 1.5]
    # plt.plot(arrow_go_to[0], arrow_go_to[1], 'ro')
    # ax.annotate(arrow_text, xy=arrow_go_to, xytext=arrow_from, 
    #             size=15, xycoords='data', textcoords='data', 
    #             va="center", ha="center", 
    #             bbox=dict(boxstyle="round4", pad=0.3, fc="w", 
    #                       linewidth=1, color='black'), 
    #             arrowprops=dict(arrowstyle='->', 
    #                             facecolor='black', 
    #                             connectionstyle="arc3,rad=0", 
    #                             linewidth=1, color='black'))


    # ticks
    plt.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.15)
    ax.grid(which='minor', linewidth='0.5', color='black', alpha=0.05)
    ax.grid(True, which='both')

    # xmin, xmax = ax.get_xlim()
    # none = plt.xticks(np.arange(round(xmin,1), xmax, 1))
    # none = ax.set_xticks(np.arange(round(xmin,1), xmax, 1), minor=True)   

    # ymin, ymax = ax.get_ylim()
    # none = plt.yticks(np.arange(round(ymin), ymax, 100)) 
    # none = ax.set_yticks(np.arange(round(ymin), ymax, 20), minor=True)     
    # plt.minorticks_off()
    # ax.tick_params(axis='both', which='both',length=5)

    # plt.tick_params(
    #     axis='x',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     bottom='off',      # ticks along the bottom edge are off
    #     top='off',         # ticks along the top edge are off
    #     labelbottom='on') # labels along the bottom edge are off

    # ml = MultipleLocator(0.02)
    # ax.xaxis.set_minor_locator(ml)
    # ax.xaxis.grid(which="major", color='k', linestyle='-.', linewidth=0)
    # ax.xaxis.grid(which="minor", color='k', linestyle='-.', linewidth=0)

    # sns.despine()
    # add labels
    # for p in ax.patches:
    #     label = str(int(p.get_height() * 100))
    #     offset_x = 0
    #     if len(label) == 1:
    #         offset_x = -0.025
    #     else:
    #         offset_x = 0.005
    #     nothing = ax.annotate(label, (p.get_x()-offset_x, 
    #                                   p.get_height() + 0.01), 
    #                           fontsize=7);    

    # saving
    fig.savefig(name +'.pdf', bbox_inches='tight', pad_inches=0.01)
    # fig.savefig(name + '.png', format='png', dpi=1000, bbox_inches='tight', pad_inches=0.01)
    # query_indices
    # plot_at_k(metrics_at_rank)
    # plot_map_mrr(map_, mrr)


# In[ ]:


def plot_map_mrr(map_, mrr, name):
    # data_types = ['sequential', 'diverging', 'qualitative']
    # data_type = 'diverging'
    # colormap = sns.choose_colorbrewer_palette(data_type=data_type, as_cmap=True)

    # https://seaborn.pydata.org/index.html#
    # plot type
    kinds = ['area', 'bar', 'barh', 'box', 'density', 'hexbin', 'hist', 'kde', 'line', 'pie', 'scatter']
    kind = 'bar'

    # style
    style_overall = ['bmh','classic', 'dark_background', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright',
     'seaborn-colorblind','seaborn-dark-palette','seaborn-dark','seaborn-darkgrid','seaborn-deep',
     'seaborn-muted','seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 
     'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'seaborn']
    matplotlib.style.use('seaborn-whitegrid');

    # font
    plt.rcParams['font.family'] = 'Sans-serif'
    plt.rcParams['font.serif'] = 'Palatino'
    # plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    large_size = 12
    small_size = 8
    plt.rcParams['font.size'] = small_size
    plt.rcParams['axes.labelsize'] = large_size
    plt.rcParams['axes.labelweight'] = 'normal' #'bold'
    plt.rcParams['xtick.labelsize'] = large_size
    plt.rcParams['ytick.labelsize'] = large_size
    plt.rcParams['legend.fontsize'] = large_size
    plt.rcParams['figure.titlesize'] = large_size
    font_size = small_size

    fig, axes = plt.subplots(nrows=1, ncols=1)
    ax = axes

    # size
    aspect_ratio = 3/3
    scale = 1
    width, height = plt.figaspect(aspect_ratio)
    width = width*scale
    height = height*scale

    # matlab color palettes
    colormaps_perceptual_uniform = ['viridis', 'plasma', 'inferno', 'magma']
    colormaps_sequantial_1 = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    colormaps_sequantial_2 = ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper']
    diverging = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
    qualitative = ['Pastel1', 'Pastel2', 'Paired', 'Accent','Dark2', 'Set1', 'Set2', 'Set3','tab10', 'tab20', 'tab20b', 'tab20c']
    miscellaneous = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv', 'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

    # seaborn color palettes
    desat = 1
    n_colors = 4
    lightness = 0.3

    sns_categorical = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']
    cicular_color_systems = ['hls', 'husl']
    sns_colors_mixed = ['hls', 'Set1']
    sns_colors_two = ['RdBu']
    sns_colors_base = ['Blues_d']
    palette = 'muted'
    sns_color = sns.color_palette(palette=palette, n_colors=n_colors, desat=desat).as_hex()
    # sns_color = sns.hls_palette(n_colors, l=lightness, s=desat)
    colormap = ListedColormap(sns_color) #sns.cubehelix_palette(8, as_cmap=True)

    rot_xlabel = 0
    is_horizontal = True
    title = None
    xlabel = 'Cut-off Point: k'
    ylabel = 'Metric Score'
    xlims = [1, 10]
    ylims = [0, 1]

    # data settings, take log?
    logx = False
    logy = False

    # subplots
    subplots = False
    sharex = False
    sharey = False
    sort_columns = False
    use_index = False

    # show std or average
    yerr = None # gp3.std() 

    # bars
    space_between_bars = 0.7
    is_stacked = False
    linewidth = 40

    # lines and grids
    line_styles = ['-', '--', '-.', ':']
    line_style = {0:'-'}
    xticks = np.arange(1,11,1)
    yticks = np.arange(0,1.1,0.1)
    grid = True

    mrr_random = pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + '/evaluation/mrr_random.p', 'rb'))
    map_random = pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + '/evaluation/map_random.p', 'rb'))
    data = np.array([[map_random, mrr_random],[map_, mrr]])
    df = pd.DataFrame(data)
    df.columns = ['map', 'mrr']
    df.index = ['random (baseline)', 'sLSTM']
    # actual plot
    ax=df.plot(ax=ax, 
                 cmap=colormap, 
                 figsize=[width, height], 
                 fontsize=font_size, 
                 grid=grid, 
                 kind=kind, 
                 rot=rot_xlabel, 
                 title=title,
                 xlim=xlims, 
                 ylim=ylims, 
                 logx=logx, 
                 logy=logy, 
                 sharex=sharex, 
                 sharey=sharey, 
                 sort_columns=sort_columns, 
                 yerr=yerr, 
    #              style=line_style, 
    #              subplots=subplots, 
                 xticks=xticks, 
                 yticks=yticks, 
    #              use_index=use_index, 
                 width=space_between_bars, 
    #              stacked=is_stacked, 
    #              linewidth=linewidth
                )
    fig = plt.gcf()
    fig.tight_layout()
    none = plt.xlabel(xlabel, labelpad=None)
    none = plt.ylabel(ylabel, labelpad=None)

    # legend handling
    # handles, labels = ax.get_legend_handles_labels()
    # locs = ['best', 'upper right', 'upper left', 'lower left', 
    #         'lower right', 'right', 'center left', 'center right', 
    #         'lower center', 'upper center', 'center']
    # loc = 'upper right'

    # legend = ax.legend(loc=loc, # bbox_to_anchor=(0.5, 1.05)
    #           ncol=1, fancybox=False, shadow=False, frameon=True, labelspacing=0.1, borderpad=0.25, handlelength=0.5, handletextpad=0.25, columnspacing=1)
    # frame = legend.get_frame()
    # transparant_ratio = 0.95
    # frame.set_alpha(transparant_ratio)
    # legend_border_color = 'white'
    # legend_color = 'white'
    # frame.set_edgecolor(legend_border_color)
    # frame.set_facecolor(legend_color)
    # frame.set_linewidth(1)
    # for legobj in legend.legendHandles:
    #     legobj.set_linewidth(3.0)


    # figure background color
    # background_color = '#EFF0F1'
    # background_color = '#FFFFFF'
    # ax.set_facecolor(background_color)

    # add line
    # plt.axhline(1, linestyle='--', linewidth=1, color='k')
    # plt.axvline(1, linestyle='--', linewidth=1, color='k')

    # arrows
    # markers : https://matplotlib.org/api/markers_api.html
    # arrows : https://matplotlib.org/users/annotations_guide.html
    # arrow_text = 'local_max'
    # arrow_go_to = [0.1, 0.1]
    # arrow_from = [0.5, 1.5]
    # plt.plot(arrow_go_to[0], arrow_go_to[1], 'ro')
    # ax.annotate(arrow_text, xy=arrow_go_to, xytext=arrow_from, 
    #             size=15, xycoords='data', textcoords='data', 
    #             va="center", ha="center", 
    #             bbox=dict(boxstyle="round4", pad=0.3, fc="w", 
    #                       linewidth=1, color='black'), 
    #             arrowprops=dict(arrowstyle='->', 
    #                             facecolor='black', 
    #                             connectionstyle="arc3,rad=0", 
    #                             linewidth=1, color='black'))


    # ticks
    plt.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.15)
    ax.grid(which='minor', linewidth='0.5', color='black', alpha=0.05)
    ax.grid(True, which='both')

    # xmin, xmax = ax.get_xlim()
    # none = plt.xticks(np.arange(round(xmin,1), xmax, 1))
    # none = ax.set_xticks(np.arange(round(xmin,1), xmax, 1), minor=True)   

    # ymin, ymax = ax.get_ylim()
    # none = plt.yticks(np.arange(round(ymin), ymax, 100)) 
    # none = ax.set_yticks(np.arange(round(ymin), ymax, 20), minor=True)     
    # plt.minorticks_off()
    # ax.tick_params(axis='both', which='both',length=5)

    # plt.tick_params(
    #     axis='x',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     bottom='off',      # ticks along the bottom edge are off
    #     top='off',         # ticks along the top edge are off
    #     labelbottom='on') # labels along the bottom edge are off

    # ml = MultipleLocator(0.02)
    # ax.xaxis.set_minor_locator(ml)
    # ax.xaxis.grid(which="major", color='k', linestyle='-.', linewidth=0)
    # ax.xaxis.grid(which="minor", color='k', linestyle='-.', linewidth=0)

    # sns.despine()
    # add labels
    # for p in ax.patches:
    #     label = str(int(p.get_height() * 100))
    #     offset_x = 0
    #     if len(label) == 1:
    #         offset_x = -0.025
    #     else:
    #         offset_x = 0.005
    #     nothing = ax.annotate(label, (p.get_x()-offset_x, 
    #                                   p.get_height() + 0.01), 
    #                           fontsize=7);    

    # saving
    fig.savefig(name +'.pdf', bbox_inches='tight', pad_inches=0.01)


# In[37]:

# calculate_metrics
# plot_at_k(metrics_at_rank, 'metrics_at_rank_test')
# plot_map_mrr(map_, mrr, 'map_mrr_test')


# # Quantitative Analysis
