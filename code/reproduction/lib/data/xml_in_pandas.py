import gzip
import shutil
import pickle
from bs4 import BeautifulSoup
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import re
import smart_open
import numpy as np
import pandas as pd

NAME_OF_INPUT_FILE = 'FullOct2007.xml.gz'

def remove_tag(snippit):
    snippit = snippit.replace('<br />', '')
    return snippit

def check(snippit):
    if snippit:
        return snippit.text
    else:
        return('')

def parse(xml):

    new_q = []
    soup = BeautifulSoup(xml, 'html.parser')
    # double check 
    q = soup.find('vespaadd')
    # all tags
    language = check(q.find('language'))
    topic = q.find('document').get('type')
    question = remove_tag(q.find('subject').text)
    uri = check(q.find('uri'))
    explenation = remove_tag(check(q.find('content')))
    best_answer = remove_tag(check(q.find('bestanswer')))
    cat = check(q.find('cat'))
    subcat = check(q.find('maincat'))
    maincat = check(q.find('maincat'))
    
    return [uri, question, explenation, best_answer, cat, maincat, subcat, topic, language]


def sample_positive_examples(name_input_file):
    pandas_rows = []
    iteration = 0
    counter = 0
    save_step = 50000
    print_step = 10000

    cols = ['uri', 'subject', 'content', 'bestanswer', 'cat', 'maincat', 'subcat', 'document_type', 'language']
    data_stream = []
    with smart_open.smart_open(name_input_file) as fin:
        for line in fin:
            data_stream.append(str(line))
            if '</vespaadd>' in str(line):
                s = ''.join(data_stream)
                row = parse(s)
                pandas_rows.append(row)
                counter += 1
                if counter % print_step == 0:
                    print('counter',counter)
                if counter % save_step == 0:
                    iteration += 1
                    counter = 0
                    print('N_rows per dataframe:',len(pandas_rows))
                    df = pd.DataFrame(data=pandas_rows, columns=cols)
                    save_name = 'panda_pickles/part_'+str(iteration)+'.p'
                    df.to_pickle(save_name)
                    pandas_rows = []
                data_stream = []

    print('N_rows per dataframe:',len(pandas_rows))
    df = pd.DataFrame(data=pandas_rows, columns=cols)
    save_name = 'panda_files/part_'+str(iteration)+'.p'
    df.to_pickle(save_name)
          
def generate_sample_indexes(N_in, N_out):
    indexes = np.random.random_integers(0, N_in, N_out)
    indexes_binary = [1 if x in indexes else 0 for x in range(N_in)]
    return indexes_binary


sample_positive_examples(NAME_OF_INPUT_FILE)

