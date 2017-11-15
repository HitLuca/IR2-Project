
import pickle
from collections import Counter
from collections import defaultdict
import numpy as np

# TO DO: remove punctuations

class BM25 :
    '''
    BM25 
    pickle_file format is the dict thingie with the questions as strings
    '''
    def __init__(self, pickle_file = None, name = '', df_pickle = None) :
        if pickle_file:
            file = open(pickle_file, 'rb')
            data = pickle.load(file)
            self.data = data
            self.df = self.get_df(data = data ,  name = name )
        elif df_pickle: 
            file = open(df_pickle, 'rb')
            self.df = pickle.load(file)
        else:
            print('No df model specified')
            []

    def get_df(self, data, name = 'df', key1 = 'question1', key2 = 'question2'):
        mean_doc_lenght = 0
        tf = defaultdict() 
        keys = [key1, key2]
        for pair in data:
            for key in keys:
                # all terms in the quetions
                terms  = [str(w) for w in pair[key].split()]
                mean_doc_lenght += len(terms) 
                for q in set(terms):
                    # if term is in dict
                    if q in tf:
                        tf[q] += 1
                    else:
                        tf[q] = 1
        file = [tf, round(mean_doc_lenght / float(len(data)*2),4), len(data)]            
        pickle.dump( file, open( name +".p", "wb" ))
        return file
    
    def score(self, Q, A, k = 1.2, b = 0.75):

        score = 0
        mean_doc_lenght = self.df[1]
        q_terms = Q.split()
        a_terms = A.split()
        # count the document terms
        counts = Counter(a_terms)
        # check if the query term is in the document
        for q in q_terms:
            if q in counts:
                # compute BM25 term frequencies
                tf = (counts[q]*(1+k)) / (counts[q]+ k* ( (1-b) + b* len(a_terms) / mean_doc_lenght))
                # compute BM25 inverse document frequencies
                if q in self.df[0]:
                    df = self.df[0][q]
                    idf = np.log((self.df[2] - df + 0.5) / (df + 0.5))
                else:
                    idf = np.log((len(a_terms)  + 0.5) / 0.5)
                score += tf*idf
        return score / float(len(q_terms))


bm = BM25(pickle_file = 'quora_duplicate_questions.pk1', name = 'quora_df')
bm = BM25(df_pickle = 'quora_df.p')
print(bm.score('cow', 'cow horse  cat'))


