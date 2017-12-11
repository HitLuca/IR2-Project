"""
This is a script to extract the rows of embedding matrix according to a given vocabulary
input:
1) embedding vocabulary npy
2) embedding matrix npy
3) vocabulary of interest pickle

output:
1) re-ordered vocabulary txt (intersection of the vocab in embedding matrix and given vocab)
2) part of embedding matrix of interest npy
"""

import smart_open as so
import sys
import numpy as np
import time
import pickle

# path of original embedding matrix and its vocabulary
vocabulary = np.load("/media/henglin/data/ir2_tmp/embedding_vocabulary.npy")
embedding_matrix = np.load("/media/henglin/data/ir2_tmp/embedding_matrix.npy")

# path of the vocabulary of interest
test_vocab_path = "../reproduction/lib/data/testset_vocab.txt"

# load the vocab of interest into numpy array
with open(test_vocab_path, 'r') as f:
    yahoo_vocab = np.array([word.rstrip() for word in f])

print("Vocabulary size: {}".format(len(vocabulary)))
print("Yahoo Vocabulary size: {}".format(len(yahoo_vocab)))

mask = np.in1d(vocabulary, yahoo_vocab)     # find the overlap of vocab
idx = np.where(mask == True)                # find the index of overlapped vocab

partial_embedding_matrix = embedding_matrix[idx[0], :]  # select the part from original embedding matrix
print(partial_embedding_matrix.shape)

# ADD EXTRA 2 ROWS (padding and unknown token)
r2 = np.zeros(shape=[2, partial_embedding_matrix.shape[1]])
partial_embedding_matrix = np.concatenate([partial_embedding_matrix, r2], axis=0)

np.save("testset_partial_embedding_matrix", partial_embedding_matrix)

# select the overlapped vocabulary
selected_vocab = vocabulary[idx[0]]

with open("vocabulary.txt", "w") as f:
    for word in selected_vocab:
        f.write(word+"\n")
    f.write('<PAD>'+'\n')
    f.write('<UNK>'+'\n')
    f.flush()
