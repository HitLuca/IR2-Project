"""
This is a script to convert original lexvec into two npy matrix
1) vocabulary
2) embedding matrix
"""

import smart_open as so
import numpy as np
import time

# path of the original word vector from lexvec
embedding_path = "./embedding/lexvec.commoncrawl.300d.W.pos.vectors.gz"

vocabulary = list()
word_matrix = np.zeros(shape=[2000000, 300], dtype=np.float32)

start = time.time()
for i, line in enumerate(so.smart_open(embedding_path)):
    if i == 0:
        continue

    tokens = line.split()

    word = tokens[0].decode('UTF-8')
    vocabulary.append(word)

    word_vec = np.array(list(map(float, tokens[1:])))
    word_matrix[i-1, :] = word_vec

    if i % 10000 == 0:
        print("Entries: {}, time elapsed per 10000 entries: {}".format(i, time.time() - start))
        start = time.time()


print("saving embedding vocabulary")
np.save("embedding_vocabulary", vocabulary)
print("Completed!")
print("saving embedding matrix")
np.save('embedding_matrix', word_matrix)
print("Completed!")
