import pickle
import argparse
import numpy as np

from vocab import Vocabulary
from gensim.models import KeyedVectors


def main(opt):
    vocab = pickle.load(open(opt.vocab_path, 'rb'))
    num = len(vocab)
    print (num)
    model = KeyedVectors.load_word2vec_format(opt.embed_weight, binary=True)
    

    matrix_len = num
    weights_matrix = np.zeros((num, 300))
    words_found = 0 
    mask = np.zeros(num, dtype=int)

    for i, word in enumerate(vocab.idx2word):
        try: 
            weights_matrix[i] = model[vocab.idx2word[i]]
            words_found += 1
            mask[i] = 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.1, size=(300, ))

    print (words_found)

    np.save("./embed/f30kword2vec300dim_3.npy", weights_matrix)
    np.save("./embed/f30kword2vecmask_3.npy", mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', default='')
    parser.add_argument('--embed_weight', default='')
    opt = parser.parse_args()
    main(opt)