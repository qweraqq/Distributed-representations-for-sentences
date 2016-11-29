# -*- coding: utf-8 -*-
from __future__ import division
import jieba
import logging
import numpy as np
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def load_rae_parameters(w1_filename, w2_filename, b_filename):
    """
    load trained parameters from txt files
    :param w1_filename:
    :param w2_filename:
    :param b_filename:
    :return: np.matrix W1,W2,b
    """
    w1_tmp = np.loadtxt(w1_filename)
    w2_tmp = np.loadtxt(w2_filename)
    b_tmp = np.loadtxt(b_filename)
    return w1_tmp, w2_tmp, b_tmp


def str_to_vector(tokenized_sentence, w1, w2, b):
    """
    convert a news_headline to distributed representation
    (just like word2vec)

    Input: a tokenized string, actually a matrix of float
           size(matirx) = sentence length, word embeddings dim
           you can use tokenizeSentence() API to tokenize
    :param tokenized_sentence:
    :param w1:
    :param w2:
    :param b:
    :return:
    """
    vector_dim = tokenized_sentence.shape[1]
    h = np.zeros(vector_dim)  # init hidden
    sentence_len = tokenized_str.shape[0]
    for x in range(sentence_len):
        h = np.tanh((1/(x+1))*np.dot(tokenized_sentence[x, :], w1) +
                    (x/(x+1))*np.dot(h, w2))
        h = h/(np.sum(h**2)**0.5)  # normalization
        if np.isnan(h[0]) or np.isnan(h[1]):  # special case
            h = np.zeros(vector_dim)
    return h


def get_reconstruction_error(h1, n1, h2, n2, h, w1, w2, b):
    """
    :param h1:
    :param n1:
    :param h2:
    :param n2:
    :param h:
    :param w1:
    :param w2:
    :param b:
    :return:
    """


def load_word_embeddings(vocab_file, vectors_file):
    with open(vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]

    with open(vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]
            # vectors['中国']=[0.5,0.4,...0.1]
    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}
    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v
    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    return W_norm, vocab, ivocab


def get_word_distance(w1, w2, W_norm, vocab):
    """
    cosine distance between two words w1&w2
    """
    v1 = W_norm[vocab[w1], :]
    v2 = W_norm[vocab[w2], :]
    dist = np.dot(v1, v2.T)
    return dist


def show_nearest_words(w, W_norm, vocab, ivocab, n=10):
    """
    print top n(default 10) nearest words to word w
    """
    v = W_norm[vocab[w], :]
    distances = np.dot(v, W_norm.T)
    orders = np.argsort(-distances)
    for i in range(n):
        print i, ivocab[orders[i]], distances[orders[i]]
        # print i, ivocab[orders[i]].decode('utf8').encode('gbk'), distances[orders[i]]


def tokenize_sentence(line, W_norm, vocab, tmp_lambda=0.8):
    """
    convert a sentence to a matrix && Chinese word seg revision
    matrix size = sentence length, word embeddings dim

    Input: line is the sentence(type string) to tokenize
           tmp_lambda(type float) is a word seg revision parameter
    """
    vector_dim = W_norm.shape[1]
    rvalue = np.zeros((1, vector_dim))
    sen = list(jieba.cut(line))
    sen_cut = []
    wjm1 = "tmp"
    for j, w in enumerate(sen):
        w = w.encode('utf-8')
        if w not in vocab:
            if j >= 1:
                sen_cut.append(wjm1)
                rvalue = np.vstack((rvalue, Wjm1))
            Wjm1 = np.zeros((1, vector_dim))  # word 'unk'=[0]
            wjm1 = w
            continue

        if j == 0:
            Wjm1 = np.array(W_norm[vocab[w], :])
            wjm1 = w
            continue

        Wj = np.array(W_norm[vocab[w], :])
        if np.dot(Wjm1, Wj) > tmp_lambda:  # merge word
            np.add(Wj, Wjm1, Wjm1)
            Wjm1 = (Wjm1.T / (np.sum(Wjm1 ** 2) ** 0.5)).T
            wjm1 = wjm1 + w
        else:
            sen_cut.append(wjm1)
            rvalue = np.vstack((rvalue, Wjm1))
            Wjm1 = Wj
            wjm1 = w

        if j == len(sen)-1:  # end
            rvalue = np.vstack((rvalue, Wjm1))
            sen_cut.append(wjm1)

    return np.delete(rvalue, 0, 0), sen_cut


if __name__ == '__main__':
    # load dataset to tokenize
    # data_file = 'ths_news_2015.txt'
    # load word embeddings
    vocab_file = 'jieba_ths_vocab_big.txt'
    vectors_file = 'jieba_ths_vectors_big.txt'
    W_norm, vocab, ivocab = load_word_embeddings(vocab_file, vectors_file)

    tokenized_str, sen_cut = tokenize_sentence("花好“纳税人”的每一分钱——我国深化财税体制改革综述", W_norm, vocab, 0.6)
    for x in sen_cut:
        print x.decode('utf-8') + ' ',
    print
    for x in list(jieba.cut("花好“纳税人”的每一分钱——我国深化财税体制改革综述")):
        print x.encode('utf-8') + ' ',
    print

