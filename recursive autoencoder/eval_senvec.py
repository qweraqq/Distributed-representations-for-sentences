# -*- coding: utf-8 -*-
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load dataset to tokenize
W = np.loadtxt('sen_vec')

W_norm = np.zeros(W.shape)
d = (np.sum(W ** 2, 1) ** (0.5))
W_norm = (W.T / d).T


def show_nearest_words(i,n=10):
    v =  W_norm[i, :]
    distances = np.dot(v,W_norm.T)
    orders = np.argsort(-distances)
    for i in range(0,n):
        if distances[orders[1]] > 0.82 and distances[orders[1]] < 0.9:
            print orders[i]+1, distances[orders[i]]

for i in range(2000,3000):
    show_nearest_words(i,3)

