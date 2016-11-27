#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import gensim
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

def read_data(filename):
    data = {}
    files = os.listdir(filename)
    neg_files = [neg for neg in files if neg.endswith("neg.txt")]
    pos_files = [pos for pos in files if pos.endswith("pos.txt")]
    neg_sents = []
    pos_sents = []
    for i in neg_files:       
        with open(i, "r") as f:
            neg_sents += f.readlines()
    for j in pos_files:
        with open(j, "r") as f:
            pos_sents += f.readlines()
    data['neg'] = neg_sents
    data['pos'] = pos_sents
    return data

def remove_stopwords(stopwords_path, data):
    with open(stopwords_path,'r') as file:  
        stop_list = set([line.strip() for line in file])  
    data['neg'] = [[word for word in document.split() if word not in stop_list] for document in data['neg']]
    data['pos'] = [[word for word in document.split() if word not in stop_list] for document in data['pos']]
    return data

def vectorize(data):
    big_data = data['neg'] + data['pos']
    dictionary = gensim.corpora.Dictionary(big_data)
    corpus = [dictionary.doc2bow(text) for text in big_data]
    #matrix = gensim.matutils.corpus2dense(corpus, num_terms=len(dictionary))
    matrix = gensim.matutils.corpus2csc(corpus)
    return matrix.T

def build_model(matrix):
    print "shape of the matrix:  ", matrix.shape
    y = np.concatenate((np.ones(8000), np.zeros(8000)), axis=0)
    x_train, x_test, y_train, y_test = train_test_split(matrix, y, test_size=0.2)
    print "training set:  ", x_train.shape
    print "test set:  ", x_test.shape
    model = LogisticRegression(penalty='l1')
    model.fit(x_train, y_train)
    print 'Test Accuracy: %.3f'%model.score(x_test, y_test)

if __name__ == '__main__':
    data = read_data(".")
    data = remove_stopwords('/home/zack/stop_list_li.txt', data)
    matrix = vectorize(data)
    pred = build_model(matrix)
