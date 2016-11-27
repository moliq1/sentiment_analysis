#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import numpy as np
import gensim
#from random import shuffle
from gensim.models.doc2vec import Doc2Vec,LabeledSentence
from sklearn.cross_validation import train_test_split

def get_dataset(write=True):
    files = os.listdir('.')
    neg_files = [neg for neg in files if neg.endswith("neg.txt")]
    pos_files = [pos for pos in files if pos.endswith("pos.txt")]
    neg_reviews = []
    pos_reviews = []
    for i in neg_files:
        print 'read data from --', i
        with open(i, "r") as f:
            neg_reviews += (line.decode('utf-8') for line in f.readlines())
            
    for j in pos_files:
        print 'read data from --', j
        with open(j, "r") as f:
            pos_reviews += (line.decode('utf-8') for line in f.readlines())
    
    print 'done!!!'
    print 'neg_reviews: ', len(neg_reviews)
    print 'pos_reviews: ', len(pos_reviews)
    #print neg_reviews[0]
    #y = np.concatenate((np.ones(8000), np.zeros(8000)), axis=0)
    #x_train, x_test, y_train, y_test = train_test_split(matrix, y, test_size=0.2)
    if(write):
        with open('pos_reviews.txt', 'w') as f:
            for line in pos_reviews:
                f.write(line.encode('utf-8') + '\n')
        with open('neg_reviews.txt', 'w') as f:
            for line in neg_reviews:
                f.write(line.encode('utf-8') + '\n')
    
    y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))
    # 将数据分割为训练与测试集
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)

    # 数据清洗预处理，中文根据需要进行修改
    def cleanText(corpus):
        punctuation = """.,?!:;(){}[]，。！？：“‘（）【】；""".decode('utf-8')
        corpus = [z.replace('\n','') for z in corpus]
        corpus = [z.replace('<br />', ' ') for z in corpus]

        #treat punctuation as individual words
        for c in punctuation:
            corpus = [z.replace(c, ' %s '%c) for z in corpus]
        corpus = [z.split() for z in corpus]
        return corpus

    x_train = cleanText(x_train)
    x_test = cleanText(x_test)
    
    #Gensim的Doc2Vec应用于训练要求每一篇文章/句子有一个唯一标识的label.
    #我们使用Gensim自带的LabeledSentence方法. 标识的格式为"TRAIN_i"和"TEST_i"，其中i为序号
    def labelizeReviews(reviews, label_type):
        labelized = []
        for i,v in enumerate(reviews):
            label = '%s_%s'%(label_type,i)
            labelized.append(LabeledSentence(v, [label]))
        return labelized
    
    print 'labeling documents .......'
    x_train = labelizeReviews(x_train, 'TRAIN')
    x_test = labelizeReviews(x_test, 'TEST')

    return x_train,x_test,y_train, y_test

def build_model(x_train, x_test, iteration =5, save=True):
    if(save):
        big_list = x_train + x_test
        model = Doc2Vec(min_count=2, window=10, size=100, sample=1e-4, negative=5, workers=8)
        model.build_vocab(big_list)
	for i in range(iteration):
            model.train(big_list)
	print 'saving model to file.....'  
        model.save('./sentim.d2v')
    else:
	print 'loading model from file.....'
	model = Doc2Vec.load('./sentim.d2v')
    return model

def get_vectors(model, x_train, x_test):
    print 'getting vectors .........'
    
    
    #infer from model
    xtrain = []
    xtest = []
    for doc_id in range(len(x_train)):
        inferred_vector = model.infer_vector(x_train[doc_id].words)
        xtrain.append(inferred_vector)
    for doc_id in range(len(x_test)):
        inferred_vector = model.infer_vector(x_test[doc_id].words)
        xtest.append(inferred_vector)
    return np.array(xtrain), np.array(xtest)
    '''
    train_tags = []
    test_tags = []
    for i in range(len(x_train)):
        train_tags += x_train[i].tags
        
    for i in range(len(x_test)):
        test_tags += x_test[i].tags
        
    x_train = [model.docvecs[tag] for tag in train_tags]
    x_test = [model.docvecs[tag] for tag in test_tags]
    return np.array(x_train), np.array(x_test)
    '''  

def Classifier(x_train,y_train,x_test, y_test):
    #使用sklearn的SGD分类器
    print 'building model ........'
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(penalty='l2', solver='lbfgs')
    lr.fit(x_train, y_train)

    print 'Test Accuracy: %.2f'%lr.score(x_test, y_test)

    return lr

def ROC_curve(lr,y_test):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    pred_probas = lr.predict_proba(x_test)[:,1]

    fpr,tpr,_ = roc_curve(y_test, pred_probas)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.show()

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_dataset(write=False)
    model = build_model(x_train, x_test, save=False)
    x_train, x_test = get_vectors(model, x_train, x_test)
    lr = Classifier(x_train, y_train, x_test, y_test)
    ROC_curve(lr, y_test)
