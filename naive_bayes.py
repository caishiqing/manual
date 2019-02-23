# -*- coding: utf-8 -*-
"""
标题：朴素贝叶斯
作者：蔡世清
说明：实现高斯分布和多项分布两种类型，多项分布需要对连续型数据进行离散化，
      参见决策树对连续型属性动态规划搜索最优多级划分。
最后修订日：2019-02-18
"""
from __future__ import print_function,unicode_literals
import numpy as np
from tools import *
from tools.quantilize import Discretor
from math import pi
from collections import Counter

epsilon = 1e-15

class Static:
    """统计分布类基类"""
    def predict_log_proba(self, x):
        p = self.predict_proba(x)
        log_p = np.log(p + epsilon)
        return log_p

    def joint_log_proba(self, x):
        log_p = self.predict_log_proba(x)
        return np.sum(log_p, axis=1)

class Normal(Static):
    """正态分布类"""
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def fit(self, x, axis=0):  #对每个维度独立统计参数
        self.mean = np.mean(x, axis=axis, keepdims=True)
        self.std = np.std(x, axis=axis, keepdims=True)

    def predict_proba(self, x):
        a = 1 / (np.sqrt(2*pi) * self.std)
        p = np.exp(-(x-self.mean)**2 / (2 * self.std**2))
        return a * p

class Multinomial(Static):
    """多项分布类"""
    def __init__(self):
        self.pdfs = []

    def fit(self, x):
        n_samples, n_features = x.shape
        for j in range(n_features):
            xj = x[:,j]
            count = Counter(xj)
            pdf = {}
            for k,v in count.items():
                pdf[k] = float(v) / n_samples
            self.pdfs.append(pdf)

    def pdf_lookup(self, pdf):
        def map_fun(val):
            if val in pdf:
                return pdf[val]
            else:
                return 0
        return map_fun

    def predict_proba(self, x):
        n_samples, n_features = x.shape
        proba = np.zeros_like(x)
        for j in range(n_features):
            pdf = self.pdfs[j]
            proba[:,j] = np.vectorize(self.pdf_lookup(pdf))(x[:,j])
        return proba
        

class NaiveBayes:
    """朴素贝叶斯类
    参数：
        pdf: 似然函数分布类型，支持'gaussian'(高斯分布)和'multinomial'
             (多项分布)两种，str类型；
        discrete_level: 最大离散量化级别，如果是多项分布则用到此参数。
    """
    def __init__(self, pdf='gaussian', discrete_level=3):
        self.pdf = pdf
        if pdf == 'multinomial':
            self.discretor = Discretor(min_number=discrete_level)

    def fit(self, x, y):
        x, y = check_data(x, y)
        self.classes_ = np.unique(y)
        n_samples = len(y)
        self.log_priors = np.zeros(len(self.classes_))  #类型的先验概率
        if self.pdf ==  'gaussian':
            self.statics = [Normal() for _ in range(len(self.classes_))]
        elif self.pdf == 'multinomial':
            self.discretor.fit(x, y)
            x = self.discretor.predict(x)
            self.statics = [Multinomial() for _ in range(len(self.classes_))]

        for i,c in enumerate(self.classes_):
            c_x = x[y == c, :]
            self.statics[i].fit(c_x)
            self.log_priors[i] = np.log(len(c_x)) - np.log(n_samples)

    def predict_log_proba(self, x):
        x = np.asarray(x)
        if self.pdf == 'multinomial':
            x = self.discretor.predict(x)
        log_like = np.zeros((x.shape[0],len(self.classes_)))  #似然概率
        for j,c in enumerate(self.classes_):
            log_like[:,j] = self.statics[j].joint_log_proba(x)
        log_post = log_like + self.log_priors  #后验概率
        return log_post

    def predict_proba(self, x):
        log_p = self.predict_log_proba(x)
        p = np.exp(log_p)
        return p / (np.sum(p, axis=1, keepdims=True) + epsilon)

    def predict(self, x):
        p = self.predict_proba(x)
        return self.classes_[p.argmax(1)]

if __name__=='__main__':
    from dataset import *
    from tools.compress import LDA
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='gaussian')
    args = parser.parse_args()
    
    print('Loading data...',end='')
    (x_train,y_train), (x_test,y_test) = load_data()
    x_train = x_train + np.random.normal(0,0.0001,x_train.shape)
    x_test = x_test + np.random.normal(0,0.0001,x_test.shape)
    print('Done!')
    
    print('Compress data with LDA ...',end='')
    lda = LDA(dims=8)  #朴素贝叶斯处理高维数据效果较差
    lda.fit(x_train, y_train)
    x_train = lda.transform(x_train)
    x_test = lda.transform(x_test)
    print('Done!')

    x_train = normalize(x_train)
    x_test = normalize(x_test)
    
    model = NaiveBayes(pdf=args.mode)
    #from sklearn.naive_bayes import GaussianNB
    #model = GaussianNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print('Valid accuracy: {}'.format(score))
    
            
