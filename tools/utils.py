# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals
import numpy as np
from collections import Counter
from copy import copy

epsilon = 1e-15
inf = 1e15

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(x,axis=-1):
    expx = np.exp(x)
    return expx / (np.sum(expx, axis=axis, keepdims=True) + epsilon)

def normalize(x):  #标准正态化
    n_samples, n_features = x.shape
    x_mean = x.mean(0)
    x_std = x.std(axis=0)
    x_std[x_std==0] = 1.
    return (x - x_mean) / (x_std + epsilon)

def normalization(x, lower=0., upper=1., axis=-1):  #规范化到区间
    x_min = x.min(axis=axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    new_x = (x - x_min) / (x_max - x_min + epsilon)
    interval = upper - lower
    return (new_x * interval) + lower

def one_hot(label, size=10):  #标签one-hot编码
    length = len(label)
    categorical_output = np.zeros((length, size), dtype='float32')
    categorical_output[range(length),label] = 1
    return categorical_output

def accuracy_score(y_true, y_pred):
    res = np.equal(y_true, y_pred).astype('float32')
    return np.mean(res)

def binary_crossentropy(y_true, y_pred):
    L = y_true * np.log(y_pred + epsilon) + (1-y_true) * \
        np.log(1 - y_pred + epsilon)
    return -np.mean(np.mean(L, 1), 0)

def binary_accuracy(y_true, y_pred):
    y_true_ = (y_true > 0.5)
    y_pred_ = (y_pred > 0.5)
    acc = np.equal(y_true_, y_pred_).astype('float32')
    return np.mean(np.mean(acc, 1), 0)

def categorical_crossentropy(y_true, y_pred):
    L = y_true * np.log(y_pred)
    return -np.mean(np.sum(L, axis=1), axis=0)

def categorical_accuracy(y_true, y_pred):
    argmax_true = np.argmax(y_true, 1)
    argmax_pred = np.argmax(y_pred, 1)
    return np.mean(np.equal(argmax_true, argmax_pred), 0)

def entropy(pdf, normed=False):  #计算概率分布的信息熵
    #normed: 是否归一化熵
    pdf = np.asarray(pdf, dtype='float')
    if not pdf.shape[-1]:
        return inf
    if pdf.shape[-1] == 1:
        return 0.0
    pdf /= pdf.sum(-1)
    log_p = np.log2(pdf + epsilon)
    e = -(pdf * log_p).sum(-1)
    if normed:
        length = pdf.shape[-1]
        return e / (np.log(length) + epsilon)
    else:
        return e

def check_data(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if np.ndim(x) != 2:
        raise Exception('X must has 2 dimensions (sample,feature)!')
    if np.ndim(y) != 1:
        raise Exception('Y must be 1 dimensions (sample,)!')
    if len(set(y)) <= 1:
        raise Warning('Labels less than 2!')
    return x, y

