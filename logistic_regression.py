# -*- coding: utf-8 -*-
"""
标题：逻辑回归
作者：蔡世清
说明：numpy手撕逻辑回归，支持二分类、多分类，使用Adam优化器优化参数，除了
      dropout暂时不考虑其它正则。
最后修订日：2019-02-18
"""
from __future__ import print_function,unicode_literals
from tools import *
from tools.optimizer import *
import numpy as np
import random,time

class LogisticRegression:
    """逻辑回归类
    参数：
        lr: 初始学习率
        dropout：随机失活率
    """
    def __init__(self, lr=0.01, dropout=0.0):
        assert(lr > 0)
        self.lr = float(lr)
        self.dropout = min(max(dropout, 0.0), 1.0)
        self.optimizer = Adam(alpha=self.lr)

    def compute_output(self, x, epochs=10):
        x = np.asarray(x)
        if self.mode == 'binary':
            y = sigmoid(np.dot(x, self.w))
        elif self.mode == 'categorical':
            y = softmax(np.dot(x, self.w))
        return y

    def fit(self, x, y, epochs=10, verbose=0):
        """训练方法
        梯度公式：
            注：▽sigmoid = y(1-y)，y为标量
                ▽softmax = diag(y) - y*yT，y为向量
                
            二分类选择sigmoid输出，标签为y，则
            y_hat = sigmoid(w*x)
            Loss = -[y*log(y_hat) + (1-y)*log(1-y_hat)]
              ▽w = ∂L/∂y_hat * ∂y_hat/∂w
                  = [(1-y)/(1-y_hat) - y/y_hat] * y_hat*(1-y_hat) * x
                  = [(1-y)*y_hat - y*(1-y_hat)] * x
                  = y_res * x

            多分类选择softmax输出，标签为y，则
            y_hat = softmax(w*x)，y_hat是分布向量，y是one-hot向量，
            Loss = -yT*log(y_hat)
              ▽w = ∂L/∂y_hat * ∂y_hat/∂w
                  = -diag(y_hat)^-1*y * [diag(y_hat)-y_hat*y_hatT] * x
                  = (y_hat - y) * x
                  = y_res * x

            两种激活函数仅y_res不同；
            此处未考虑维度和batch，具体实现时需要注意维度扩展；
        """
        x, y = check_data(x, y)
        n_samples, n_features = x.shape
        self._classes = np.unique(y)
                
        if len(self._classes) > 2:
            self.mode = 'categorical'
            output_dim = len(self._classes)
            y = one_hot(y, size=output_dim)
        else:
            self.mode = 'binary'
            output_dim = 1
            y = np.expand_dims(y, 1)

        self.w = np.random.uniform(-0.05, 0.05, (n_features,output_dim))

        for epoch in range(1, epochs+1):
            y_hat = self.compute_output(x)
            if self.mode == 'categorical':
                y_res = np.expand_dims(y_hat-y, 1)
                x_expand = np.expand_dims(x, 2)
                grad = np.mean(x_expand * y_res, 0)
            elif self.mode == 'binary':
                y_res = np.expand_dims((1-y_true)*y - y_true*(1-y), 1)
                grad = np.mean(x * y_res, axis=0, keepdims=True)

            if self.dropout == 0:
                mask = 1
            else:
                mask = np.random.binomial(1, 1-self.dropout, self.w.shape)
            self.w -= self.optimizer.update(grad) * mask
            if verbose:
                if self.mode == 'binary':
                    loss = binary_crossentropy(y, y_hat)
                    acc = binary_accuracy(y, y_hat)
                elif self.mode == 'categorical':
                    loss = categorical_crossentropy(y, y_hat)
                    acc = categorical_accuracy(y, y_hat)
                desc = "Epoch {}/{} - loss: {:.4f} - acc: {:.4f} "
                print(desc.format(epoch, epochs, loss, acc))


    def predict_proba(self, x):
        x = np.asarray(x)
        y = self.compute_output(x)
        if self.mode == 'binary':
            y = np.vstack([1-y, y])
        return y

    def predict(self, x):
        y = self.predict_proba(x)
        return y.argmax(1)

if __name__=='__main__':
    from dataset import *
    from tools.compress import LDA
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()
    
    print('Loading data...',end='')
    (x_train,y_train), (x_test,y_test) = load_data()
    x_train = x_train + np.random.normal(0,0.0001,x_train.shape)
    x_test = x_test + np.random.normal(0,0.0001,x_test.shape)
    print('Done!')
    
    print('Compress data with LDA ...',end='')
    lda = LDA(dims=128)
    lda.fit(x_train, y_train)
    x_train = lda.transform(x_train)
    x_test = lda.transform(x_test)
    print('Done!')

    print('Preprocessing ...',end='')
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    print('Down!')
    
    model = LogisticRegression(lr=0.1, dropout=args.dropout)
    model.fit(x_train, y_train, epochs=args.epochs, verbose=1)
    '''
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(verbose=1)
    model.fit(x_train, y_train)
    '''
    
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print('Valid accuray: {:.4f}'.format(score))
