# -*- coding: utf-8 -*-
"""
标题：决策树
作者：蔡世清
说明：对连续型特征进行多级量化编码，不像传统决策树只使用二分量化，而是使用
      动态规划对连续型属性搜索最优多级划分量化方案，量化算法的复杂度随量化
      级别增加，参见tools.quantilize
最后修订日：2019-02-18
"""
from __future__ import print_function,unicode_literals
from tools import *
from tools._tree import *
from tools.quantilize import Discretor
import numpy as np

class DecisionTree:
    """决策树类
    参数：
        max_depth: 决策树最大深度
        discrete_level: 连续型属性最大量化级别
    """
    def __init__(self, max_depth=8, discrete_level=3):
        self.max_depth = max_depth
        self.discretor = Discretor(discrete_level, 30)

    def fit(self, x, y):
        x, y = check_data(x, y)
        self.discretor.fit(x, y)
        x = self.discretor.predict(x)
        self._classes = list(np.unique(y))
        n_samples, n_features = x.shape
        dataSet = np.column_stack([x, np.expand_dims(y, 1)])
        self._tree = createTree(dataSet, range(n_features), self._classes,
                                self.max_depth)
        return self

    def search(self, vec, tree):  #递归地搜索特征向量对应的节点路径
        feat = list(tree.keys())[0]
        val = vec[feat]
        try:
            sub_tree = tree[feat][val]
            if type(list(sub_tree.values())[0]) != dict: #如果到了叶子层
                return sub_tree
            return self.search(vec, sub_tree)
        except:  #如果出现未知特征值
            try:
                results = [self.search(vec, tree[feat][v]) for v in tree[feat]]
            except: #如果到了叶子层
                return tree
            return dictAdd(results)
        

    def predict_proba(self, x):
        x = self.discretor.predict(x)
        res = np.zeros((len(x), len(self._classes)))
        for i,vec in enumerate(x):
            class_count = self.search(vec, self._tree)
            for f,num in class_count.items():
                res[i][f] = num
        res /= (np.sum(res, axis=1, keepdims=True) + epsilon)
        return res

    def predict(self, x):
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=1)
    
if __name__=='__main__':
    from dataset import *
    from tools.compress import LDA
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--max_depth', type=int, default=9)
    args = parser.parse_args()
    
    print('Loading data...',end='')
    (x_train,y_train), (x_test,y_test) = load_data()
    x_train = x_train + np.random.normal(0,0.0001,x_train.shape)
    x_test = x_test + np.random.normal(0,0.0001,x_test.shape)
    print('Done!')
    
    print('Compress data with LDA ...',end='')
    lda = LDA(dims=args.max_depth)
    lda.fit(x_train, y_train)
    x_train = lda.transform(x_train)
    x_test = lda.transform(x_test)
    print('Done!')
    
    model = DecisionTree(max_depth=args.max_depth)
    '''
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(max_depth=9)
    '''
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print('Valid accuray: {}'.format(score))
    
