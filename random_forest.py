# -*- coding: utf-8 -*-
"""
标题：随机森林
作者：蔡世清
说明：使用多进程并行构建森林，连续型属性使用动态规划搜索最优多级划分,
      参见决策树。
最后修订日：2019-02-16
"""
from __future__ import print_function,unicode_literals
import numpy as np
from tools import *
from tools._tree import *
from tools.quantilize import Discretor
from multiprocessing import Process,Pool
import random,copy

def create_tree(args):
    dataSet, features, classSet, max_depth = args #参数打包
    return createTree(dataSet, features, classSet, max_depth, 0)

def search(vec, tree):  #递归地搜索特征向量对应的节点路径
        feat = list(tree.keys())[0]
        val = vec[feat]
        try:
            sub_tree = tree[feat][val]
            if type(list(sub_tree.values())[0]) != dict: #如果到了叶子层
                return sub_tree
            return search(vec, sub_tree)
        except:  #如果出现未知特征值
            try:
                results = [search(vec, tree[feat][v]) for v in tree[feat]]
            except:  #如果到了叶子层
                return tree
            return dictAdd(results)

def loop_search(args):
    x, tree, n_classes = args
    res = np.zeros((len(x), n_classes))
    for i,vec in enumerate(x):
        class_count = search(vec, tree)
        for c,num in class_count.items():
            res[i][c] = num
    res /= np.sum(res, axis=1, keepdims=True)
    return res

class RandomForest:
    """随机森林类
    使用多进程并行构建子树。
    参数：
        n_estimators: 子树的数量
        max_depth: 每棵树最大深度
        multiprocess: 是否使用多进程
        raw_sample_rate: 行采样率，每颗子树用一份子数据集训练
        discrete_level: 连续型属性的量化级别
        workers: 进程数量
    """
    def __init__(self, n_estimators=30, max_depth=7,
                 multiprocess=False, raw_sample_rate=None,
                 discrete_level=3, workers=4):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.discretor = Discretor(discrete_level, 30)
        self.multiprocess = multiprocess
        if raw_sample_rate == None:
            self.raw_sample_rate = 1.0 / np.log(n_estimators + 2)
        else:
            self.raw_sample_rate = raw_sample_rate
        if self.multiprocess:
            self.workers = workers

    def gen_dataset(self, x, y):
        return np.column_stack([x, np.expand_dims(y, 1)])
    
    def fit(self, x, y):
        x, y = check_data(x, y)
        self.discretor.fit(x, y)
        x = self.discretor.predict(x)
        self._classes = list(np.unique(y))
        n_samples, n_features = x.shape
        self.n_features = n_features
        self.features = range(n_features)
        #记录每个子模型对应的特征集合
        self.sub_features = [random.sample(self.features, self.max_depth) \
                             for _ in range(self.n_estimators)]
        
        raw_samples = int(n_samples * self.raw_sample_rate)
        sub_dataset_ids = [random.sample(range(n_samples), raw_samples) \
                           for _ in range(self.n_estimators)]
        #行采样和列采样
        dataSets = [self.gen_dataset(x[ids][:,fs], y[ids]) for \
                    ids,fs in zip(sub_dataset_ids,self.sub_features)]
        
        max_depths = [self.max_depth for _ in range(self.n_estimators)]
        
        kargs = [(dataSets[i], range(len(self.sub_features[i])), self._classes,
                  max_depths[i]) for i in range(self.n_estimators)]
        if self.multiprocess:
            pool = Pool(self.workers)  #多进程实现并行构建森林
            self.trees = list(pool.map(create_tree, kargs))
        else:
            self.trees = list(map(create_tree, kargs))
    
    def predict_proba(self, x):
        x = np.asarray(x)
        x = self.discretor.predict(x)
        sub_xs = [x[:,fs] for fs in self.sub_features]
        classes = [len(self._classes) for _ in range(self.n_estimators)]
        if self.multiprocess:
            pool = Pool(self.workers)
            results = list(pool.map(loop_search,
                                    zip(*[sub_xs,self.trees,classes])))
        else:
            results = list(map(loop_search,
                               zip(*[sub_xs,self.trees,classes])))
        result = np.mean(np.array(results), 0)
        return result

    def predict(self, x):
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=1)
            
            
if __name__=='__main__':
    from dataset import *
    from tools.compress import LDA
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=30)
    parser.add_argument('--max_depth', type=int, default=9)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    
    print('Loading data...',end='')
    (x_train,y_train), (x_test,y_test) = load_data()
    x_train = x_train + np.random.normal(0,0.0001,x_train.shape)
    x_test = x_test + np.random.normal(0,0.0001,x_test.shape)
    print('Done!')
    
    print('Compress data with LDA ...',end='')
    lda = LDA(dims=16)
    lda.fit(x_train, y_train)
    x_train = lda.transform(x_train)
    x_test = lda.transform(x_test)
    print('Done!')

    model = RandomForest(n_estimators=args.n_estimators,
                         max_depth=args.max_depth,
                         workers=args.workers)
    #from sklearn.ensemble import RandomForestClassifier
    #model = RandomForestClassifier(n_estimators=40, max_depth=9)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print('Valid accuray: {}'.format(score))
        

