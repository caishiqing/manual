# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals
import numpy as np
from .utils import *

class Quantilization:  #特征值量化编码
    def __init__(self, levels, upper_percent=96, lower_percent=4):
        self.levels = levels
        self.upper_percent = upper_percent
        self.lower_percent = lower_percent

    def fit(self, x):
        n_samples, n_features = x.shape
        self.upper_bound = np.percentile(x, self.upper_percent, axis=0)
        self.lower_bound = np.percentile(x, self.lower_percent, axis=0)

    def predict(self, x):
        x_new = (x - self.lower_bound) / (self.upper_bound - \
                                          self.lower_bound + epsilon)
        x_new[x_new < 0] = 0
        x_new[x_new > 1] = 1 - epsilon
        x_new = (x_new * self.levels).astype('int32')
        return x_new

class Discretor:
    """离散编码器
    参数：
        min_number: 最小取值数量，如果取值数量小于此则认为是离散变量，
                    否则认为是连续变量；
        max_number: 如果变量为连续型，则先量化成不多于此值的区间中
    """
    def __init__(self, min_number=10, max_number=20):
        self.min_number = min_number
        self.max_number = max_number
        self.quant = Quantilization(max_number, 99, 1)
        self.splits = []

    def split(self, column, labels, split_val, margin=1.0):
        """切割数据集并返回切割结果的熵
        数据集已按特征值大小排好序，对应排序后的labels，每一个value
        可能对应不止一个样本。
        参数：
            column: 排好序的特征列，最多有self.max_number种取值
            labels: 当前数据集对应的标签序列
            split_val: 用什么数值来分割数据集
            margin: 边际值，表示每次分割都会产生一定代价，防止过多分割。
        """
        if len(labels) == 0:
            return 0
        split_idx = np.where(column==split_val)[0][-1] + 1
        split_labels = labels[:split_idx]
        label_count = Counter(split_labels)
        pdf = list(label_count.values())
        split_entr = entropy(pdf)  #切割数据集信息熵
        #系数少一个分母常量无影响
        return split_entr * len(split_labels) + margin

    def best_split_route(self, xj, y):
        totle_samples = len(xj)
        vals = sorted(np.unique(xj))
        argidx = xj.argsort()
        column = xj.copy()[argidx]
        labels = y.copy()[argidx]
        #记录每一步、每一个分割点对应的最优子分割值与熵
        routes = {self.min_number:{vals[-1]:(0,-1)}}
        for step in range(self.min_number-1, -1, -1):
            routes[step] = {}
            if step <= 0:
                now_vals = [0]  #第0步为根节点，用0表示
            else:
                now_vals = vals  #从第一步开始在值域中搜索最优切割值
            for now_v in now_vals:
                now_idx = np.where(column==now_v)[0][-1] + 1
                sub_x = column[now_idx:]
                sub_y = labels[now_idx:]
                res = []
                for next_v in routes[step+1]:
                    if next_v < now_v:
                        continue
                    try:
                        entr = self.split(sub_x, sub_y, next_v)
                    except: #非根节点的next_v不能是最小value
                        continue
                    res.append((entr+routes[step+1][next_v][0], next_v))
                if not res:
                    continue
                best_sub_split = min(res)
                routes[step][now_v] = best_sub_split
        split_vals = [routes[0][0][1]] #第0步的最优子分割值，即第1步最优分割
        for step in range(1, self.min_number):
            now_v = split_vals[-1]
            next_v = routes[step][now_v][1]
            if next_v >= vals[-1]:
                break
            split_vals.append(next_v)
        return split_vals

    def fit(self, x, y):
        x, y = check_data(x, y)
        n_samples, n_features = x.shape
        val_nums = [len(np.unique(x[:,j])) for j in range(n_features)]
        self.discrete_idx = []
        self.continuous_idx = []
        for i,n in enumerate(val_nums):
            if n > self.min_number:
                self.continuous_idx.append(i)
            else:
                self.discrete_idx.append(i)

        con_x = x[:, self.continuous_idx]
        self.quant.fit(con_x)
        con_x = self.quant.predict(con_x)
        self.best_splits = [] #记录每个连续型特征的最优切割值序列
        for j,idx in enumerate(self.continuous_idx):
            xj = con_x[:,j]
            best_split_vals = self.best_split_route(xj, y)
            self.best_splits.append(sorted(best_split_vals))

    def quantilize(self, split_vals):
        def map_fun(val):
            res = 0
            for v in split_vals:
                if val <= v:
                    return res
                res += 1
            return res
        return map_fun

    def predict(self, x):
        new_x = x.copy()
        con_x = x[:,self.continuous_idx]
        con_x = self.quant.predict(con_x)
        for j,vals in enumerate(self.best_splits):
            col = self.continuous_idx[j]
            new_x[:,j] = np.vectorize(self.quantilize(vals))(con_x[:,j])
        return new_x

if __name__=='__main__':
    x_train=[0,1,2,3,4,5,6,7,8,9,10]
    y_train=[0,0,0,0,0,2,2,2,1,1,1]
    x_train=np.expand_dims(x_train,1)
    model = Discretor()
    model.fit(x_train, y_train)
    x = model.predict(x_train)
                                                    
        
