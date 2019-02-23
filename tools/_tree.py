# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals
from math import log
from .utils import epsilon, entropy
import numpy as np
from collections import Counter

def calcShannonEnt(dataSet):  #计算数据集的熵
    dataSet = np.asarray(dataSet)
    labels = dataSet[:,-1]
    label_count = Counter(labels)
    pdf = list(label_count.values())
    return entropy(pdf)

def splitDataSet(dataSet, axis, value):
    dataSet = np.asarray(dataSet)
    idx = np.where(dataSet[:,axis]==value)[0]
    sub_data = dataSet[idx, :]
    cols = list(range(sub_data.shape[1]))
    cols.remove(axis)  #去除第axis列
    sub_data = sub_data[:, cols]  #第axis列特征等于value的子数据块        
    return sub_data

def chooseBestFeatureToSplit(dataSet):
    dataSet = np.asarray(dataSet)
    numFeatures = len(dataSet[0]) - 1   #feature个数
    baseEntropy = calcShannonEnt(dataSet)   #整个dataset的熵
    bestInfoGainRatio = 0.0
    bestFeature = 0
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  #每个feature的list
        uniqueVals = set(featList)  #每个list的唯一值集合                 
        newEntropy = 0.0
        splitInfo = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)  #每个唯一值对应的剩余feature的组成子集
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            splitInfo += -prob * log(prob, 2)
        infoGain = baseEntropy - newEntropy     #这个feature的infoGain
        if (splitInfo == 0): 
            continue
        infoGainRatio = infoGain / splitInfo    #这个feature的infoGainRatio      
        if (infoGainRatio > bestInfoGainRatio): #选择最大的gain ratio
            bestInfoGainRatio = infoGainRatio
            bestFeature = i #选择最大的gain ratio对应的feature
    return bestFeature
            
def majorityCnt(classList, classSet):
    classCount = {c:0 for c in classSet}
    for vote in classList:
        classCount[vote] += 1
    return classCount

    
def createTree(dataSet, features, classSet, max_depth=1e10, depth=0):
    classList = [example[-1] for example in dataSet]
    features = list(features)
    if classList.count(classList[0]) == len(classList):
        #classList所有元素都相等，即类别完全相同，停止划分
        return majorityCnt(classList, classSet) #splitDataSet(dataSet, 0, 0)此时全是N，返回N
    if len(dataSet[0]) == 1 or depth >= max_depth: 
        #返回叶子层
        return majorityCnt(classList, classSet)
    bestFeat = chooseBestFeatureToSplit(dataSet)  
        #选择最大的gain ratio对应的feature
    bestFeatLabel = features[bestFeat]    
    tree = {bestFeatLabel:{}} #用嵌套字典构造决策树
    del(features[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]    
    uniqueVals = set(featValues)
    for value in uniqueVals:
        sub_features = features[:]
        tree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value),
            sub_features, classSet, max_depth, depth+1)
        #划分数据，为下一层计算准备
    return tree

def dictAdd(dicts):  #合并多个划分结果
    res = {}
    for d in dicts:
        for k,v in d.items():
            if k not in res:
                res[k] = v
            else:
                res[k] += v
    return res
