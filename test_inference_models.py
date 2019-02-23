# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals
from dataset import *
from tools.compress import LDA,PCA
from tools import *
import random

from sklearn.ensemble import RandomForestClassifier as sklearn_RF
from sklearn.neural_network import MLPClassifier as sklearn_NN
from sklearn.linear_model import LogisticRegression as sklearn_LR
from sklearn.naive_bayes import GaussianNB as sklearn_NB
from sklearn.tree import DecisionTreeClassifier as sklearn_DT
from sklearn.svm import SVC as sklearn_SVM

from neural_network import NeuralNetwork as my_NN
from random_forest import RandomForest as my_RF
from logistic_regression import LogisticRegression as my_LR
from decision_tree import DecisionTree as my_DT
from svm import LSSVM as my_SVM
from naive_bayes import NaiveBayes as my_NB

model_names = ['神经网络','支持向量机','逻辑回归','朴素贝叶斯',
               '随机森林','决策树']

competitors = ['sklearn','手撕']

results = {c:{m:0 for m in model_names} for c in competitors}

print('Loading data...',end='')
(x_train,y_train), (x_test,y_test) = load_data(flatten=True)
x_train = x_train + np.random.normal(0, 0.0001, x_train.shape)
x_test = x_test + np.random.normal(0, 0.0001, x_test.shape)
print('Done!\n')
    

# ----------------------------- 朴素贝叶斯 ----------------------------- #
print('开始训练朴素贝叶斯... ')
lda = LDA(dims=8)  #朴素贝叶斯处理高维数据容易欠拟合
lda.fit(x_train, y_train)
_x_train = lda.transform(x_train)
_x_test = lda.transform(x_test)
_x_train = normalize(_x_train)
_x_test = normalize(_x_test)

my_model = my_NB(pdf='gaussian')
sklearn_model = sklearn_NB()
print('训练sklearn模型... ', end='')
sklearn_model.fit(_x_train, y_train)
print('Done!')
y_pred = sklearn_model.predict(_x_test)
sklearn_score = accuracy_score(y_test, y_pred)
print('sklearn朴素贝叶斯准确率：{:.4}'.format(sklearn_score))
print('训练手撕模型... ', end='')
my_model.fit(_x_train, y_train)
print('Done!')
y_pred = my_model.predict(_x_test)
my_score = accuracy_score(y_test, y_pred)
print('手撕朴素贝叶斯准确率：{:.4}'.format(my_score))
results['sklearn']['朴素贝叶斯'] = sklearn_score
results['手撕']['朴素贝叶斯'] = my_score
print('\n')
del my_model, sklearn_model
# ---------------------------------------------------------------------- #

# -----------------------------   决策树   ----------------------------- #
print('开始训练决策树... ')
lda = LDA(dims=9)  #决策树处理高维数据容易过拟合
lda.fit(x_train, y_train)
_x_train = lda.transform(x_train)
_x_test = lda.transform(x_test)

my_model = my_DT(max_depth=9)
sklearn_model = sklearn_DT(max_depth=9)
print('训练sklearn模型... ', end='')
sklearn_model.fit(_x_train, y_train)
print('Done!')
y_pred = sklearn_model.predict(_x_test)
sklearn_score = accuracy_score(y_test, y_pred)
print('sklearn决策树准确率：{:.4}'.format(sklearn_score))
print('训练手撕模型... ', end='')
my_model.fit(_x_train, y_train)
print('Done!')
y_pred = my_model.predict(_x_test)
my_score = accuracy_score(y_test, y_pred)
print('手撕决策树准确率：{:.4}'.format(my_score))
results['sklearn']['决策树'] = sklearn_score
results['手撕']['决策树'] = my_score
print('\n')
del my_model, sklearn_model
# ---------------------------------------------------------------------- #

# ----------------------------   随机森林   ---------------------------- #
print('开始训练随机森林... ')
lda = LDA(dims=16)
lda.fit(x_train, y_train)
_x_train = lda.transform(x_train)
_x_test = lda.transform(x_test)

my_model = my_RF(n_estimators=30, max_depth=9, multiprocess=False)
sklearn_model = sklearn_RF(n_estimators=30, max_depth=9)
print('训练sklearn模型... ', end='')
sklearn_model.fit(_x_train, y_train)
print('Done!')
y_pred = sklearn_model.predict(_x_test)
sklearn_score = accuracy_score(y_test, y_pred)
print('sklearn随机森林准确率：{:.4}'.format(sklearn_score))
print('训练手撕模型... ', end='')
my_model.fit(_x_train, y_train)
print('Done!')
y_pred = my_model.predict(_x_test)
my_score = accuracy_score(y_test, y_pred)
print('手撕随机森林准确率：{:.4}'.format(my_score))
results['sklearn']['随机森林'] = sklearn_score
results['手撕']['随机森林'] = my_score
print('\n')
del my_model, sklearn_model
# ---------------------------------------------------------------------- #

# ----------------------------   逻辑回归   ---------------------------- #
print('开始训练逻辑回归... ')
lda = LDA(dims=128)
lda.fit(x_train, y_train)
_x_train = lda.transform(x_train)
_x_test = lda.transform(x_test)
_x_train = normalize(_x_train)
_x_test = normalize(_x_test)

my_model = my_LR(lr=0.1, dropout=0.1)
sklearn_model = sklearn_LR()
print('训练sklearn模型... ', end='')
sklearn_model.fit(_x_train, y_train)
print('Done!')
y_pred = sklearn_model.predict(_x_test)
sklearn_score = accuracy_score(y_test, y_pred)
print('sklearn逻辑回归准确率：{:.4}'.format(sklearn_score))
print('训练手撕模型... ', end='')
my_model.fit(_x_train, y_train, epochs=50)
print('Done!')
y_pred = my_model.predict(_x_test)
my_score = accuracy_score(y_test, y_pred)
print('手撕逻辑回归准确率：{:.4}'.format(my_score))
results['sklearn']['逻辑回归'] = sklearn_score
results['手撕']['逻辑回归'] = my_score
print('\n')
del my_model, sklearn_model
# ---------------------------------------------------------------------- #

# ---------------------------   支持向量机   --------------------------- #
print('开始训练支持向量机... ')
lda = LDA(dims=8)  #维度太高svm空间复杂度过高
lda.fit(x_train, y_train)
_x_train = lda.transform(x_train)
_x_test = lda.transform(x_test)
_x_train = normalize(_x_train)
_x_test = normalize(_x_test)
#训练样本重采样，减小空间复杂度
train_idx = random.sample(range(len(x_train)), 20000)
_x_train = _x_train[train_idx]
_y_train = y_train[train_idx]

my_model = my_SVM(kernel='rbf', C=1.0)
sklearn_model = sklearn_SVM(C=1.0)
print('训练sklearn模型... ', end='')
sklearn_model.fit(_x_train, _y_train)
print('Done!')
y_pred = sklearn_model.predict(_x_test)
sklearn_score = accuracy_score(y_test, y_pred)
print('sklearn支持向量机准确率：{:.4}'.format(sklearn_score))
print('训练手撕模型... ', end='')
my_model.fit(_x_train, _y_train)
print('Done!')
y_pred = my_model.predict(_x_test)
my_score = accuracy_score(y_test, y_pred)
print('手撕支持向量机准确率：{:.4}'.format(my_score))
results['sklearn']['支持向量机'] = sklearn_score
results['手撕']['支持向量机'] = my_score
print('\n')
del my_model, sklearn_model
# ---------------------------------------------------------------------- #

# ----------------------------   神经网络   ---------------------------- #
print('开始训练神经网络... ')
batch_size = 200
epochs = 10
hidden_dim = 64
lr = 0.005
dropout = 0.5

lda = LDA(dims=128)
lda.fit(x_train, y_train)
_x_train = lda.transform(x_train)
_x_test = lda.transform(x_test)
_x_train = normalize(_x_train)
_x_test = normalize(_x_test)

my_model = my_NN(input_dim=128, hidden_dim=hidden_dim,
                 output_dim=10, lr=lr, dropout=dropout)
sklearn_model = sklearn_NN(hidden_layer_sizes=(hidden_dim,),
                           batch_size=batch_size,
                           learning_rate_init=lr,
                           #max_iter=len(_x_train)/batch_size*epochs,
                           activation='tanh')
print('训练sklearn模型... ', end='')
sklearn_model.fit(_x_train, y_train)
print('Done!')
y_pred = sklearn_model.predict(_x_test)
sklearn_score = accuracy_score(y_test, y_pred)
print('sklearn神经网络准确率：{:.4}'.format(sklearn_score))
print('训练手撕模型... ', end='')
my_model.fit((_x_train, y_train),
             batch_size=batch_size,
             epochs=10)
print('Done!')
y_pred = my_model.predict(_x_test)
my_score = accuracy_score(y_test, y_pred)
print('手撕神经网络准确率：{:.4}'.format(my_score))
results['sklearn']['神经网络'] = sklearn_score
results['手撕']['神经网络'] = my_score
print('\n')
del my_model, sklearn_model
# ---------------------------------------------------------------------- #

from matplotlib import pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

x = np.arange(len(model_names))
total_width, n = 0.6, 2
width = total_width / n
x = x - (total_width - width)

my_scores = [results['手撕'][m] for m in model_names]
sklearn_scores = [results['sklearn'][m] for m in model_names]

plt.figure(figsize=(9,5))
plt.bar(x, my_scores,  width=width, label='手撕',
        color='blue')#, tick_label=model_names)
plt.bar(x + width, sklearn_scores, width=width,
        color='coral', label='sklearn')
plt.subplots_adjust(left=0.09, bottom=0.13, right=0.95, top=None)
plt.xticks(range(len(x)), model_names, rotation=20, fontsize=14)
plt.xlim(-0.8, 5.8)
plt.ylim(0.6, 1.0)
plt.ylabel('准确率', fontsize=13.5)
plt.legend(fontsize=13)
plt.show()


