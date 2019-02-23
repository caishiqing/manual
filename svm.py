# -*- coding: utf-8 -*-
"""
标题：最小二乘支持向量机
作者：蔡世清
最后修订日：2019-02-16

说明：分别基于稀疏分块矩阵和稠密矩阵实现，稀疏分块矩阵的实现版本目前还有BUG，
      基于numpy稠密矩阵的实现空间复杂度较高，利用训练集预采样与内置均衡采样
      降低映射维度，此demo为基于numpy稠密矩阵的实现。
"""
from __future__ import print_function,unicode_literals
import numpy as np
from scipy.sparse import csr,linalg,eye,vstack
from tools import *
import random

MAX_PARTITION = 10000
MAX_N_ELEMENTS = MAX_PARTITION * MAX_PARTITION

# ------------------------------ kernel functions --------------------------- #
def linear_kernel(xi, xj):
    return np.dot(xi, xj.T)

def poly_kernel(xi, xj, n=2):
    return np.power(linear_kernel(xi, xj), n)

def rbf_kernel(xi, xj, sigma=1.0):
    _xi = np.expand_dims(xi, 1)
    _xj = np.expand_dims(xj, 0)
    dist = np.sum(np.power((_xi-_xj), 2), axis=-1) / (2 * sigma**2)
    return np.exp(-dist)

def sigmoid_kernel(xi, xj, theta=0, beta=0.01):
    return np.tanh(beta * linear_kernel(xi, xj) + theta)

kernel_functions = {'linear':linear_kernel, 'poly':poly_kernel,
                    'rbf':rbf_kernel, 'sigmoid':sigmoid_kernel}
# ---------------------------------------------------------------------------- #

def sparse_kernel(xi, xj, limit=None, kernel='rbf', percent=100):
    assert(xi.shape[1] == xj.shape[1])
    r_samples, n_features = xi.shape
    c_samples, n_features = xj.shape
    if limit is None:
        col_limit = min(MAX_N_ELEMENTS/min(MAX_PARTITION, r_samples)/int(n_features**0.5), c_samples)
        row_limit = min(MAX_N_ELEMENTS/min(MAX_PARTITION, c_samples)/int(n_features**0.5), r_samples)
        limit = max(row_limit, col_limit)
    if kernel not in ['linear', 'poly', 'rbf', 'sigmoid']:
        raise Exception('Kernel function must be in "linear","poly",\
                        "rbf","sigmoid"')
    kernel_function = kernel_functions[kernel]

    csr_mat = csr.csr_matrix((r_samples,c_samples), dtype='float32')
    r_blocks = (r_samples-1) / limit + 1 #每行有多少块
    c_blocks = (c_samples-1) / limit + 1#每列有多少块
    for i in range(r_blocks):
        for j in range(c_blocks):
            row_lower, row_upper = i*limit, min((i+1)*limit, r_samples)
            col_lower, col_upper = j*limit, min((j+1)*limit, c_samples)
            print(r_samples,row_lower,row_upper,' ',c_samples,col_lower,col_upper)
            _xi = xi[row_lower: row_upper,:]
            _xj = xj[col_lower: col_upper,:]
            block = kernel_function(_xi, _xj)
            #block = block - block.min()
            upper_bound = np.percentile(block, 100-percent)
            block[block < upper_bound] = 0  #只保留最大的percent%的元素
            csr_mat[row_lower: row_upper, col_lower: col_upper] = \
                               csr.csr_matrix(block)

    return csr_mat

def sparse_multiply_dense(sparse_mat, dense_mat, limit=None):
    assert(sparse_mat.shape == dense_mat.shape)
    rows, cols = sparse_mat.shape
    if limit is None:
        col_limit = min(MAX_N_ELEMENTS / min(MAX_PARTITION, rows), cols)
        row_limit = min(MAX_N_ELEMENTS / min(MAX_PARTITION, cols), rows)
        limit = max(row_limit, col_limit)
    r_blocks = (rows-1) / limit + 1 #每行有多少块
    c_blocks = (cols-1) / limit + 1 #每列有多少块
    res = csr.csr_matrix((rows, cols), dtype=sparse_mat.dtype)
    for i in range(r_blocks):
        for j in range(c_blocks):
            row_lower, row_upper = i*limit, min((i+1)*limit, rows)
            col_lower, col_upper = j*limit, min((j+1)*limit, cols)
            mat1 = sparse_mat[row_lower: row_upper, col_lower: col_upper]
            mat2 = dense_mat[row_lower: row_upper, col_lower: col_upper]
            block = mat1.multiply(csr.csr_matrix(mat2))  #分块乘
            res[row_lower: row_upper, col_lower: col_upper] = block
    return res

class SPSVM:
    """稀疏分块矩阵实现最小二乘支持向量机类
    训练集较大时LSSVM空间复杂度太高，使用矩阵分块计算和阈值截断保留稀疏矩阵
    降低空间复杂度，目前存在BUG。
    解alpha和b的方程组：
     —            —  —  —    —  —
    | 0 |   y_T     | |  b  |   |  0  |
    |———————-| |-----| = |-----|
    | y | omiga+I/C | |alpha|   |  1v |
     —            —  —  —    —  —
     其中alpha是支持向量的权重，b是偏置，y是支持向量对应的标签，
     omiga是核矩阵，C是松弛度惩罚系数。
    """
    def __init__(self, kernel='rbf', C=1.0, limit=None,
                 percent=5, verbose=1):
        self.kernel = kernel
        self.C = max(float(C), epsilon)
        self.limit = limit
        self.percent = percent
        self.verbose = verbose

    def __signal_mat__(self, y):  #符号矩阵：Yij = yi*yj
        row_y = np.expand_dims(y, 1).astype('int8')
        col_y = np.expand_dims(y, 0).astype('int8')
        return row_y * col_y

    def __complete_mat__(self, mat, y):  #完善系数矩阵
        c_diag = 1/self.C * eye(len(y), dtype='float32')
        sign = self.__signal_mat__(y)
        sign_mat = sparse_multiply_dense(mat, sign, self.limit)
        temp = vstack([y, sign_mat+c_diag])
        y_ = [0] + list(y)
        complete_mat = vstack([y_, temp.T]).T
        return csr.csr_matrix(complete_mat)

    def fit(self, x, y):
        self.n_classes = len(np.unique(y))
        if self.n_classes > 2:
            y = one_hot(y, size=self.n_classes)
        if np.ndim(y) == 1:
            y = np.expand_dims(y, 1)

        y[y==y.min()] = -1
        y[y==y.max()] = 1  #标签统一成1和-1
        n_samples, n_classes = y.shape
        n_samples, n_features = x.shape
        self.n_classes = n_classes
        support_vectors_num = int(self.percent/100.0 * n_samples)  #支持向量的数量
        #存储每个类型的支持向量
        self.support_vectors = np.zeros((support_vectors_num, self.n_classes, n_features))
        #存储每个类型的支持向量的标签
        self.support_vector_labels = np.zeros((support_vectors_num, self.n_classes))
        self.coefs = np.zeros((support_vectors_num+1, n_classes)) #只保留支持向量的系数
        kernel_csr = sparse_kernel(x, x, self.limit,
                                   self.kernel, self.percent)

        const = np.array([0]+[1]*n_samples, dtype='float32')
        const = csr.csr_matrix(np.expand_dims(const, 1)) #(n_samples+1, 1)
        self.support_index = np.zeros((support_vectors_num, self.n_classes)) ####
        for c in range(self.n_classes):
            yc = y[:,c]
            A = self.__complete_mat__(kernel_csr, yc)
            if self.verbose:
                print('Solving class {}\'s coefficients...'.format(c))
            coefs = linalg.spsolve(A, const)  #解线性方程组求系数
            #第c个类型的支持向量的索引，按照前self.percent%大的alpha截取
            support_index = np.argsort(-np.abs(coefs[1:]))[:support_vectors_num]
            self.support_vectors[:,c,:] = x[support_index,:]
            self.support_vector_labels[:,c] = y[support_index,c]
            self.coefs[:,c] = coefs[[0]+list(support_index)]
            self.support_index[:,c] = support_index

    def decision(self, x):
        n_samples, _ = x.shape
        outputs = np.zeros((n_samples, self.n_classes), dtype='float32')
        for c in range(self.n_classes):
            support_vectors = self.support_vectors[:,c,:]
            coef = self.coefs[:,c]
            y_ = self.support_vector_labels[:,c].astype('int8')
            y_expand = np.tile(y_, [n_samples,1])
            #核矩阵
            kernel_csr = sparse_kernel(x, support_vectors, self.limit, self.kernel)
            #映射到高维的新数据矩阵
            new_x = sparse_multiply_dense(kernel_csr, y_expand, self.limit)
            csr_mat = vstack([[1]*n_samples, new_x.T]).T
            output = csr_mat.dot(coef)
            outputs[:,c] = output

        return outputs

    def predict_proba(self, x):
        outputs = self.decision(x)
        if self.n_classes == 2:
            positive = sigmoid(outputs)
            negtive = 1 - positive
            return np.column_stack([negtive,positive])
        elif self.n_classes > 2:
            outputs = normalization(outputs)
            return softmax(outputs)

    def predict(self, x):
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=1)

class LSSVM:
    """稠密矩阵实现最小二乘SVM类型
    使用稠密矩阵纯numpy实现，空间复杂度高，需预先对样本重采样；内置均衡
    采样策略，平衡正负例比例同时进一步降低映射维度。
    解alpha和b的方程组：
     —            —  —  —    —  —
    | 0 |   y_T     | |  b  |   |  0  |
    |———————-| |-----| = |-----|
    | y | omiga+I/C | |alpha|   |  1v |
     —            —  —  —    —  —
     其中alpha是支持向量的权重，b是偏置，y是支持向量对应的标签，
     omiga是核矩阵，C是松弛度惩罚系数。
    参数：
        C：松弛因子惩罚系数，float型
        kernel：核函数类型，str型
    """
    def __init__(self, C=1.0, kernel='rbf'):
        self.C = float(C)
        self.kernel = kernel_functions[kernel.lower()]

    def fit(self, x, y):
        """
        标签数据统一转换成2D数组，第2个维度代表类型，如果是2分类则第2个维度
        长度为1，如果是多分类则每个类型单独看成一个2分类。
        """
        self.n_classes = len(np.unique(y))
        if self.n_classes > 2:
            y = one_hot(y, size=self.n_classes)
        if np.ndim(y) == 1:
            y = np.expand_dims(y, 1)

        y[y==y.min()] = -1
        y[y==y.max()] = 1  #标签统一成1和-1
        n_samples, n_classes = y.shape
        n_samples, n_features = x.shape
        self.n_classes = n_classes
        self.coefs = []  #每个类型训练不同的系数
        self.support_index = []  #存储每个类型支持向量的id
        self.x = x
        self.y = y

        for c in range(self.n_classes):
            yc = self.y[:,c]  #第c类样本的标签
            pos_idx = list(np.where(yc == 1)[0])
            neg_idx = list(np.where(yc == -1)[0])
            if len(pos_idx) < len(neg_idx):
                neg_idx = random.sample(neg_idx, len(pos_idx))
            elif len(pos_idx) > len(neg_idx):
                pos_idx = random.sample(pos_idx, len(neg_idx))
            idx = list(pos_idx) + list(neg_idx)
            self.support_index.append(idx)
            yc = yc[idx]  #均衡正负例样本
            xc = x[idx]
            kernel_mat = self.kernel(xc, xc)
            y_ = np.expand_dims(yc, 1).astype('int8')
            _y = np.expand_dims(yc, 0).astype('int8')
            yy = y_ * _y
            omiga = kernel_mat * yy
            diag = np.diag([1.0/self.C]*len(idx))
            mat = omiga + diag
            mat = np.vstack([yc, mat])
            A = np.vstack([[0]+list(yc),mat.T]).T
            const = np.array([0]+[1]*len(idx))
            coef = np.linalg.solve(A, const)
            self.coefs.append(coef)

    def decision(self, x):  #决策函数
        n_samples, _ = x.shape
        res = np.zeros((n_samples, self.n_classes))
        for c in range(self.n_classes):
            idx = self.support_index[c]
            labels = self.y[idx]  #取出每个类型的支持向量以及对应标签
            support_vectors = self.x[idx]
            y_ = np.expand_dims(labels[:,c], 0)
            kernel_mat = self.kernel(x, support_vectors)
            mat = y_ * kernel_mat
            mat = np.vstack([[1]*n_samples,mat.T]).T
            res[:,c] = np.dot(mat, self.coefs[c])
        return res

    def predict_proba(self, x):
        outputs = self.decision(x)
        if self.n_classes == 2:
            positive = sigmoid(outputs)
            negtive = 1 - positive
            return np.column_stack([negtive,positive])
        elif self.n_classes > 2:
            outputs = normalization(outputs)
            return softmax(outputs)

    def predict(self, x):
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=1)
    
if __name__=='__main__':
    from dataset import *
    from tools.compress import LDA
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--kernel', type=str, default='rbf')
    parser.add_argument('--C', type=float, default=1.0)
    args = parser.parse_args()

    print('Loading data...',end='')
    (x_train,y_train), (x_test,y_test) = load_data()
    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')
    x_train = x_train + np.random.normal(0,0.0001,x_train.shape)
    x_test = x_test + np.random.normal(0,0.0001,x_test.shape)
    print('Done!')
    
    print('Compress data with LDA ...',end='')
    lda = LDA(dims=8)
    lda.fit(x_train, y_train)
    x_train = lda.transform(x_train)
    x_test = lda.transform(x_test)
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    #训练样本重采样，减小空间复杂度
    train_idx = random.sample(range(len(x_train)), 20000)
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]
    print('Done!')

    model = LSSVM(kernel=args.kernel, C=args.C)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print('Valid accuray: {}'.format(score))
