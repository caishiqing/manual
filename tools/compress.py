# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals
import numpy as np
import warnings
from scipy.linalg import eigh

class LDA:
        def __init__(self, dims=None):
            self.dims = dims

        def _solve_(self, x, y):
            n_samples, n_features = x.shape
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)
            self.priors_ = np.ones(n_classes, dtype='float32') / n_classes
            means = []; covs = []
            for group in self.classes_:
                x_ = x[y==group,:]
                means.append(x_.mean(0))
                covs.append(np.cov(x_.T, bias=1))
            means = np.asarray(means)
            Sw = np.mean(covs, axis=0)
            St = np.cov(x.T, bias=1)
            Sb = St - Sw
            
            evals, evecs = eigh(Sb, Sw)
            #evals, evecs = np.linalg.eigh(np.dot(np.linalg.inv(Sw), Sb))
            evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
            evecs /= np.apply_along_axis(np.linalg.norm, 0, evecs)

            self.scalings_ = evecs
        

        def fit(self, x, y):
            self._solve_(x, y)
            return self

        def transform(self, x):
            if self.dims == None:
                self.dims = x.shape[1]
            x_new = np.dot(x, self.scalings_)
            return x_new[:, :self.dims]

class PCA:
    def __init__(self, dims=None):
        self.dims = dims

    def fit(self, x):
        meanVals = np.mean(x, axis=0, keepdims=True)
        meanRemoved = x - meanVals
        covMat = np.cov(meanRemoved, rowvar=0)
        vals, vecs = np.linalg.eig(covMat)
        argsort = np.argsort(-vals)
        vals = vals[argsort]
        vecs = vecs[:,argsort]
        self.vals, self.vecs = vals, vecs

    def transform(self, x):
        if self.dims == None:
            self.dims = x.shape[1]
        rejectMat = self.vecs[:,:self.dims]
        return np.dot(x,rejectMat)


if __name__=='__main__':
    from dataset import *
    (x_train,y_train), (x_test,y_test) = load_data()
    x_train = down_sample(x_train, 2)
    x_test = down_sample(x_test, 2)
    samples, w, h = x_test.shape
    x_test = x_test + np.random.normal(0,0.01,x_test.shape)
    x_test = x_test.reshape([samples,w*h])/255.0
    '''
    pca = PCA(dims=2)
    pca.fit(x_test)
    x = pca.transform(x_test)
    '''
    lda = LDA(dims=2)
    #from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    #lda = LinearDiscriminantAnalysis(n_components=2, solver='eigen')
    lda.fit(x_test, y_test)
    x = lda.transform(x_test)
    x = normalize(x)
    
    from matplotlib import pyplot as plt
    #colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'gold', 'goldenrod']
    colours = 'rgb'
    for i in range(3):
        idx = np.where(y_test==i)
        x_data = x[idx]
        plt.scatter(x_data[:,0], x_data[:,1], c=colours[i])
    plt.show()
    
