# -*- coding: utf-8 -*-
"""
梯度优化器
"""
from __future__ import print_function,unicode_literals
import numpy as np
from .utils import epsilon

class Adam:  #adam优化器
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999):
        self.alpha = alpha
        self.lr = float(self.alpha)
        self.beta1 = beta1
        self.beta2 = beta2
        self.mt = 0
        self.vt = 0
        self.t = 0

    def update(self, grad):
        self.t += 1
        self.mt = self.beta1 * self.mt + (1-self.beta1) * grad
        self.vt = self.beta2 * self.vt + (1-self.beta2) * (grad**2)
        self.lr = (self.alpha * np.sqrt(1-self.beta2**self.t)/
                   (1-self.beta1**self.t))
        return self.lr * self.mt / (np.sqrt(self.vt) + epsilon)
