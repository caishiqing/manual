# -*- coding: utf-8 -*-
"""
标题：神经网络
作者：蔡世清
说明：纯numpy实现浅层神经网络，适用于二分类、多分类，手推梯度公式，支持
      dropout，使用Adam优化器，可显示训练进度条。
最后修订日：2019-02-16
"""
from __future__ import print_function,unicode_literals
from tools import *
from tools.optimizer import *
import numpy as np
import random,time
from tqdm import tqdm
        
class NeuralNetwork(object):
    """神经网络类
    参数：
        input_dim: 输入向量的维度
        hidden_dim: 隐藏层神经元数量
        output_dim: 输出层神经元数量
        lr: 初始学习率
        dropout: 随机失活概率
    """
    def __init__(self, input_dim, hidden_dim, output_dim,
                 lr=0.01, dropout=0.0):
               
        #结构参数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        if output_dim <= 2:
            self.mode = 'binary'
        else:
            self.mode = 'categorical'
        #网络权重
        self.coefs = np.zeros((input_dim+output_dim+1, hidden_dim+1)) #参数打包
        self.W = self.coefs[:input_dim, :hidden_dim]  #(input_dim, hidden_dim)
        self.W[:] = np.random.uniform(-0.05, 0.05, (input_dim, hidden_dim))
        self.b1 = self.coefs[input_dim, :hidden_dim]  #(hidden_dim)
        self.U = self.coefs.T[:-1, input_dim+1:]  #(hidden_dim, output_dim)
        self.U[:] = np.random.uniform(-0.05, 0.05, (hidden_dim, output_dim))
        self.b2 = self.coefs[input_dim+1:, -1]  #(output_dim)
        #学习参数
        assert(lr>0)
        self.lr = float(lr)
        self.dropout = min(max(dropout, 0.0), 1.0)
        self.optimizer = Adam(alpha=self.lr)

    def __compute_h__(self, x):
        h = tanh(np.dot(x, self.W) + np.expand_dims(self.b1, 0))
        return h

    def __compute_y__(self, h):
        if self.mode == 'binary':
            y = sigmoid(np.dot(h, self.U) + np.expand_dims(self.b2, 0))
        elif self.mode == 'categorical':
            y = softmax(np.dot(h, self.U) + np.expand_dims(self.b2, 0))
        return y

    def __update__(self, x, y):
        """更新网络参数
        手推梯度公式：
            注：▽sigmoid = y(1-y)， ▽tanh = (1-y^2), y为标量
                ▽softmax = diag(y) - y*yT，y为分布向量
                
            输出激活若为sigmoid，标签为y，则
            h = tanh(w*x + b1),  y_hat = sigmoid(u*h + b2)
            Loss = -[y*log(y_hat) + (1-y)*log(1-y_hat)]
              ▽b2 = ∂L/∂y_hat * ∂y_hat/∂b2
                   = [(1-y)/(1-y_hat) - y/y_hat] * y_hat*(1-y_hat) * 1
                   = (1-y)*y_hat - y*(1-y_hat)
                令 = y_res
            则▽u = ∂L/∂y_hat * ∂y_hat/∂u
                  = y_res * h
              ▽b1 = ∂L/∂y_hat * ∂y_hat/∂h * ∂h/∂b1
                   = y_res * u * (1-h^2) * 1
            则▽w = ∂L/∂y_hat * ∂y_hat/∂h * ∂h/∂w
                  = y_res * u * (1-h^2) * x

            输出激活若为softmax，标签为y，则
            h = tanh(w*x + b1),  y_hat = softmax(u*h + b2)
            Loss = -yT*log(y_hat)
              ▽b2 = ∂L/∂y_hat * ∂y_hat/∂b2
                   = -diag(y_hat)^-1*y * [diag(y_hat)-y_hat*y_hatT] * 1
                   = y_hat - y
                令 = y_res
            则▽u = ∂L/∂y_hat * ∂y_hat/∂u
                  = y_res * h
              ▽b1 = ∂L/∂y_hat * ∂y_hat/∂h * ∂h/∂b1
                   =y_res * u * (1-h^2) * 1
            则▽w = ∂L/∂y_hat * ∂y_hat/∂h * ∂h/∂w
                  = y_res * u * (1-h^2) * x

            两种激活函数仅y_res不同；
            此处未考虑维度和batch，具体实现时需要注意维度扩展；
        """
        h = self.__compute_h__(x)  #(batch, hidden_dim)
        y_hat = self.__compute_y__(h)  #(batch, output_dim)
        if self.mode == 'binary':
            y_res = (1-y)*y_hat - y*(1-y_hat)  #(batch, output_dim)
        elif self.mode == 'categorical':
            y_res = y_hat - y  #(batch, output_dim)
        b2_ = np.mean(y_res, axis=0)  #输出层偏置梯度
        y_res = np.expand_dims(y_res, 1)  #(batch, 1, output_dim)
        #隐藏层网络参数梯度
        u_ = np.mean(y_res * np.expand_dims(h, 2), axis=0)
        
        x_expand = np.expand_dims(x, 2)  
        x_expand = np.expand_dims(x_expand, 3) #(batch, input_dim, 1, 1)
        h_expand = np.expand_dims(h, 1)  
        h_expand = np.expand_dims(h_expand, 3)  #(batch, 1, hidden_dim, 1)
        y_res_expand = np.expand_dims(y_res, 2)  #(batch, 1, 1, output_dim)
        u_expand = np.expand_dims(self.U, 0)  
        u_expand = np.expand_dims(u_expand, 1)  #(1, 1, hidden_dim, output_dim)
        w_ = y_res_expand * u_expand * (1-h_expand**2) * x_expand
        #输入层网络参数梯度
        w_ = np.mean(np.mean(w_, 0), 2)  #(input_dim, hidden_dim)
        b1_ = y_res_expand * u_expand * (1-h_expand**2)
        b1_ = np.mean(np.mean(np.squeeze(b1_, 1), 2), 0)  #隐藏层偏置梯度
        #添加dropout，随机屏蔽参数更新
        if self.dropout == 0:
            mask = 1
        else:
            mask = np.random.binomial(1, 1-self.dropout,
                                      (self.input_dim+self.output_dim+1,
                                       self.hidden_dim+1))
        grad = np.zeros((self.input_dim+self.output_dim+1, self.hidden_dim+1))
        grad[:self.input_dim, :-1] = w_
        grad[self.input_dim, :-1] = b1_
        grad[self.input_dim+1:, :-1] = u_.T
        grad[self.input_dim+1:, -1] = b2_
        self.coefs -= self.optimizer.update(grad) * mask
        return y_hat

    def fit(self, train_data, valid_data=None, batch_size=128,
            epochs=1, monitor='acc', verbose=0):
        """训练方法
        参数：
            train_data: 训练集（train_x, train_y)
            valid_data: 验证集（valid_x, valid_y）
            batch_size: mini_batch的大小
            epochs: 训练次数
            monitor: checkpoint监控指标，'acc'或'loss'，如果有验证集就参考
                     验证集的指标，如果没有就参考训练集的指标
        """
        x_train, y_train = train_data
        train_num = len(x_train)
        if valid_data:
            x_valid, y_valid = valid_data
            valid_num = len(x_valid)
        n_classes = len(np.unique(y_train))
        if n_classes < 2:
            raise Warning('There are less than 2 classes in the train set!')
        elif n_classes == 2:
            assert(self.output_dim == 1)
        else:
            assert(n_classes == self.output_dim)
            
        if self.mode == 'categorical':
            if y_train.ndim < 2:
                y_train = one_hot(y_train, size=n_classes)
            if valid_data:
                if y_valid.ndim < 2:
                    y_valid = one_hot(y_valid, size=n_classes)
        else:
            if y_train.ndim < 2:
                y_train = np.expand_dims(y_train, 1)
            if valid_data:
                if y_valid.ndim < 2:
                    y_valid = np.expand_dims(y_valid, 1)
             
        loops = int(train_num / batch_size)
        NCOLS = 100
        log_interval = max(1, int(loops/100))
        temp_coef = np.zeros_like(self.coefs)  #临时参数
        min_loss = inf  #初始设置为无穷大
        max_acc = -inf  #初始设置为无穷小

        for epoch in range(1, epochs+1):
            losses = []; accs = []
            if verbose:
                desc = "Epoch {}/{} - loss: {:.4f} - acc: {:.4f} "
                pbar = tqdm(initial=0, leave=True, total=loops, ncols=NCOLS)
            for loop in range(loops):
                train_ids = random.sample(range(train_num), batch_size)
                x_train_ = x_train[train_ids]
                y_train_ = y_train[train_ids]
                y = self.__update__(x_train_, y_train_)
                if self.mode == 'binary':
                    loss = binary_crossentropy(y_train_, y)
                    acc = binary_accuracy(y_train_, y)
                elif self.mode == 'categorical':
                    loss = categorical_crossentropy(y_train_, y)
                    acc = categorical_accuracy(y_train_, y)
                losses.append(loss)
                accs.append(acc)
                if verbose:
                    if loop % log_interval == 0:
                        pbar.desc = desc.format(epoch, epochs, loss, acc)
                        pbar.update(max(1,int(loops/100)))  #更新进度条
                
            if verbose: pbar.close()
            train_loss = np.mean(losses)
            train_acc = np.mean(accs)
            if valid_data:
                y_valid_pred = self.predict_proba(x_valid)
                if self.mode == 'binary':
                    valid_loss = binary_crossentropy(y_valid, y_valid_pred)
                    valid_acc = binary_accuracy(y_valid, y_valid_pred)
                elif self.mode == 'categorical':
                    valid_loss = categorical_crossentropy(y_valid, y_valid_pred)
                    valid_acc = categorical_accuracy(y_valid, y_valid_pred)

            if verbose:
                print('   Train - loss: %.4f, acc: %.4f'%(train_loss,
                                                          train_acc), end='\t')
                if valid_data:
                    print('   Valid - loss: %.4f, acc: %.4f'%(valid_loss,
                                                              valid_acc), '\n')

            #更新最优指标以及对应的网络参数
            if valid_data:
                _loss = valid_loss
                _acc = valid_acc
            else:
                _loss = train_loss
                _acc = train_acc
                
            if _loss < min_loss:
                min_loss = _loss
                if monitor == 'loss':
                    temp_coef[:] = self.coefs[:]
            if _acc > max_acc:
                max_acc = _acc
                if monitor == 'acc':
                    temp_coef[:] = self.coefs[:]

        self.coefs[:] = temp_coef[:]
        return self

    def __call__(self, x):
        x = np.asarray(x)
        h = self.__compute_h__(x)
        y = self.__compute_y__(h)
        return y

    def predict_proba(self, x):
        y = self.__call__(x)
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
    parser.add_argument('--batch', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.5)
    args = parser.parse_args()

    dimension = 128
    hidden_dim = 64
    dropout = args.dropout
    batch_size = args.batch
    epochs = args.epochs
    
    print('Loading data...',end='')
    (x_train,y_train), (x_test,y_test) = load_data()
    x_train = x_train + np.random.normal(0,0.0001,x_train.shape)
    x_test = x_test + np.random.normal(0,0.0001,x_test.shape)
    print('Done!')
    
    print('Compress data with LDA ...',end='')
    lda = LDA(dims=dimension)
    lda.fit(x_train, y_train)
    x_train = lda.transform(x_train)
    x_test = lda.transform(x_test)
    print('Done!')

    print('Preprocessing ...',end='')
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    print('Down!')

    model = NeuralNetwork(input_dim=dimension, hidden_dim=hidden_dim,
                          output_dim=10, lr=0.005, dropout=dropout)
    
    model.fit((x_train,y_train),
              (x_test,y_test),
              batch_size=batch_size,
              epochs=epochs,
              verbose=1)
    
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print('Valid accuray: {:.4f}'.format(score))
    
