# -*- coding: utf-8 -*-
"""
标题：变分自动编码器
作者：蔡世清
说明：浅层网络手撕VAE，复用神经网络类型，编码器和解码器都为三层神经网络
      （输入层、隐藏层、输出层），其中隐藏层全部为tanh激活，解码器完全复
      用神经网络类（包括优化），输出层sigmoid激活；编码器输出为线性激活，
      损失梯度重新求导。条件变分自动编码器（CVAE）暂时有问题。
最后修订日：2019-02-23
"""
from __future__ import print_function,unicode_literals
from tools import *
from tools.optimizer import *
from neural_network import NeuralNetwork as NN
import numpy as np
import random,time
from tqdm import tqdm

def normal_sample(u=0., sigma=1., n_samples=1, dim=1):
    e = np.random.normal(loc=0, scale=1, size=(n_samples, dim))
    z = u + sigma * e
    return z, e  #需要保存e，作为采样函数的参数

def VAE_KL(u, sigma):
    return np.mean(-0.5 * (1 + 2*np.log(sigma) - u**2 - sigma**2))

def CVAE_KL(u_h, sigma_h, u_f, sigma_f):
    loss = 0.5 * (2*np.log(sigma_f) - 2*np.log(sigma_h) + \
                  sigma_h**2/sigma_f**2 + (u_h-u_f)**2/sigma_f**2 -1)
    return np.mean(loss)

class SimpleNN(NN):
    """输出为线性激活的简单神经网络"""
    def __compute__y__(self, h):
        y = np.dot(h, self.U) + np.expand_dims(self.b2, 0)
        return y

class VAE(object):
    """变分自动编码器
    模型结构：
        编码器：h = tanh(w1*x + b1),  [u,sigma] = W*h + B
                u = w2*h + b2,    sigma = w3*h + b3, 编码器输出线性激活
        采样函数：z = u + sigma * e,  e是标准正态分布采样，记录为常量
        解码器：g = tanh(w4*z + b4),  x_hat = sigmoid(w5*g + b5)
    参数：
        x_dim: 数据的维度
        h_dim: 编码器隐藏层维度
        z_dim: 隐状态维度
        g_dim: 解码器隐藏层维度
        lr: 初始学习率
        dropout: 随机失活率
        
    """
    def __init__(self, x_dim, h_dim, z_dim, g_dim, lr=0.01, dropout=0.0):
        #结构参数
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.g_dim = g_dim
        self.x_dim = x_dim

        #学习参数
        assert(lr>0)
        self.lr = float(lr)
        self.dropout = min(max(dropout, 0.0), 1.0)

        #网络参数
        self.encoder = SimpleNN(input_dim=x_dim,
                                hidden_dim=h_dim,
                                output_dim=2*z_dim,
                                lr=self.lr, dropout=self.dropout)
        
        self.encoder_optimizer = Adam(alpha=self.lr)

        self.decoder = NN(input_dim=z_dim,
                          hidden_dim=g_dim,
                          output_dim=x_dim,
                          lr=self.lr, dropout=self.dropout)

        self.decoder.mode = 'binary'

    def __update_decoder__(self, z, x):
        """更新解码器参数就使用常规对数似然损失来更新"""
        x_hat = self.decoder.__update__(z, x)
        return x_hat

    def __update_encoder__(self, x, y=None):
        """更新编码器参数使用对数似然与KL散度损失来更新
        标准VAE的标签y用不上。
        """
        n_samples, dim = x.shape
        h = self.encoder.__compute_h__(x)
        u_sigma = self.encoder.__compute_y__(h)
        u, sigma = u_sigma[:,:self.z_dim], u_sigma[:,self.z_dim:]
        z, e = normal_sample(u, sigma, n_samples, self.z_dim)
        g = self.decoder.__compute_h__(z)
        x_hat = self.decoder.__compute_y__(g)

        KL_grad = self.__encoder_KL_grad__(x, h, u, sigma, y=y)
        Likely_grad = self.__encoder_Likely_grad__(x, h, u, sigma, e, g, x_hat)
        grad = KL_grad + Likely_grad  #KL散度损失+对数似然损失

        if self.dropout == 0:
            mask = 1
        else:
            mask = np.random.binomial(1, 1-self.dropout,
                                      (self.x_dim+self.z_dim+1,
                                       self.h_dim+1))

        self.encoder.coefs -= self.encoder_optimizer.update(grad) * mask
        KL_loss = VAE_KL(u, sigma)
        return z, KL_loss

    def __encoder_KL_grad__(self, x, h, u, sigma, y=None):
        """KL散度损失函数的梯度,标准VAE不使用标签y：
        编码器：
            h = tanh(w1*x + b1),  [u,sigma] = W*h + B
            u = w2*h + b2,    sigma = w3*h + b3
        根据KL散度公式可得：
            KL = -0.5 * [1 + 2*log(sigma) - u^2 - sigma^2]
        那么求导得：
            ▽b3 = ∂KL/∂sigma * ∂sigma/∂b3
                 = (sigma - 1/sigma) * 1 = ▽sigma
            ▽w3 = ∂KL/∂sigma * ∂sigma/∂w3
                 = (sigma - 1/sigma) * h
            ▽b2 = ∂KL/∂u * ∂u/∂b2
                 = u * 1
            ▽w2 = ∂KL/∂u * ∂u/∂w2
                 = u * h
            ▽b1 = (∂KL/∂u * ∂u/∂h + ∂KL/∂sigma * ∂sigma/∂h) * ∂h/∂b1
                 = (u*w2 + ▽sigma*w3) * (1-h^2) * 1
            ▽w1 = (∂KL/∂u * ∂u/∂h + ∂KL/∂sigma * ∂sigma/∂h) * ∂h/∂w1
                 = (u*w2 + ▽sigma*w3) * (1-h^2) * x
        此处公式不考虑batch与变量维度，具体实现时需注意维度扩展。
        """
        sigma_grad = sigma - 1 / (sigma + epsilon)
        b3_ = np.mean(sigma_grad, 0)
        
        sigma_grad_expand = np.expand_dims(sigma_grad, 1) #(batch, 1, z_dim)
        h_expand = np.expand_dims(h, -1) #(batch, h_dim, 1)
        w3_ = np.mean(sigma_grad_expand * h_expand, 0)

        b2_ = np.mean(u, 0)
        u_expand = np.expand_dims(u, 1) #(batch, 1, z_dim)
        w2_ = np.mean(u_expand * h_expand, 0)

        #(1, h_dim, z_dim)
        w2_expand = np.expand_dims(self.encoder.U[:,:self.z_dim], 0)
        #(1, h_dim, z_dim)
        w3_expand = np.expand_dims(self.encoder.U[:,self.z_dim:], 0)
        b1_ = (u_expand * w2_expand + sigma_grad_expand * w3_expand) * \
              (1 - h_expand**2)
        b1_expand = np.expand_dims(b1_, 1) #(batch, 1, h_dim, z_dim)
        #(batch, x_dim, 1, 1)
        x_expand = np.expand_dims(np.expand_dims(x, -1), -1)
        w1_ = b1_expand * x_expand
        w1_ = np.mean(np.mean(w1_, 0), -1)
        b1_ = np.mean(np.mean(b1_, 0), -1)

        grad = np.zeros((self.x_dim+2*self.z_dim+1, self.h_dim+1))
        grad[:self.x_dim, :-1] = w1_
        grad[self.x_dim, :-1] = b1_
        grad[self.x_dim+1:, :-1] = np.column_stack([w2_,w3_]).T
        grad[self.x_dim+1:, -1] = np.vstack([b2_, b3_]).reshape([-1])
        return grad

    def __encoder_Likely_grad__(self, x, h, u, sigma, e, g, x_hat):
        """对数似然损失的梯度
        模型结构：
            编码器：h = tanh(w1*x + b1),  [u,sigma] = W*h + B
                    u = w2*h + b2,    sigma = w3*h + b3
            采样函数：z = u + sigma * e
            解码器：g = tanh(w4*z + b4),  x_hat = sigmoid(w5*g + b5)
        对数似然损失公式：
            L = -[x*log(x_hat) + (1-x)*log(1-x_hat)]
        那么求导得(更具体的推导参考神经网络)：
            ▽x_hat = (1-x)/(1-x_hat) - x/x_hat
            ▽z = ∂L/∂x_hat * ∂x_hat/∂g * ∂g/∂z
                = ▽x_hat * x_hat*(1-x_hat)*w5 + (1-g^2)*w4
                = [(1-x)*x_hat - x(1-x_hat)] * (1-g^2) * w4 * w5
            ∂z/∂u = 1,  ∂z/∂sigma = e
            ▽b3 = ∂L/∂z * ∂z/∂sigma * ∂sigma/∂b3
                 = ▽z * e * 1
            ▽w3 = ∂L/∂z * ∂z/∂sigma * ∂sigma/∂w3
                 = ▽z * e * h
            ▽b2 = ∂L/∂z * ∂z/∂u * ∂u/∂b2
                 = ▽z * 1 * 1
            ▽w2 = ∂L/∂z * ∂z/∂u * ∂u/∂w2
                 = ▽z * 1 * h
            ▽b1 = ∂L/∂z * (∂z/∂u * ∂u/∂h + ∂z/∂sigma * ∂sigma/∂h) * ∂h/∂b1
                 = ▽z * (w2 + e*w3) * 1
            ▽w1 = ∂L/∂z * (∂z/∂u * ∂u/∂h + ∂z/∂sigma * ∂sigma/∂h) * ∂h/∂w1
                 = ▽z * (w2 + e*w3) * x
        此处公式不考虑batch与变量维度，具体实现时需注意维度扩展。
        """
        x_res = (1-x) * x_hat - x * (1-x_hat)
        #(batch, 1, 1, x_dim)
        x_res_expand = np.expand_dims(np.expand_dims(x_res, 1), 1)
        #(batch, 1, g_dim, 1)
        g_expand = np.expand_dims(np.expand_dims(g, 1), -1)
        #(1, 1, g_dim, x_dim)
        w5_expand = np.expand_dims(
            np.expand_dims(self.decoder.U, 0), 0)
        #(1, z_dim, g_dim, 1)
        w4_expand = np.expand_dims(
            np.expand_dims(self.decoder.W, 0), -1)
        #(batch, z_dim, g_dim, x_dim)
        z_grad = x_res_expand * (1-g_expand**2) * w4_expand * w5_expand
        #reduce成(batch, z_dim)
        z_grad = np.mean(np.mean(z_grad, -1), -1)

        b2_ = np.mean(z_grad, 0)
        b3_ = np.mean(z_grad * e, 0)
        e_expand = np.expand_dims(e, 1)  #(batch, 1, z_dim)
        h_expand = np.expand_dims(h, -1) #(batch, h_dim, 1)
        z_grad_expand = np.expand_dims(z_grad, 1) #(batch, 1, z_dim)
        w2_ = np.mean(z_grad_expand * h_expand, 0)
        w3_ = np.mean(z_grad_expand * h_expand * e_expand, 0)

        #(batch, 1, 1, z_dim)
        z_grad_expand = np.expand_dims(z_grad_expand, 1)
        #(batch, x_dim, 1, 1)
        x_expand = np.expand_dims(np.expand_dims(x, -1), -1)
        #(1, 1, h_dim, z_dim)
        w2_expand = np.expand_dims(
            np.expand_dims(self.encoder.U[:,:self.z_dim], 0), 0)
        #(1, 1, h_dim, z_dim)
        w3_expand = np.expand_dims(
            np.expand_dims(self.encoder.U[:,self.z_dim:], 0), 0)
        #(batch, 1, 1, z_dim)
        e_expand = np.expand_dims(e_expand, 1)

        b1_ = z_grad_expand * (w2_expand + e_expand*w3_expand)
        w1_ = b1_ * x_expand
        w1_ = np.mean(np.mean(w1_, 0), -1)
        b1_ = np.squeeze(b1_, 1)
        b1_ = np.mean(np.mean(b1_, 0), -1)

        grad = np.zeros((self.x_dim+2*self.z_dim+1, self.h_dim+1))
        grad[:self.x_dim, :-1] = w1_
        grad[self.x_dim, :-1] = b1_
        grad[self.x_dim+1:, :-1] = np.column_stack([w2_,w3_]).T
        grad[self.x_dim+1:, -1] = np.vstack([b2_, b3_]).reshape([-1])
        return grad

    def fit(self, x, batch_size=100, epochs=1, verbose=0):
        """训练方法
        标准VAE为自监督模型，所以不需要标签
        """
        x = np.asarray(x)
        n_samples, dim = x.shape
        assert(self.x_dim == dim)
        loops = int(n_samples / batch_size)
        NCOLS = 80#min(100, loops)
        log_interval = max(1, int(loops/100))

        for epoch in range(1, epochs+1):
            if verbose:
                desc = "Epoch {}/{} - loss: {:.4f} "
                pbar = tqdm(initial=0, leave=True, total=loops, ncols=NCOLS)
            for loop in range(loops):
                idx = random.sample(range(n_samples), batch_size)
                x_sample = x[idx]
                z, KL_loss = self.__update_encoder__(x_sample)
                x_hat = self.__update_decoder__(z, x_sample)
                if verbose:
                    Likely_loss = binary_crossentropy(x_sample, x_hat)
                    loss = KL_loss + Likely_loss
                    if loop % log_interval == 0:
                        pbar.desc = desc.format(epoch, epochs, loss)
                        pbar.update(max(1,int(loops/100)))  #更新进度条

            if verbose: pbar.close()

        return self

    def encode(self, x):
        n_samples, dim = x.shape
        u_sigma = self.encoder(x)
        u, sigma = u_sigma[:,:dim], u_sigma[:,dim:]
        z, e = normal_sample(u, sigma, n_samples, dim)
        return z

    def decode(self, z):
        return self.decoder(z)

    def generate(self, n_samples=1):
        z, e = normal_sample(n_samples=n_samples, dim=self.z_dim)
        return self.decode(z)


class CVAE(VAE):
    """条件变分自动编码器，目前还存在问题
    模型结构：
        编码器：h = tanh(w1*[x;y] + b1),  [u,sigma] = W*h + B
                u = w2*h + b2,    sigma = w3*h + b3, 编码器输出线性激活
        采样函数：z = u + sigma * e,  e是标准正态分布采样，记录为常量
        解码器：g = tanh(w4*z + b4),  x_hat = sigmoid(w5*g + b5)
    参数：
        n_classes：条件类型数量
        embedding_dim：将条件变量嵌入该向量空间
        其它参数与VAE一样
    """
    def __init__(self, x_dim, h_dim, z_dim, g_dim, n_classes,
                 embedding_dim, lr=0.01, dropout=0.0):
        new_x_dim = x_dim + embedding_dim
        #输入维度加上条件维度
        super(CVAE, self).__init__(new_x_dim, h_dim, z_dim, g_dim,
                                   lr=lr, dropout=dropout)

        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.embeddings = np.random.uniform(0, 1, size=(n_classes,
                                                         embedding_dim))
        #类型的先验信息全部初始化为随机常量
        self.prior_z_mean = np.random.uniform(-1, 1, size=(n_classes, z_dim))
        self.prior_z_sigma = np.random.uniform(0, 1, size=(n_classes, z_dim))

    def __concatenate_xy__(self, x, y):
        c_vect = self.embeddings[y, :]
        new_x = np.column_stack([x, c_vect])  #拼接特征与条件向量
        return new_x

    def __update_encoder__(self, x, y):
        """CVAE训练编码器需要类型标签"""
        new_x = self.__concatenate_xy__(x, y)
        z, KL_loss = super(CVAE, self).__update_encoder__(new_x, y)
        return z, KL_loss

    def __encoder_KL_grad__(self, x, h, u, sigma, y):
        """KL散度损失函数的梯度
        """
        prior_u = self.prior_z_mean[y]
        prior_sigma = self.prior_z_sigma[y]

        sigma_grad = sigma / (prior_sigma**2 + epsilon) - \
                     1 / (sigma + epsilon)
        b3_ = np.mean(sigma_grad, 0)
        sigma_grad_expand = np.expand_dims(sigma, 1) #(batch, 1, z_dim)
        h_expand = np.expand_dims(h, -1) #(batch, h_dim, 1)
        w3_ = np.mean(sigma_grad_expand * h_expand, 0)

        u_grad = (u - prior_u) / (prior_sigma**2 + epsilon)
        b2_ = np.mean(u_grad, 0)
        u_grad_expand = np.expand_dims(u_grad, 1) #(batch, 1, z_dim)
        w2_ = np.mean(u_grad_expand * h_expand, 0)

        #(1, h_dim, z_dim)
        w2_expand = np.expand_dims(self.encoder.U[:,:self.z_dim], 0)
        #(1, h_dim, z_dim)
        w3_expand = np.expand_dims(self.encoder.U[:,self.z_dim:], 0)
        b1_ = (u_grad_expand * w2_expand + sigma_grad_expand * w3_expand)
        b1_expand = np.expand_dims(b1_, 1) #(batch, 1, h_dim, z_dim)
        #(batch, x_dim, 1, 1)
        x_expand = np.expand_dims(np.expand_dims(x, -1), -1)
        w1_ = b1_expand * x_expand
        w1_ = np.mean(np.mean(w1_, 0), -1)
        b1_ = np.mean(np.mean(b1_, 0), -1)

        input_dim = self.x_dim
        output_dim = self.z_dim * 2
        grad = np.zeros((input_dim+output_dim+1, self.h_dim+1))
        grad[:input_dim, :-1] = w1_
        grad[input_dim, :-1] = b1_
        grad[input_dim+1:, :-1] = np.column_stack([w2_,w3_]).T
        grad[input_dim+1:, -1] = np.vstack([b2_, b3_]).reshape([-1])
        return grad

    def fit(self, x, y, batch_size=100, epochs=1, verbose=0):
        """训练方法
        
        """
        x, y = check_data(x, y)
        n_samples, dim = x.shape
        assert(self.x_dim == dim+self.embedding_dim)
        loops = int(n_samples / batch_size)
        NCOLS = 80
        log_interval = max(1, int(loops/100))

        for epoch in range(1, epochs+1):
            if verbose:
                desc = "Epoch {}/{} - loss: {:.4f} "
                pbar = tqdm(initial=0, leave=True, total=loops, ncols=NCOLS)
            for loop in range(loops):
                idx = random.sample(range(n_samples), batch_size)
                x_sample = x[idx]
                y_sample = y[idx]
                #数据与条件变量拼接起来一起预测
                input = self.__concatenate_xy__(x_sample, y_sample)
                z, _ = self.__update_encoder__(x_sample, y_sample)
                output = self.__update_decoder__(z, input)
                if verbose:
                    u_sigma = self.encoder(input)
                    u, sigma = u_sigma[:,:self.z_dim], u_sigma[:,self.z_dim:]
                    prior_u = self.prior_z_mean[y_sample]
                    prior_sigma = self.prior_z_sigma[y_sample]
                    KL_loss = CVAE_KL(u, sigma, prior_u, prior_sigma)
                    Likely_loss = binary_crossentropy(input, output)
                    loss = KL_loss + Likely_loss
                    if loop % log_interval == 0:
                        pbar.desc = desc.format(epoch, epochs, loss)
                        pbar.update(max(1,int(loops/100)))  #更新进度条

            if verbose: pbar.close()

        return self

    def encode(self, x, y):
        new_x = self.__concatenate_xy__(x, y)
        n_samples, dim = new_x.shape
        u_sigma = self.encoder(new_x)
        u, sigma = u_sigma[:,:dim], u_sigma[:,dim:]
        z, e = normal_sample(u, sigma, n_samples, dim)
        return z

    def decode(self, z):
        return self.decoder(z)[:,:self.x_dim-self.embedding_dim]

    def generate(self, y, n_samples=1):
        prior_u = np.expand_dims(self.prior_z_mean[y], 0)
        prior_sigma = np.expand_dims(self.prior_z_sigma[y], 0)
        z, e = normal_sample(u=prior_u, sigma=prior_sigma,
                             n_samples=n_samples, dim=self.z_dim)
        return self.decode(z)
            
if __name__=='__main__':
    from dataset import *
    from tools.compress import LDA
    from argparse import ArgumentParser
    from matplotlib import pyplot as plt

    parser = ArgumentParser()
    parser.add_argument('--batch', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--state_dim', type=int, default=16)
    args = parser.parse_args()

    z_dim = args.state_dim
    g_dim = 32
    h_dim = 32
    batch_size = args.batch
    epochs = args.epochs
    
    print('Loading data...',end='')
    (x_train,y_train), (x_test,y_test) = load_data()
    _, x_dim = x_train.shape
    print('Done!')

    models = []
    for i in range(10):
        print('Training class {} :'.format(i))
        x = x_train[y_train==i]
        model = VAE(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim,
                    g_dim=g_dim, lr=0.005)
        
        model.fit(x, verbose=1,
                  batch_size=batch_size,
                  epochs=epochs)

        models.append(model)

    n = 10
    digit_size = 14
    figure = np.zeros((digit_size * n, digit_size * n))
    fake_x = model.generate(n*n)
    #fake_x[fake_x<0.4] = 0
    for i in range(n):
        fake_x = models[i].generate(n)
        for j in range(n):
            image = np.reshape(fake_x[j], [digit_size, digit_size])
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = image

    plt.figure(figsize=(6, 6))
    plt.imshow(figure, cmap='Greys_r')
    plt.title('VAE')
    plt.axis('off')
    plt.show()

