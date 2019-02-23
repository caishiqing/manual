# -*- coding: utf-8 -*-
"""
标题：生成对抗网络
作者：蔡世清
说明：浅层网络手撕GAN，生成器与判别器都适用三层网络（输入层、隐藏层、输出层），
      隐藏层都使用tanh激活，输出层都使用sigmoid，复用神经网络类型，其中判别
      器的完全复用二分类网络（包括优化），生成器损失梯度重新推导。另外浅层
      网络GAN两个网络很难达成均衡，生成器很难与判别器抗衡，所以加一点辅助监督
      帮助生成器（给生成器随机喂一批真实数据）。AcGAN暂时有问题。
最后修订日：2019-02-22
"""
from __future__ import print_function,unicode_literals
from tools import *
from tools.optimizer import *
from neural_network import NeuralNetwork as NN
import numpy as np
import random,time
from tqdm import tqdm

def normal_sample(u=0, sigma=1, dim=1, n=1):
    return np.random.normal(loc=u, scale=sigma, size=(n,dim))

def uniform_sample(lower=0., upper=1., dim=1, n=1):
    return np.random.uniform(lower, upper, size=(n,dim))

class GAN(object):
    """生成对抗网络
    网络结构：
        隐藏状态为z，观测数据为x，输出真假标签为y，则
        生成器：g = tanh(w1*z + b1),  x = sigmoid(u1*g + b2)
        判别器：h = tanh(w2*x + b3),  y = sigmoid(u2*h + b4)
    参数：
        z_dim: 隐状态维度
        g_dim: 生成器隐藏层维度
        x_dim: 数据的维度
        h_dim: 判别器隐藏层维度
        lr: 初始学习率
        dropout: 随机失活率
    """
    def __init__(self, z_dim, g_dim, x_dim, h_dim,
                 lr=0.01, dropout=0.0):
        #结构参数
        self.z_dim = z_dim
        self.g_dim = g_dim
        self.x_dim = x_dim
        self.h_dim = h_dim
        
        #学习参数
        assert(lr>0)
        self.lr = float(lr)
        self.dropout = min(max(dropout, 0.0), 1.0)
        ##要为生成器和判别器分别分配优化器
        self.generator_optimizer = Adam(alpha=self.lr)
        #self.discriminator_optimizer = Adam(alpha=self.lr)

        #网络参数
        self.generator = NN(input_dim=z_dim,
                            hidden_dim=g_dim,
                            output_dim=x_dim, lr=self.lr*0.5,
                            dropout=self.dropout)
        
        self.generator.mode = 'binary'
        
        self.discriminator = NN(input_dim=x_dim,
                                hidden_dim=h_dim,
                                output_dim=1, lr=self.lr*0.1,
                                dropout=self.dropout)

    def __update_discriminator__(self, x, y):
        """更新判别器参数就当做正常的二分类网络更新"""
        y_hat = self.discriminator.__update__(x, y)
        return y_hat

    def __update_generator__(self, z):
        """损失函数求导更新生成器
        模型结构：
            生成器：g = tanh(w1*z + b1),  x = sigmoid(u1*g + b2)
            判别器：h = tanh(w2*x + b3),  y_hat = sigmoid(u2*h + b4)
        对数似然损失函数（y为真伪标签）：
            L = -[y*log(y_hat) + (1-y)*log(1-y_hat)]
        求导得：
            ▽y_hat = (1-y)/(1-y_hat) - y/y_hat
            ▽x = ∂L/∂y_hat * ∂y_hat/∂h * ∂h/∂x
                = ▽y_hat * y_hat*(1-y_hat)*u2 * (1-h^2)*w2
                = [(1-y)*y_hat-y*(1-y_hat)] * u2*(1-h^2)*w2
            ∂x/∂u1 = x*(1-x)*g
                令 = x_res * g
            ∂x/∂w1 = ∂x/∂g * ∂g/∂w1
                   = x_res*u1 * (1-g^2)*z
            ▽b2 = ∂L/∂x * ∂x/∂b2
                 = ▽x * x_res * 1
            ▽u1 = ∂L/∂x * ∂x/∂u1
                 = ▽x * x_res * g
            ▽b1 = ∂L/∂x * ∂x/∂g * ∂g/∂b1
                 = ▽x * x_res*u1 * (1-g^2)*1
            ▽w1 = ∂L/∂x * ∂x/∂g * ∂g/∂21
                 = ▽x * x_res*u1 * (1-g^2)*z
        此处公式不考虑batch与变量维度，具体实现时需注意维度扩展。
        """
        g = self.generator.__compute_h__(z)
        x = self.generator.__compute_y__(g)
        h = self.discriminator.__compute_h__(x)
        y_hat = self.discriminator.__compute_y__(h)
        y_res = y_hat
        #(batch, 1, 1, y_dim)
        y_res_expand = np.expand_dims(np.expand_dims(y_res,1), 1)
        #(1, 1, h_dim, y_dim)
        u2_expand = np.expand_dims(
            np.expand_dims(self.discriminator.U, 0), 0)
        #(batch, 1, h_dim, 1)
        h_expand = np.expand_dims(np.expand_dims(h, 1), 3)
        #(1, x_dim, h_dim, 1)
        w2_expand = np.expand_dims(
            np.expand_dims(self.discriminator.W, 0), 3)
        #(batch, x_dim, h_dim, y_dim)
        x_grad = y_res_expand * u2_expand * (1-h_expand**2) * w2_expand
        #reduce成(batch, x_dim)
        x_grad = np.mean(np.mean(x_grad, -1), -1)

        x_res = x*(1-x)
        #(batch, 1, 1, x_dim)
        x_res_expand = np.expand_dims(np.expand_dims(x_res, 1), 2)
        #(batch, 1, g_dim, x_dim)
        g_expand = np.expand_dims(np.expand_dims(g, 1), 3)
        #(1, 1, g_dim, x_dim)
        u1_expand = np.expand_dims(
            np.expand_dims(self.generator.U, 0), 0)
        #(batch, z_dim, 1, 1)
        z_expand = np.expand_dims(np.expand_dims(z, -1), -1)
        #(batch, 1, 1, x_dim)
        x_grad_expand = np.expand_dims(
            np.expand_dims(x_grad, 1), 1)
        
        b2_ = x_grad_expand * x_res_expand
        u1_ = b2_ * g_expand
        u1_= np.mean(np.squeeze(u1_), 0)
        b2_ = np.mean(np.squeeze(b2_), axis=0)

        b1_ = x_grad_expand * x_res_expand * u1_expand * (1-g_expand**2)
        w1_ = b1_ * z_expand
        w1_ = np.mean(np.mean(w1_, 0), -1)
        b1_ = np.mean(np.mean(np.squeeze(b1_), 0), -1)

        if self.dropout == 0:
            mask = 1
        else:
            mask = np.random.binomial(1, 1-self.dropout,
                                      (self.z_dim+self.x_dim+1,
                                       self.g_dim+1))
            
        grad = np.zeros((self.z_dim+self.x_dim+1, self.g_dim+1))
        grad[:self.z_dim, :-1] = w1_
        grad[self.z_dim, :-1] = b1_
        grad[self.z_dim+1:, :-1] = u1_.T
        grad[self.z_dim+1:, -1] = b2_
        #要骗过判别器，所以要让损失函数增加
        self.generator.coefs += self.generator_optimizer.update(grad) * mask
        return y_hat

    def generate_z(self, n_samples):
        z = normal_sample(dim=self.z_dim, n=n_samples)
        return z

    def generate_x(self, z):
        fake_x = self.generator(z)
        return fake_x

    def generate(self, n_samples):
        z = normal_sample(dim=self.z_dim, n=n_samples)
        fake_x = self.generator(z)
        return fake_x

    def fit(self, x, batch_size=128, epochs=1, verbose=0):
        x = np.asarray(x)
        n_samples, n_features = x.shape
        if n_features != self.x_dim:
            raise Exception('Data dimensions should be equal to x_dim!')
        loops = int(n_samples / batch_size)
        NCOLS = 100
        log_interval = max(1, int(loops/100))
        
        for epoch in range(1, epochs+1):
            gen_x = self.generate(n_samples)
            if verbose:
                print("Epoch {}/{}:".format(epoch, epochs))
                desc = "Training Discriminator - loss: {:.4f} - acc: {:.4f} "
                pbar = tqdm(initial=0, leave=True, total=loops, ncols=NCOLS)
            for loop in range(loops):  #训练判别器
                idx = random.sample(range(n_samples), batch_size)
                fake_x = gen_x[idx]
                data_x = x[idx]
                mix_x = np.vstack([data_x, fake_x])
                labels = np.array([1]*batch_size + [0]*batch_size)
                shuff_idx = list(range(2*batch_size))
                random.shuffle(shuff_idx)
                mix_x = mix_x[shuff_idx]
                y = np.expand_dims(labels[shuff_idx], 1)
                y_pred = self.__update_discriminator__(mix_x, y)
                if verbose:
                    loss = binary_crossentropy(y, y_pred)
                    acc = binary_accuracy(y, y_pred)
                    if loop % log_interval == 0:
                        pbar.desc = desc.format(loss, acc)
                        pbar.update(max(1,int(loops/100)))  #更新进度条

            gen_z = self.generate_z(n_samples)
            if verbose:
                pbar.close()
                desc = "Training Generator - loss: {:.4f} - acc: {:.4f} "
                pbar = tqdm(initial=0, leave=True, total=loops, ncols=NCOLS)
            for loop in range(loops):  #训练生成器
                idx = random.sample(range(n_samples), batch_size)
                fake_z = gen_z[idx]
                data_x = x[idx]
                """
                加一点辅助监督：浅层网络生成器很难与判别器抗衡，所以加一点
                    辅助监督帮助生成器。
                """
                _ = self.generator.__update__(fake_z, data_x)
                y_hat = self.__update_generator__(fake_z) #只训练生成数据
                if verbose:
                    data_x = x[idx]
                    fake_x = self.generate_x(fake_z)
                    mix_x = np.vstack([data_x, fake_x])
                    y_true_pred = self.discriminator(data_x)
                    y_pred = np.vstack([y_true_pred, y_hat])
                    labels = np.array([1]*batch_size + [0]*batch_size)
                    y = np.expand_dims(labels, 1)
                    loss = binary_crossentropy(y, y_pred)
                    acc = binary_accuracy(y, y_pred)
                    if loop % log_interval == 0:
                        pbar.desc = desc.format(loss, acc)
                        pbar.update(max(1,int(loops/100)))  #更新进度条
            if verbose: pbar.close()

        return self

class AcGAN(GAN):
    def __init__(self, z_dim, g_dim, x_dim, h_dim,
                 n_classes, embedding_dim=8, lr=0.01, dropout=0.0):

        self.n_classes = n_classes
        new_z_dim = z_dim + embedding_dim
        super(AcGAN, self).__init__(new_z_dim, g_dim, x_dim,
                                    h_dim, lr, dropout)

        self.classifier = NN(input_dim=x_dim,
                             hidden_dim=h_dim,
                             output_dim=n_classes,
                             lr=self.lr,
                             dropout=self.dropout)
        #将条件变量嵌入到一个向量空间，并保持为常量
        self.c_vectors = normal_sample(dim=embedding_dim, n=n_classes)
        self.embedding_dim = embedding_dim

    def generate_z(self, n_samples, c):
        z = normal_sample(dim=self.z_dim-self.embedding_dim, n=n_samples)
        condition_stack = np.tile(self.c_vectors[c], [n_samples,1])
        #将条件变量拼接到状态向量上
        new_z = np.column_stack([z, condition_stack])
        return new_z

    def generate(self, n_samples, c):  #生成指定条件(类型)的数据
        z = self.generate_z(n_samples, c)
        fake_x = self.generate_x(z)
        return fake_x

    def __update_classifier__(self, x, y):
        y_hat = self.classifier.__update__(x, y)
        return y_hat

    def __update_generator_by_classifier__(self, z, y):
        g = self.generator.__compute_h__(z)
        x = self.generator.__compute_y__(g)
        h = self.discriminator.__compute_h__(x)
        y_hat = self.discriminator.__compute_y__(h)

        y_res = y_hat - y
        #(batch, 1, 1, y_dim)
        y_res_expand = np.expand_dims(np.expand_dims(y_res,1), 1)
        #(1, 1, h_dim, y_dim)
        u2_expand = np.expand_dims(
            np.expand_dims(self.discriminator.U, 0), 0)
        #(batch, 1, h_dim, 1)
        h_expand = np.expand_dims(np.expand_dims(h, 1), 3)
        #(1, x_dim, h_dim, 1)
        w2_expand = np.expand_dims(
            np.expand_dims(self.discriminator.W, 0), 3)
        #(batch, x_dim, h_dim, y_dim)
        x_grad = y_res_expand * u2_expand * (1-h_expand**2) * w2_expand
        #reduce成(batch, x_dim)
        x_grad = np.mean(np.mean(x_grad, -1), -1)

        x_res = x*(1-x)
        #(batch, 1, 1, x_dim)
        x_res_expand = np.expand_dims(np.expand_dims(x_res, 1), 2)
        #(batch, 1, g_dim, x_dim)
        g_expand = np.expand_dims(np.expand_dims(g, 1), 3)
        #(1, 1, g_dim, x_dim)
        u1_expand = np.expand_dims(
            np.expand_dims(self.generator.U, 0), 0)
        #(batch, z_dim, 1, 1)
        z_expand = np.expand_dims(np.expand_dims(z, -1), -1)
        #(batch, 1, 1, x_dim)
        x_grad_expand = np.expand_dims(
            np.expand_dims(x_grad, 1), 1)
        
        b2_ = x_grad_expand * x_res_expand
        u1_ = b2_ * g_expand
        u1_= np.mean(np.squeeze(u1_), 0)
        b2_ = np.mean(np.squeeze(b2_), axis=0)

        b1_ = x_grad_expand * x_res_expand * u1_expand * (1-g_expand**2)
        w1_ = b1_ * z_expand
        w1_ = np.mean(np.mean(w1_, 0), -1)
        b1_ = np.mean(np.mean(np.squeeze(b1_), 0), -1)

        if self.dropout == 0:
            mask = 1
        else:
            mask = np.random.binomial(1, 1-self.dropout,
                                      (self.z_dim+self.x_dim+1,
                                       self.g_dim+1))
            
        grad = np.zeros((self.z_dim+self.x_dim+1, self.g_dim+1))
        grad[:self.z_dim, :-1] = w1_
        grad[self.z_dim, :-1] = b1_
        grad[self.z_dim+1:, :-1] = u1_.T
        grad[self.z_dim+1:, -1] = b2_
        #生成器要使生成数据能让分类器正确分类，即减小分类损失
        self.generator.coefs -= self.generator_optimizer.update(grad) * mask
        return y_hat
        

    def fit(self, x, y, batch_size=128, epochs=1, verbose=0):
        x, y = check_data(x, y)
        n_samples, n_features = x.shape
        y = one_hot(y, self.n_classes)
        if n_features != self.x_dim:
            raise Exception('Data dimensions should be equal to x_dim!')
        loops = int(n_samples / batch_size)
        NCOLS = 100
        log_interval = max(1, int(loops/100))
        
        for epoch in range(1, epochs+1):
            gen_x = []
            for c in range(self.n_classes):
                gen = self.generate(int(n_samples/self.n_classes), c)
                gen_x.append(gen)
            gen_x = np.vstack(gen_x)
            shuff_idx = range(len(gen_x))
            random.shuffle(shuff_idx)
            gen_x = gen_x[shuff_idx]
            if verbose:
                print("Epoch {}/{}:".format(epoch, epochs))
                desc = "Discriminator - loss: {:.4f} - acc: {:.4f}  \
- classify_loss: {:.4} - classify_acc: {:.4} "
                pbar = tqdm(initial=0, leave=True, total=loops, ncols=NCOLS)
            for loop in range(loops):  #训练判别器
                idx = random.sample(range(n_samples), batch_size)
                fake_x = gen_x[idx]
                data_x = x[idx]
                data_y = y[idx]
                mix_x = np.vstack([data_x, fake_x])
                labels = np.array([1]*batch_size + [0]*batch_size)
                shuff_idx = range(2*batch_size)
                random.shuffle(shuff_idx)
                mix_x = mix_x[shuff_idx]
                fake = np.expand_dims(labels[shuff_idx], 1)
                fake_pred = self.__update_discriminator__(mix_x, fake)
                y_pred = self.__update_classifier__(data_x, data_y)
                if verbose:
                    loss = binary_crossentropy(fake, fake_pred)
                    acc = binary_accuracy(fake, fake_pred)
                    c_loss = categorical_crossentropy(data_y, y_pred)
                    c_acc = categorical_accuracy(data_y, y_pred)
                    if loop % log_interval == 0:
                        pbar.desc = desc.format(loss, acc, c_loss, c_acc)
                        pbar.update(max(1,int(loops/100)))  #更新进度条

            gen_z = []; classes = []
            for c in range(self.n_classes):
                gen = self.generate_z(int(n_samples/self.n_classes), c)
                gen_z.append(gen)
                classes.extend([c]*int(n_samples/self.n_classes))
            gen_z = np.vstack(gen_z)
            classes = one_hot(classes, size=self.n_classes)
            shuff_idx = range(len(classes))
            random.shuffle(shuff_idx)
            gen_z = gen_z[shuff_idx]
            classes = classes[shuff_idx]
            if verbose:
                pbar.close()
                desc = "Generator - loss: {:.4f} - acc: {:.4f} \
- classify_loss: {:.4} - classify_acc: {:.4} "
                pbar = tqdm(initial=0, leave=True, total=loops, ncols=NCOLS)
            for loop in range(loops):  #训练生成器
                idx = random.sample(range(len(classes)), batch_size)
                fake_z = gen_z[idx]
                fake_hat = self.__update_generator__(fake_z) #只训练生成数据
                fake_y = classes[idx]
                y_pred = self.__update_generator_by_classifier__(fake_z, fake_y)
                if verbose:
                    data_x = x[idx]
                    fake_x = self.generate_x(fake_z)
                    mix_x = np.vstack([data_x, fake_x])
                    fake_true_pred = self.discriminator(data_x)
                    fake_pred = np.vstack([fake_true_pred, fake_hat])
                    fakes = np.array([1]*batch_size + [0]*batch_size)
                    fake_labels = np.expand_dims(fakes, 1)
                    loss = binary_crossentropy(fake_labels, fake_pred)
                    acc = binary_accuracy(fake_labels, fake_pred)
                    c_loss = categorical_crossentropy(fake_y, y_pred)
                    c_acc = categorical_accuracy(fake_y, y_pred)
                    if loop % log_interval == 0:
                        pbar.desc = desc.format(loss, acc, c_loss, c_acc)
                        pbar.update(max(1,int(loops/100)))  #更新进度条
            if verbose: pbar.close()

        return self

if __name__=='__main__':
    from dataset import *
    from tools.compress import LDA
    from argparse import ArgumentParser
    from matplotlib import pyplot as plt

    parser = ArgumentParser()
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=5)
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
        x = x_train[y_train==i]
        model = GAN(z_dim=z_dim, g_dim=g_dim,
                    x_dim=x_dim, h_dim=h_dim,
                    lr=0.005)
        print('Training class {} :'.format(i))
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
    plt.title('GAN')
    plt.axis('off')
    plt.show()
            
