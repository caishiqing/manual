import numpy as np
import os

def abs_path(name):
    return os.path.normpath(os.path.join(os.getcwd(),
                                         os.path.dirname(__file__),
                                         name))

def load_data(path=abs_path('mnist.npz'), flatten=True,
              norm=True, lower=2):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    
    if norm:
        x_train = x_train / 255.0
        x_test = x_test / 255.0
    if lower:
        x_train = down_sample(x_train, lower)
        x_test = down_sample(x_test, lower)
    if flatten:
        x_train = np.reshape(x_train, [len(x_train),-1])
        x_test = np.reshape(x_test, [len(x_test),-1])
        
    return (x_train, y_train), (x_test, y_test)

def pooling(x, stride=(2,2)):
    r,c = stride
    kernel = np.ones((1,r,c))
    samples, row, col = x.shape
    new_row, new_col = int(row/r), int(col/c)
    new_x = np.zeros((samples, new_row, new_col))
    for i in range(new_row):
        for j in range(new_col):
            new_x[:, i,j] = np.mean(np.mean(x[:, i * r : (i+1) * r,
                                              j * c : (j+1) * c] * kernel,
                                            1), 1)
    return new_x

def down_sample(x, times=2):
    return pooling(x, (times, times))



    
