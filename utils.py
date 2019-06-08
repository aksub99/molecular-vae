import torch
import torch.nn as nn
import gzip
import pandas
import h5py
import numpy as np

def one_hot_array(i, n):
    return map(int, [ix == i for ix in xrange(n)])

def one_hot_index(vec, charset):
    return map(charset.index, vec)

def from_one_hot_array(vec):
    oh = np.where(vec == 1)
    if oh[0].shape == (0, ):
        return None
    return int(oh[0][0])

def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()

def load_dataset(filename, split = True):
    h5f = h5py.File(filename, 'r')
    if split:
        data_train = h5f['data_train'][:]
    else:
        data_train = None
    data_test = h5f['data_test'][:]
    charset =  h5f['charset'][:]
    h5f.close()
    if split:
        return (data_train, data_test, charset)
    else:
        return (data_test, charset)


class LambdaLayer(nn.Module):
    def __init__(self, lambd, output_shape, name):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
        self.output_shape = output_shape
        self.name = name
    
    def forward(self, x):
        return self.lambd(x).view(self.output_shape)


class RepeatVector(nn.Module):
    def __init__(self, n):
        super(RepeatVector, self).__init__()
        self.n = n

    def forward(self, x):
        h = x.repeat(self.n, 1)
        return h.view(x.shape[0], -1, x.shape[1])
    