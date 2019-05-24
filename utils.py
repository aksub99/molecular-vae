import torch
import torch.nn as nn

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

    