import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import LambdaLayer
from utils import RepeatVector

class Decoder(nn.Module):
    def __init__(self, latent_rep_size, max_length, charset_length):
        super(Decoder, self).__init__()
        
        self.linear_1 = nn.Linear(latent_rep_size, latent_rep_size) # (None, latent_rep_size)
        self.repeat_vector = RepeatVector(max_length) # (None, n, latent_rep_size)
        self.gru_1 = nn.GRU(latent_rep_size, 501, 3) # (None, n, 501)
        self.linear_2 = nn.Linear(501, charset_length)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = self.relu(self.linear_1(x))
        x = self.repeat_vector(x)
        output, hn = self.gru_1(x)
        output = self.softmax(self.linear_2(output))
        return output
