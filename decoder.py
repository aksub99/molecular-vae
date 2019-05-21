import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import LambdaLayer
from utils import RepeatVector

class Decoder(nn.Module):
    def __init__(self, latent_rep_size, max_length, charset_length):
        super(Decoder, self).__init__()
        
        self.latent_rep_size = latent_rep_size
        self.max_length = max_length
        self.charset_length = charset_length

        self.linear_1 = nn.Linear(latent_rep_size, latent_rep_size)
        self.repeat_vector = RepeatVector(self.n)
