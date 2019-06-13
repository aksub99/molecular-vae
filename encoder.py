import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import LambdaLayer

# input size (Batch size, 120, 33)
class Encoder(nn.Module):
    
    def __init__(self, latent_rep_size, max_length = 120, epsilon_std = 0.01):
        super(Encoder, self).__init__()

        self.latent_rep_size = latent_rep_size
        self.max_length = max_length
        self.epsilon_std = epsilon_std

        self.conv_1 = nn.Conv1d(in_channels = 120, out_channels = 9, kernel_size = 9)
        self.conv_2 = nn.Conv1d(in_channels = 9, out_channels = 9, kernel_size = 9)
        self.conv_3 = nn.Conv1d(in_channels = 9, out_channels = 10, kernel_size = 11)
        self.linear_1 = nn.Linear(10*7, 435)
        self.linear_2 = nn.Linear(435, latent_rep_size)
        self.linear_3 = nn.Linear(435, latent_rep_size)

        self.relu = nn.ReLU()
    
    def sampling(self, args):
        z_mean_, z_log_var_ = args
        batch_size = list(z_mean_.shape)[0]
        epsilon = torch.randn(batch_size, self.latent_rep_size) * self.epsilon_std + 0.
        epsilon = epsilon.cuda()
        return z_mean_ + torch.exp(z_log_var_ / 2) * epsilon

    def forward(self, x): # run this first
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = self.relu(self.conv_1(x)) # (None, 9, 25)
        x = self.relu(self.conv_2(x)) # (None, 9, 17)
        x = self.relu(self.conv_3(x)) # (None, 10, 7)
        x = x.view(x.size()[0], -1) # (None, 70)
        x = self.relu(self.linear_1(x)) # (None, 435)
        z_mean = self.linear_2(x)
        z_log_var = self.linear_3(x)
        lambd = LambdaLayer(self.sampling, output_shape = (-1, self.latent_rep_size,), name = 'lambda').to(device)
        output = (lambd([z_mean, z_log_var]), z_mean, z_log_var)
        return output
