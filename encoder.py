import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import LambdaLayer

# input size (None, 120, 35)
class Encoder(nn.Module):
    
    def __init__(self, latent_rep_size, max_length = 120, epsilon_std = 0.01):
        super(Encoder, self).__init__()

        self.latent_rep_size = latent_rep_size
        self.max_length = max_length
        self.epsilon_std = epsilon_std
        self.z_mean = None
        self.z_log_var = None

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
        return z_mean_ + torch.exp(z_log_var_ / 2) * epsilon

    def vae_loss(self, x, x_decoded_mean): # call this from forward
        x = x.view(x.size()[0], -1)
        x_decoded_mean = x_decoded_mean.view(x_decoded_mean.size()[0], -1)
        criterion = nn.BCELoss()
        bce_loss = criterion(x, x_decoded_mean)
        xent_loss = self.max_length * bce_loss
        k1_loss = - 0.5 * torch.mean(1 + self.z_log_var - torch.pow(self.z_mean, 2) - torch.exp(self.z_log_var))
        return xent_loss + k1_loss

    def forward(self, x): # run this first
        print("6")
        print(x.device)
        x = self.relu(self.conv_1(x)) # (None, 9, 25)
        x = self.relu(self.conv_2(x)) # (None, 9, 17)
        x = self.relu(self.conv_3(x)) # (None, 10, 7)
        x = x.view(x.size()[0], -1) # (None, 70)
        x = self.relu(self.linear_1(x)) # (None, 435)
        self.z_mean = self.linear_2(x)
        self.z_log_var = self.linear_3(x)
        print("7")
        print(self.z_mean.device)
        print("8")
        print(self.z_log_var.device)
        output_tuple = (self.vae_loss, LambdaLayer(self.sampling, output_shape = (-1, self.latent_rep_size,), name = 'lambda')([self.z_mean, self.z_log_var]))
        return output_tuple

if __name__ == "__main__":
    encoder = Encoder(292, 120, 0.01)
    x = torch.rand((100, 120, 35))
    print(encoder.forward(x))
