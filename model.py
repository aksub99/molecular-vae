import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class MoleculeVAE(nn.Module):
    
    def __init__(self, charset, max_length = 120, latent_rep_size = 292):
        super(MoleculeVAE, self).__init__()
        charset_length = len(charset)
        self.encoder = Encoder(latent_rep_size, max_length)
        self.decoder = Decoder(latent_rep_size, max_length, charset_length)

    def forward(self, x, mode = 'autoencoder'):
        
        z_latent = self.encoder(x)
        decoded_string = self.decoder(z_latent)
        
        if mode == 'encoder':
            return z_latent
        elif mode == 'decoder':
            return self.decoder(x)
        else:
            return decoded_string
