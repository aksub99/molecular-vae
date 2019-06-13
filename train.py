from __future__ import print_function
import torch
import torch.nn as nn
import argparse
import os
import h5py
import numpy as np


NUM_EPOCHS = 1
BATCH_SIZE = 512
LATENT_DIM = 292
RANDOM_SEED = 1337

def vae_loss(x, x_decoded_mean, mean, logvar): # call this from forward
    x = x.view(x.size()[0], -1)
    x_decoded_mean = x_decoded_mean.view(x_decoded_mean.size()[0], -1)
    criterion = nn.BCELoss()
    bce_loss = criterion(x, x_decoded_mean)
    xent_loss = 120 * bce_loss
    k1_loss = - 0.5 * torch.mean(1 + logvar - torch.pow(mean, 2) - torch.exp(logvar))
    return xent_loss + k1_loss

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('data', type=str, help='The HDF5 file containing preprocessed data.')
    parser.add_argument('model', type=str,
                        help='Where to save the trained model. If this file exists, it will be opened and resumed.')
    parser.add_argument('--epochs', type=int, metavar='N', default=NUM_EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--batch_size', type=int, metavar='N', default=BATCH_SIZE,
                        help='Number of samples to process per minibatch during training.')
    parser.add_argument('--random_seed', type=int, metavar='N', default=RANDOM_SEED,
                        help='Seed to use to start randomizer for shuffling.')
    return parser.parse_args()

def main():
    args = get_arguments()
    np.random.seed(args.random_seed)

    from model import MoleculeVAE
    from utils import one_hot_array, one_hot_index, from_one_hot_array, \
        decode_smiles_from_indexes, load_dataset
    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    data_train, data_test, charset = load_dataset(args.data)
    model = MoleculeVAE(charset = charset, latent_rep_size = args.latent_dim)
    optimizer  = optim.Adam(model.parameters(), lr = 1e-3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_batches = int(40000/args.batch_size)
    # scheduler = ReduceLROnPlateau(optimizer, 'min')
    for epoch in range(args.epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i in range(num_batches):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = torch.from_numpy(data_train[i*args.batch_size:(i+1)*args.batch_size,]), torch.from_numpy(data_train[i*args.batch_size:(i+1)*args.batch_size,])
            inputs, labels = inputs.type(torch.FloatTensor), labels.type(torch.FloatTensor)
            inputs, labels = inputs.to(device), labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, mean, logvar = model.forward(inputs)
            
            # Debugging statements
            if i==0:
              inp = inputs.cpu().numpy()
              outp = outputs.cpu().detach().numpy()
              lab = labels.cpu().numpy()
              print("Input:")
              print(decode_smiles_from_indexes(map(from_one_hot_array, inp[0]), charset))
              print("Label:")
              print(decode_smiles_from_indexes(map(from_one_hot_array, lab[0]), charset))
              sampled = outp[0].reshape(1, 120, len(charset)).argmax(axis=2)[0]
              print("Output:")
              print(decode_smiles_from_indexes(sampled, charset))
              
            loss = vae_loss(outputs, labels, mean, logvar)
            loss.backward()
            optimizer.step()
            # scheduler.step(loss)

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

    print('Finished Training')
