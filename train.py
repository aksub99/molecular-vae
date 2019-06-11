from __future__ import print_function

import argparse
import os
import h5py
import numpy as np


NUM_EPOCHS = 1
BATCH_SIZE = 32
LATENT_DIM = 292
RANDOM_SEED = 1337

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
    optimizer  = optim.Adam(model.parameters(), lr = 1e-2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_batches = int(50000/args.batch_size)
    # scheduler = ReduceLROnPlateau(optimizer, 'min')
    for epoch in range(args.epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i in range(66):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = torch.from_numpy(data_train[i*args.batch_size:(i+1)*args.batch_size,]), torch.from_numpy(data_train[i*args.batch_size:(i+1)*args.batch_size,])
            inputs, labels = inputs.type(torch.FloatTensor), labels.type(torch.FloatTensor)
            inputs, labels = inputs.to(device), labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, vae_loss = model(inputs)
            loss = vae_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            # scheduler.step(loss)

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

    print('Finished Training')

if __name__ == '__main__':
    main()
