# Deep Learning 2023 course, held by Professor Paolo Frasconi - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/ProteinSecondaryStructurePrediction

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import time
import yaml
import wandb
import random

from trainer import Trainer
from model import ProteinTransformer
from losses import FocalLoss, CombinedLoss
from my_utils import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Loading the .yaml configuration file
with open("params.yaml", 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Extracting parameters from the configuration file
TEST_ONLY = config['TEST_ONLY']
LOADED_PATH = config['LOADED_PATH']
TRAIN_PATH = config['TRAIN_PATH']
TEST_PATH = config['TEST_PATH']

filter_proteins_by_length = config['filter_proteins_by_length']
truncate_proteins = config['truncate_proteins']
min_len = config['min_len']
max_len = config['max_len']

set_seed(config['seed'])

aminoacids = 21
classes = 9
protein_len = 700
total_features = 57

embd_dim = config['embd_dim'] - aminoacids
n_heads = config['n_heads']
n_layers = config['n_layers']
clf_hid_dim = config['clf_hid_dim']

batch_size = config['batch_size']
epochs = config['epochs']
gamma_scheduler = config['gamma_scheduler']
learning_rate = config['learning_rate']
weight_decay = config['weight_decay']
dropout = config['dropout']
softmax_temperature = config['softmax_temperature']

loss_function = config['loss_function']
label_smoothing = config['label_smoothing']
focalloss_alpha = config['focalloss_alpha']
focalloss_gamma = config['focalloss_gamma']
ce_weight = config['ce_weight']
focal_weight = config['focal_weight']
gradient_clipping = config['gradient_clipping']
max_grad_norm = config['max_grad_norm']
max_relative_position = config['max_relative_position']

wandb_log = config['wandb_log']
log_freq = config['log_freq']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Creating a unique-named directory to save checkpoints
folder = f"embd{embd_dim+aminoacids}_bs{batch_size}_lr{learning_rate}_{n_heads}heads_{n_layers}layers_dr{dropout}_wd{weight_decay}_ls{label_smoothing}_gamma{gamma_scheduler}"
directory = os.path.join('checkpoints', folder) + str(time.time())[-4:]
os.makedirs(directory, exist_ok=True)


if __name__ == "__main__":
    # Loading the raw data and converting it to PyTorch tensors
    cullpdb_data = torch.from_numpy(np.load(TRAIN_PATH))            # (3880, 700, 51)
    cb513_data = torch.from_numpy(np.load(TEST_PATH))               # (513, 700, 51)

    if filter_proteins_by_length:
        filtered_cullpdb_data = filter_protein_dataset(cullpdb_data, min_len, max_len)

        print(f"Dimensioni originali cullpdb_data: {cullpdb_data.shape}")
        print(f"Dimensioni filtrate cullpdb_data: {filtered_cullpdb_data.shape}")
        print(f"Dimensioni originali cb513_data: {cb513_data.shape}")

        cullpdb_data = filtered_cullpdb_data

    if truncate_proteins:
        cullpdb_data = cullpdb_data[:, min_len:max_len, :]

    # One-hot and integer aminoacid residues
    train_residues_1h = cullpdb_data[:,:,:21].type(torch.float)     # (3880, 700, 21)
    test_residues_1h = cb513_data[:,:,:21].type(torch.float)        # (513, 700, 21)
    train_residues_int = onehot_to_int(train_residues_1h)           # (3880,700)
    test_residues_int = onehot_to_int(test_residues_1h)             # (513,700)

    # PSSMs: analyzing PSSM data reveals that the values are correctly normalized between 0 and 1
    train_pssm = cullpdb_data[:,:,30:].type(torch.float)            # (3880, 700, 21)
    test_pssm = cb513_data[:,:,30:].type(torch.float)               # (513, 700, 21)
    # analyze_pssm(train_pssm)
    # analyze_pssm(test_pssm)

    # One-hot secondary structure targets
    train_targets = cullpdb_data[:,:,21:30]                         # (3880, 700, 9)
    test_targets = cb513_data[:,:,21:30]                            # (513, 700, 9)

    # Padding masks for the transformer
    train_padding_mask = create_padding_mask(train_residues_1h)     # (3880, 700)
    test_padding_mask = create_padding_mask(test_residues_1h)       # (513, 700)

    # Creating the PyTorch datasets
    train_set = TensorDataset(train_residues_int, train_pssm, train_targets, train_padding_mask)
    test_set = TensorDataset(test_residues_int, test_pssm, test_targets, test_padding_mask)

    # Creating the dataloaders
    g = torch.Generator()
    g.manual_seed(config['seed'])
    trainloader = DataLoader(train_set, batch_size = batch_size, shuffle = True, worker_init_fn=seed_worker, generator=g)
    testloader = DataLoader(test_set, batch_size = batch_size, shuffle = False)

    # Instantiating the model, optimizer, scheduler and criterion
    model = ProteinTransformer(aminoacids, embd_dim, classes, n_heads, n_layers, max_relative_position, clf_hid_dim, dropout, softmax_temperature, device, seed = config['seed']).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of model parameters: ", n_params)

    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    # scheduler = ExponentialLR(optimizer, gamma = gamma_scheduler)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor = gamma_scheduler, patience = 1)

    if loss_function == "focal":
        criterion = FocalLoss(alpha = focalloss_alpha, gamma = focalloss_gamma, ignore_index = 8)
    elif loss_function == "crossentropy":
        criterion = nn.CrossEntropyLoss(ignore_index = 8, label_smoothing = label_smoothing)
    elif loss_function == "combined":
        criterion = CombinedLoss(alpha = focalloss_alpha, gamma = focalloss_gamma, ce_weight = ce_weight, focal_weight = focal_weight, ignore_index = 8)

    # Instantiating a Trainer object for training and evaluation
    trainer = Trainer(model, device, gradient_clipping, epochs)

    if TEST_ONLY:
        # Testing a loaded model on the CB513 dataset
        loaded_model, _, epoch, _ = load_pretrained_model(LOADED_PATH, model, device)
        print(f"\nTesting loaded model from epoch {epoch} on the CB513 dataset:")
        test_model(loaded_model, testloader, criterion, device, f'images/{directory}')

    else:
        # Training and validating model over multiple epochs
        if wandb_log:
            wandb.login()
            wandb.init(project = 'DL23_PSSP', config = config, name = folder)
            config = wandb.config
            wandb.watch(model, criterion, log = "all", log_freq = log_freq)

        best_val_accuracy = 0
        for epoch in range(epochs):
            train_loss, train_accuracy = trainer.train(trainloader, criterion, optimizer, log_freq, max_grad_norm, epoch, wandb_log)
            val_accuracy, val_loss, val_predictions, val_targets = trainer.evaluate(testloader, criterion, epoch)
            examples_ct = trainer.examples_ct

            if wandb_log:
                wandb.log({'train_loss': train_loss,
                        'train_accuracy': train_accuracy,
                        'validation_loss': val_loss,
                        'validation_accuracy': val_accuracy,
                        'epoch': epoch}, step = examples_ct)

            print(f"End of Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Q8 Accuracy: {train_accuracy:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Q8 Accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                save_checkpoint(epoch, model, optimizer, best_val_accuracy, directory, is_best=True)
                print("New best model saved!")

            # Saving regular checkpoint
            save_checkpoint(epoch, model, optimizer, best_val_accuracy, directory)
            print(f"Checkpoint saved for epoch {epoch+1}")

            if scheduler:
                scheduler.step(val_loss)

        print(f"Best Val Accuracy: {best_val_accuracy:.4f}")

        # Testing the best model on the CB513 dataset for computing final metrics
        print("Testing the best model on the CB513 dataset:")
        os.makedirs(f'images/{directory}', exist_ok=True)
        best_model, _, epoch, _ = load_pretrained_model(os.path.join(directory, 'bestmodel.pth'), model)
        test_model(best_model, testloader, criterion, device, f'images/{directory}')

        if wandb_log:
            wandb.log({'num model parameters': n_params})
            wandb.unwatch(model)
            wandb.finish()
