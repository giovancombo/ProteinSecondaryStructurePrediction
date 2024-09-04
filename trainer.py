# Deep Learning 2023 course, held by Professor Paolo Frasconi - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/ProteinSecondaryStructurePrediction

import torch
from tqdm import tqdm
import wandb


class Trainer:
    """
    A class that encapsulates the logic for training loops, evaluation, and metric tracking
    for a deep learning model designed for Protein Secondary Structure Prediction.

    Attributes:
        model (nn.Module): The neural network model to be trained.
        device (torch.device): The device (CPU or GPU) on which to perform the computations.
        epochs (int): The total number of training epochs.
        examples_ct (int): Counter for the number of protein elements processed during training.
    """

    def __init__(self, model, device, epochs = None):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.examples_ct = 0

    def train(self, trainloader, criterion, optimizer, log_freq, epoch):
        """
        Train the model in batches for one epoch.

        Args:
            trainloader (DataLoader): DataLoader for the training set.
            criterion (nn.Module): Loss function.
            optimizer (torch.optim): Optimizer.
            log_freq (int): Frequency of print and logging with Weights&Biases.
            epoch (int): Current epoch.

        Returns:
            float: Average training loss over the epoch.
            float: Average training accuracy over the epoch.
        """

        self.model.train()
        batches_ct, train_loss = 0, 0
        epoch_total, epoch_correct = 0, 0
        accuracies, losses = [], []
        for x_residues, x_pssm, y_targets, pad_mask in tqdm(trainloader, desc = f"Epoch {epoch+1}/{self.epochs} - Training"):
            correct, total = 0, 0
            x_residues = x_residues.to(self.device)
            x_pssm = x_pssm.to(self.device)
            pad_mask = pad_mask.to(self.device)
            y_targets_indices = torch.argmax(y_targets, dim=-1).to(self.device)

            optimizer.zero_grad(set_to_none=True)
            outputs = self.model(x_residues, x_pssm)
            loss = criterion(outputs.view(-1, outputs.size(-1)), y_targets_indices.view(-1))

            loss.backward()
            optimizer.step()

            batches_ct += 1
            train_loss += loss.item()

            _, pred = torch.max(outputs.data, -1)
            mask = pad_mask == 0
            correct += ((pred == y_targets_indices) & mask).sum().item()
            total += mask.sum().item()
            epoch_correct += correct
            epoch_total += total
            self.examples_ct += mask.sum().item()
            accuracy = 100.*(correct / total)
            epoch_accuracy = 100.*(epoch_correct / epoch_total)

            losses.append(loss.item())
            accuracies.append(accuracy)

            wandb.log({'Training/train_loss': loss.item(),
                'Training/train_accuracy': accuracy}, step = self.examples_ct)

            if batches_ct % log_freq == 0:
                #print(f"Current batch: {correct} correct over {total} elements >> Batch Accuracy: {accuracy:.4f}%")
                print(f"\nEpoch {epoch+1} so far: {epoch_correct} correct over {epoch_total} elements >> Epoch Accuracy: {epoch_accuracy:.4f}%")

        train_accuracy = sum(accuracies) / len(accuracies)
        train_loss = sum(losses) / len(losses)

        return train_loss, train_accuracy

    @torch.no_grad()
    def evaluate(self, testloader, criterion, epoch = None):
        """
        Evaluate a model on the validation/test set.

        Args:
            testloader (DataLoader): DataLoader for the validation/test set.
            criterion (nn.Module): Loss function.
            epoch (int, optional): Current epoch. If None, the function can be used for testing only.

        Returns:
            float: Average validation/test accuracy.
            float: Average validation/test loss.
            list: List of all predictions.
            list: List of all targets.
        """

        self.model.eval()
        total_loss = 0
        correct, total = 0, 0
        all_predictions, all_targets = [], []

        with torch.no_grad():
            for x_residues, x_pssm, y_targets, pad_mask in tqdm(testloader, desc = f"Evaluating epoch {epoch+1} on CB513 dataset: " if epoch is not None else "Testing Model on CB513 dataset: "):
                x_residues = x_residues.to(self.device)
                x_pssm = x_pssm.to(self.device)
                pad_mask = pad_mask.to(self.device)
                y_targets_indices = torch.argmax(y_targets, dim=-1).to(self.device)

                outputs = self.model(x_residues, x_pssm)
                loss = criterion(outputs.view(-1, outputs.size(-1)), y_targets_indices.view(-1))
                total_loss += loss.item()

                _, pred = torch.max(outputs.data, -1)
                mask = pad_mask == 0
                correct += ((pred == y_targets_indices) & mask).sum().item()
                total += mask.sum().item()

                all_predictions.extend(pred[mask].cpu().numpy())
                all_targets.extend(y_targets_indices[mask].cpu().numpy())

        test_accuracy = 100*(correct / total)
        test_loss = total_loss / len(testloader)

        return test_accuracy, test_loss, all_predictions, all_targets
