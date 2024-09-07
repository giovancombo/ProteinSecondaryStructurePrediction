# Deep Learning 2023 course, held by Professor Paolo Frasconi - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/ProteinSecondaryStructurePrediction

import os
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

from trainer import Trainer


def onehot_to_int(onehot):
    """
    Convert a one-hot tensor to an integer tensor.

    Args:
        onehot (Tensor): One-hot tensor to convert.

    Returns:
        Tensor: Integer tensor.
    """

    target = torch.zeros((onehot.shape[0], onehot.shape[1]), dtype = torch.long)
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            target[i,j] = torch.argmax(onehot[i,j,:])
    return target


def create_padding_mask(data):
    """
    Create a boolean padding mask where True indicates a 'NoSeq' amino acid (padding) and False indicates
    a real amino acid. The 'NoSeq' amino acid is identified by the 21st feature (index 20) in each
    position of the input tensor.

    Args:
        data (Tensor): Input data tensor.

    Returns:
        Tensor: Padding mask for the input data tensor.
    """

    padding_mask = torch.zeros((data.shape[0], data.shape[1]), dtype = torch.bool)
    for i in range(data.shape[0]):
        padding_mask[i,:] = data[i,:,20]
    return padding_mask


def test_model(model, testloader, criterion, device, save_dir = None):
    """
    Test the model on the given testloader and save metrics.

    Args:
        model (nn.Module): The model to test.
        testloader (DataLoader): DataLoader for the test set.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the model on.
        save_dir (str, optional): Directory to save metrics plots.

    Returns:
        tuple: Test accuracy and test loss.
    """

    trainer = Trainer(model, device)
    test_accuracy, test_loss, test_predictions, test_targets = trainer.evaluate(testloader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    class_names = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
    report = classification_report(test_targets, test_predictions, target_names=class_names, zero_division=1)
    print(report)

    if save_dir is not None:
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        save_metrics_plots(test_targets, test_predictions, class_names, save_dir) 

    return test_accuracy, test_loss


def save_checkpoint(epoch, model, optimizer, best_accuracy, directory, is_best = False):
    """
    Saves a checkpoint of the model in a given directory, with the current epoch and optimizer state.
    If the model has the best accuracy so far, it also saves it as the best model.

    Args:
        epoch (int): Current epoch.
        model (nn.Module): Model to save.
        optimizer (Optimizer): Optimizer to save.
        best_accuracy (float): Best accuracy so far.
        directory (str): Directory to save the checkpoint.
        is_best (bool, optional): Whether the model has the best accuracy so far.

    Returns:
        None
    """

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_accuracy
    }
    if is_best:
        torch.save(checkpoint, os.path.join(directory, f'bestmodel.pth'))
    torch.save(checkpoint, os.path.join(directory, f'checkpoint_ep{epoch + 1}.pth'))


def load_pretrained_model(model_path, model, optimizer = None):
    """
    Load a pretrained model from a given path and return it for further training or testing.

    Args:
        model_path (str): Path to the model checkpoint.
        model (nn.Module): Model to load the state_dict into.
        optimizer (Optimizer, optional): Optimizer to load the state_dict into.

    Returns:
        tuple: Loaded model, optimizer, epoch of training and best accuracy.
    """

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']

    return model, optimizer, epoch, best_accuracy


def analyze_pssm(pssm_tensor):
    """
    Analyze the PSSM tensor and print stats about it to check normalization of values between 0 and 1.

    Args:
        pssm_tensor (Tensor): PSSM tensor to analyze.

    Returns:
        None
    """

    pssm_np = pssm_tensor.numpy()
    print(f"Shape: {pssm_np.shape}")
    print(f"Min value: {np.min(pssm_np)}")
    print(f"Max value: {np.max(pssm_np)}")
    print(f"Mean: {np.mean(pssm_np)}")
    print(f"Median: {np.median(pssm_np)}")
    print(f"Standard deviation: {np.std(pssm_np)}")
    
    # Controlla se tutti i valori sono tra 0 e 1
    if np.all((pssm_np >= 0) & (pssm_np <= 1)):
        print("All values are between 0 and 1")
    else:
        print("Some values are outside the [0, 1] range")
    
    # Calcola e stampa i percentili
    percentiles = np.percentile(pssm_np, [0, 25, 50, 75, 100])
    print(f"Percentiles [0, 25, 50, 75, 100]: {percentiles}")
    
    plt.hist(pssm_np.flatten(), bins=50)
    plt.title("PSSM Values Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


def save_metrics_plots(y_true, y_pred, class_names, save_dir):
    """
    Save metrics plots for the given tested data.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        class_names (list): List of class names.
        save_dir (str): Directory to save the plots.

    Returns:
        None
    """

    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    specificity = []
    for i in range(len(class_names)):
        tn = np.sum(cm) - (np.sum(cm[i,:]) + np.sum(cm[:,i]) - cm[i,i])
        fp = np.sum(cm[:,i]) - cm[i,i]
        specificity.append(tn / (tn + fp))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

    # Plot metrics
    metrics = ['Precision', 'Recall', 'Specificity', 'F1']
    metric_values = [precision, recall, specificity, f1]

    for metric, values in zip(metrics, metric_values):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=class_names, y=values)
        plt.title(f'{metric} per Class')
        plt.xlabel('Class')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{metric.lower()}_per_class.png'))
        plt.close()

    
def count_amino_acids(data):
    """
    Count the occurrences of each amino acid in the given data tensor.

    Args:
        data (Tensor): Data tensor to count amino acids in.

    Returns:
        ndarray: Array of amino acid counts.
    """

    amino_acid_counts = torch.sum(data[:, :, :20], dim=(0, 1)).cpu().numpy()
    return amino_acid_counts


def filter_protein_dataset(data, min_length = 50, max_length = 600):
    """
    Filter out proteins with length less than min_length or greater than max_length.

    Args:
        data (Tensor): Protein dataset tensor.
        min_length (int, optional): Minimum protein length.
        max_length (int, optional): Maximum protein length.

    Returns:
        Tensor: Filtered protein dataset
    """

    protein_lengths = torch.sum(data[:, :, :20].sum(axis = 2) != 0, axis = 1)
    mask = (protein_lengths >= min_length) & (protein_lengths <= max_length)
    filtered_data = data[mask]
    
    return filtered_data


def load_raw_data(path, seq_len, features):
    """
    Function used only the first time I loaded the raw data to reshape it and filter out proteins with
    'X' aminoacid.

    Args:
        path (str): Path to the raw data, the .npy file downloaded from the internet.
        seq_len (int): Desired sequence length to reshape the data.
        features (int): Desired number of features to reshape the data.

    Returns:
        ndarray: Filtered and reshaped raw data.
    """

    raw_data = np.load(path)
    raw_data = np.reshape(raw_data, (-1, seq_len, features))

    # filtered = tensor with proteins without 'X' aminoacid: we save it for later load (its creation needed more than 10 minutes)
    filtered = raw_data
    for i in range(raw_data.shape[0]-1, -1, -1):
        if np.unique(raw_data[i,:,20]).shape[0] != 1:
            print("found!: ", i)
            filtered = np.delete(filtered, i, axis = 0)

    # Removing features about solubility: (3880, 700, 51)
    filtered = np.delete(filtered, [20,31,32,33,34,54], axis = 2)
    # np.save("data/cb513+profile_split1_FINAL.npy", filtered)

    return filtered
