import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.sparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, f1_score
from sklearn.metrics import classification_report, average_precision_score
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix 
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

import torch
from torch_geometric.data import Data
from torch.optim import Adam
from torch_geometric.utils import from_scipy_sparse_matrix

from torch_geometric.nn import GCNConv
import torch.nn as nn





def load_dict_from_pickle(filename):
    with open(filename, 'rb') as file:
        loaded_dict = pickle.load(file)
    return loaded_dict

# Define the GCN model with one layer
class OneLayerGCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(OneLayerGCN, self).__init__()
        self.conv = GCNConv(input_dim, output_dim)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight=edge_weight)
        return x


def plot_combined_metrics_and_loss(losses, val_metrics):
    epochs = range(1, len(losses) + 1)
    # val_precisions = [metrics['precision'] for metrics in val_metrics]
    val_recalls = [metrics['recall'] for metrics in val_metrics]
    val_accuracies = [metrics['accuracy'] for metrics in val_metrics]
    val_f1_scores = [metrics['f1_score'] for metrics in val_metrics]
    val_aucs = [metrics['auc'] for metrics in val_metrics]
    val_micro_precisions = [metrics['micro_precision'] for metrics in val_metrics]
    val_macro_precisions = [metrics['macro_precision'] for metrics in val_metrics]

    fig, ax1 = plt.subplots(figsize=(15, 8))

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, losses, label='Loss', marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Metrics', color=color)  # we already handled the x-label with ax1
    # ax2.plot(epochs, val_precisions, label='Validation Precision', marker='o', color='tab:orange')
    ax2.plot(epochs, val_recalls, label='Recall', marker='o', color='tab:green')
    ax2.plot(epochs, val_accuracies, label='Accuracy', marker='o', color='tab:purple')
    ax2.plot(epochs, val_f1_scores, label='F1 Score', marker='o', color='tab:brown')
    ax2.plot(epochs, val_aucs, label='AUC', marker='o', color='tab:pink')
    ax2.plot(epochs, val_micro_precisions, label='Micro Precision', marker='o', color='tab:cyan')
    ax2.plot(epochs, val_macro_precisions, label='Macro Precision', marker='o', color='magenta')  # changed here
    ax2.tick_params(axis='y', labelcolor=color)

    # Adding the title and legend
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.suptitle('Training Loss and Validation Metrics', fontsize=16)
    fig.subplots_adjust(top=0.88)
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))

    plt.show()



def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        # Get model predictions
        logits = model(data.x, data.edge_index, data.edge_attr)
        preds = torch.sigmoid(logits[mask])
        binary_preds = (preds > 0.5).float()
        
        # Ground truth
        true_labels = data.y[mask]
        
        # Convert to CPU and numpy for sklearn metrics
        true_labels_np = true_labels.cpu().numpy()
        binary_preds_np = binary_preds.cpu().numpy()
        preds_np = preds.cpu().numpy()
        
        # Calculate accuracy
        accuracy = accuracy_score(true_labels_np, binary_preds_np)
        
        # Calculate Hamming loss
        hamming = hamming_loss(true_labels_np, binary_preds_np)
        
        # Calculate precision, recall, F1 score for micro and macro averaging
        precision_micro = precision_score(true_labels_np, binary_preds_np, average='micro', zero_division=0)
        recall_micro = recall_score(true_labels_np, binary_preds_np, average='micro', zero_division=0)
        f1_micro = f1_score(true_labels_np, binary_preds_np, average='micro', zero_division=0)
        
        precision_macro = precision_score(true_labels_np, binary_preds_np, average='macro', zero_division=0)
        recall_macro = recall_score(true_labels_np, binary_preds_np, average='macro', zero_division=0)
        f1_macro = f1_score(true_labels_np, binary_preds_np, average='macro', zero_division=0)
        
        # Calculate AUC only if there are both positive and negative samples for each label
        try:
            auc = roc_auc_score(true_labels_np, preds_np, average='macro', multi_class='ovr') if len(np.unique(true_labels_np)) > 1 else 0
        except ValueError:
            auc = 0
        
        return {
            'accuracy': accuracy,
            # 'hamming_loss': hamming,
            'micro_precision': precision_micro,
            # 'recall_micro': recall_micro,
            # 'f1_micro': f1_micro,
            'macro_precision': precision_macro,
            'recall': recall_macro,
            'f1_score': f1_macro,
            'auc': auc
        }



def reading_pickle(n):
    # print(f'Reading {n}')
    with open(f'{n}', 'rb') as f:
        data = pd.read_pickle(f)
    # print('\tDone reading...')
    return data

 

def impute_nans_with_col_mean(X):
    """
    Imputes NaNs in the given tensor with the column mean.
    
    Parameters:
    X (torch.Tensor): The input tensor with potential NaN values.
    
    Returns:
    torch.Tensor: The tensor with NaNs imputed by the column mean.
    """
    col_mean = torch.nanmean(X, dim=0)
    inds = torch.where(torch.isnan(X))
    X = X.clone()
    X[inds] = torch.take(col_mean.detach(), inds[1])
    return X

def impute_nans_with_mean(tensor):
    """
    Imputes NaNs in the given tensor with the mean of the tensor.
    
    Parameters:
    tensor (torch.Tensor): The input tensor with potential NaN values.
    
    Returns:
    torch.Tensor: The tensor with NaNs imputed by the mean.
    """
    mean_value = torch.nanmean(tensor)
    tensor = tensor.clone()
    tensor[torch.isnan(tensor)] = mean_value
    return tensor

def check_for_nans(tensor, tensor_name):
    if tensor is None:
        print(f"{tensor_name} is None")
        return True
    if torch.isnan(tensor).any():
        print(f"NaN detected in {tensor_name}")
        print(tensor)
        return True
    return False

def convert_dense_adj_to_edge_index_and_weight(adj):
    edge_index = torch.nonzero(adj, as_tuple=False).t()
    edge_weight = adj[edge_index[0], edge_index[1]]
    return edge_index, edge_weight

def load_data(file_path, device, with_SNF, super_class, num_Meta_Path=5):
    num_As = num_Meta_Path
    if with_SNF:
        num_As += 1
    # Load data, assuming the paths are correct
    A_meta = torch.load(f'{file_path}/A_meta.pt')
    X = torch.load(f'{file_path}/f_meta.pt')
    Y = torch.load(f'{file_path}/Y{super_class}.pt')
    
    # Convert A_meta to edge_index
    edge_index, edge_weight = convert_dense_adj_to_edge_index_and_weight(A_meta)

    # Reading patient information...
    Nodes = load_dict_from_pickle(f'{file_path}/Nodes.pkl')
    patient_indices = [i for i, node in enumerate(Nodes) if node[0] == 'C']
    num_patients = len(patient_indices)
    total_nodes = len(Nodes)
    del Nodes

    # Convert to tensors and move to the correct device
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float).to(device)
    else:
        X = X.to(device)

    if isinstance(Y, np.ndarray):
        Y = torch.tensor(Y, dtype=torch.float).to(device)
    else:
        Y = Y.to(device)

    # Debugging prints to check for NaNs
    print("X before NaN check:", X)
    print("Y before NaN check:", Y)
    print("Edge weights before NaN check:", edge_weight)

    # Handle NaNs in X, Y, and edge_weight
    X = impute_nans_with_col_mean(X)
    Y = impute_nans_with_col_mean(Y)
    edge_weight = impute_nans_with_mean(edge_weight)

    # Check for remaining NaNs in data
    if check_for_nans(X, "X") or check_for_nans(Y, "Y") or check_for_nans(edge_weight, "edge_weight"):
        raise ValueError("NaN detected in input data.")

    return X, Y, edge_index, edge_weight, patient_indices, total_nodes



def prepare_masks(patient_indices, total_nodes, test_size):
    # Split patient indices into train, val, and test
    train_index, temp_index = train_test_split(patient_indices, test_size=test_size, random_state=42)
    val_index, test_index = train_test_split(temp_index, test_size=0.6667, random_state=42)  # 30% into 20% and 10%

    # Initialize masks for all nodes
    train_mask = torch.zeros(total_nodes, dtype=torch.bool)
    val_mask = torch.zeros(total_nodes, dtype=torch.bool)
    test_mask = torch.zeros(total_nodes, dtype=torch.bool)

    # Set mask values to True for patient nodes
    train_mask[train_index] = True
    val_mask[val_index] = True
    test_mask[test_index] = True

    return train_mask, val_mask, test_mask

def train(model, data, train_mask, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def save_results(losses, val_metrics, file_path):
    """
    Saves the training loss and various validation metrics over epochs to a CSV file.

    Parameters:
    losses (list of float): Training losses over epochs.
    val_metrics (list of dict): Validation metrics over epochs. Each dict contains keys 'precision', 'recall',
                                'accuracy', 'f1_score', and 'auc'.
    file_path (str): The path where the CSV file will be saved.
    """
    if isinstance(losses[0], torch.Tensor):
        losses = [loss.item() for loss in losses]  # Converts each tensor to a scalar and ensures it's not on CUDA
    
    # Extract metrics from val_metrics
    # val_precisions = [metric["precision"] for metric in val_metrics]
    val_macro_precisions = [metric["macro_precision"] for metric in val_metrics]
    val_micro_precisions = [metric["micro_precision"] for metric in val_metrics]
    val_recalls = [metric["recall"] for metric in val_metrics]
    val_accuracies = [metric["accuracy"] for metric in val_metrics]
    val_f1_scores = [metric["f1_score"] for metric in val_metrics]
    val_aucs = [metric["auc"] for metric in val_metrics]
    
    # Create a DataFrame with all the metrics
    df = pd.DataFrame({
        'Loss': losses,
        'Validation Micro Precision': val_micro_precisions,
        'Validation Macro Precision': val_macro_precisions,
        'Validation Recall': val_recalls,
        'Validation Accuracy': val_accuracies,
        'Validation F1-Score': val_f1_scores,
        'Validation AUC': val_aucs
    })

    # Save the DataFrame to a CSV file
    df.to_csv(f'{file_path}.csv', index=False)


def init_weights(m):
    """
    Initialize weights of the model layers.

    Parameters:
    m (torch.nn.Module): The module (layer) to initialize.
    """
    if isinstance(m, GCNConv):
        for param in m.parameters():
            if param.requires_grad and param.dim() >= 2:
                torch.nn.init.kaiming_uniform_(param, nonlinearity='relu')
            elif param.requires_grad and param.dim() < 2:
                torch.nn.init.zeros_(param)
    elif isinstance(m, torch.nn.Linear):
        if m.weight is not None and m.weight.dim() >= 2:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None and m.bias.dim() >= 1:
            torch.nn.init.zeros_(m.bias)
    # Add support for other layers as needed
    elif isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
        if m.weight is not None and m.weight.dim() >= 2:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None and m.bias.dim() >= 1:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.BatchNorm2d):
        if m.weight is not None and m.weight.dim() >= 1:
            torch.nn.init.ones_(m.weight)
        if m.bias is not None and m.bias.dim() >= 1:
            torch.nn.init.zeros_(m.bias)
    # Example for layers needing Xavier initialization
    elif isinstance(m, torch.nn.Embedding):
        if m.weight is not None and m.weight.dim() >= 2:
            torch.nn.init.xavier_uniform_(m.weight)
    
def main(file_path, 
         GNN_Model, 
         num_epochs, 
         with_SNF = False, 
         lr=1e-5, 
         exp_name = 'emb_result',
         super_class = '',
         num_Meta_Path = 5):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'\t- Loading the data...\n{file_path}')
    X, Y, edge_index, edge_weight, patient_indices, total_nodes = load_data(file_path, device, with_SNF, super_class, num_Meta_Path=num_Meta_Path)
    
    print('\t- Generating the Data structure ...')
    data = Data(x=X, y=Y, edge_index=edge_index, edge_attr=edge_weight) 
    
    if check_for_nans(data.x, "data.x"):
        raise ValueError("NaNs detected in input features")
    if check_for_nans(data.edge_index, "data.edge_index"):
        raise ValueError("NaNs detected in edge index")
    if check_for_nans(data.edge_attr, "data.edge_attr"):
        raise ValueError("NaNs detected in edge weight matrix")

    print('\t- Generating the train, test, and validation sets...')
    
    test_size = 0.4
    train_mask, val_mask, test_mask = prepare_masks(patient_indices, total_nodes, test_size)
    data.train_mask = train_mask.to(device)
    data.val_mask = val_mask.to(device)
    data.test_mask = test_mask.to(device)

    num_features = X.size(1)
    num_classes = Y.size(1) 

    print(f'\t- Preparing the model...')
    model = GNN_Model(num_features, num_classes).to(device)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=5e-4, momentum=0.9)
        
    criterion = torch.nn.BCEWithLogitsLoss()

    losses, val_precisions = [], []

    model.apply(init_weights)  # Apply weight initialization

    print('\t- Training...')
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        if check_for_nans(out, "model output"):
            print(out)
            break
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        if check_for_nans(loss, "loss"):
            break

        loss.backward(retain_graph=True)  # Add retain_graph=True only if you need to reuse the graph
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        val_precision = evaluate(model, data, data.val_mask)
        current_precision = val_precision['macro_precision']
        current_accuracy = val_precision['accuracy']
        
        print(f'Epoch: {epoch}/{num_epochs}, Loss: {loss.item():.4f}, Val Precision: {current_precision:.4f}, ACC: {current_accuracy:.4f}')

        losses.append(loss.item())
        val_precisions.append(val_precision)

    test_precision = evaluate(model, data, data.test_mask)
    print(f'Test Precision: \n{test_precision}')

    # Extract and save embeddings
    print('\t- Extracting and saving embeddings...')
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index, data.edge_attr)
    embeddings_np = embeddings.cpu().numpy()  # Convert to NumPy array
    # Save embeddings to a .npy file
    np.save(f'{file_path}/{exp_name}.npy', embeddings_np)
    
    # Predictions for the test set
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)
        pred = (out[data.test_mask] > 0).float().cpu()  # Binarize the output (assuming binary classification)
        correct = data.y[data.test_mask].cpu()

    labels = [f'Class {i}' for i in range(num_classes)]  # Replace with actual class names if available
    create_multilabel_confusion_matrix(pred, correct, labels)

    return losses, val_precisions









def create_multilabel_confusion_matrix(pred, correct, labels):
    # Compute the classification report
    classification_rep = classification_report(correct, pred, target_names=labels, zero_division=0)
    print(classification_rep)

    report_as_dictionary = classification_report(correct, pred, target_names=labels, output_dict=True, zero_division=0)
    print(report_as_dictionary)


