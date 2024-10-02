disease_name = 'sample/5000'
num_Labels =  100
data_path = f'/home/almusawiaf/MyDocuments/PhD_Projects/PSG_SURVIVAL_ANALYSIS/Data/{num_Labels}_Diagnoses/{disease_name}/HGNN_data'


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse
from sklearn.metrics import precision_score

# Define the GNN model
class GNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x

# Aggregation function
def aggregate_features(features_list, method='max'):
    if method == 'mean':
        return torch.mean(torch.stack(features_list), dim=0)
    elif method == 'concat':
        return torch.cat(features_list, dim=1)
    elif method == 'max':
        return torch.max(torch.stack(features_list), dim=0).values
    else:
        raise ValueError("Unsupported aggregation method")

# Attention mechanism to compute attention weights
class AttentionNetwork(nn.Module):
    def __init__(self, input_dim, num_meta_paths):
        super(AttentionNetwork, self).__init__()
        self.num_meta_paths = num_meta_paths
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(input_dim * 2, 1)) for _ in range(num_meta_paths)])
    
    def forward(self, f_meta):
        N = f_meta.shape[0]
        attention_weights = torch.zeros(N, N, self.num_meta_paths, device=f_meta.device)
        for i in range(N):
            for j in range(N):
                combined_features = torch.cat([f_meta[i], f_meta[j]], dim=0)
                attention_scores = []
                for weight in self.weights:
                    score = torch.exp(F.relu(torch.matmul(combined_features, weight)))
                    attention_scores.append(score)
                attention_scores = torch.stack(attention_scores, dim=-1)
                attention_weights[i, j] = attention_scores / torch.sum(attention_scores, dim=-1, keepdim=True)
        return attention_weights

# HSGNN model
class HSGNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_meta_paths, num_classes, aggregation_method='mean'):
        super(HSGNN, self).__init__()
        self.gnn_layers = nn.ModuleList([GNN(input_dim, output_dim) for _ in range(num_meta_paths)])
        self.attention_network = AttentionNetwork(output_dim, num_meta_paths)
        self.aggregation_method = aggregation_method
        self.final_gnn = GCNConv(output_dim, num_classes)  # Final GCN layer for classification

    def forward(self, features, similarity_matrices):
        N = features.shape[0]
        K = len(similarity_matrices)
        
        # Generate node features from each similarity matrix using GNNs
        features_list = []
        for k in range(K):
            edge_index = np.vstack(similarity_matrices[k].nonzero()).astype(np.int64)
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=features.device)
            x = features.clone().detach()
            features_list.append(self.gnn_layers[k](x, edge_index))
        
        # Aggregate features to get F_meta
        f_meta = aggregate_features(features_list, method=self.aggregation_method)
        
        # Compute attention weights using F_meta
        attention_weights = self.attention_network(f_meta)
        
        # Compute A_meta
        A_meta = torch.zeros(N, N, device=features.device)
        for k in range(K):
            A_dense = torch.tensor(similarity_matrices[k].toarray(), dtype=torch.float32, device=features.device)  # Convert sparse matrix to dense tensor
            A_meta += attention_weights[:, :, k] * A_dense  # Element-wise multiplication
        
        # Use the aggregated features and integrated adjacency matrix for final prediction
        edge_index = np.vstack(A_meta.nonzero()).astype(np.int64)
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=features.device)
        predictions = self.final_gnn(f_meta, edge_index)
        
        return f_meta, A_meta, predictions

# Function to compute micro and macro precision
def compute_precisions(predictions, labels):
    # Apply sigmoid to get probabilities and then threshold at 0.5
    preds = torch.sigmoid(predictions).cpu().detach().numpy()
    preds = (preds >= 0.5).astype(int)
    labels = labels.cpu().detach().numpy()
    
    micro_precision = precision_score(labels, preds, average='micro')
    macro_precision = precision_score(labels, preds, average='macro')
    
    return micro_precision, macro_precision

# Training and optimization process
def train_hsgnn(model, features, similarity_matrices, labels, epochs=100, lr=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    features = features.to(device)
    labels = torch.tensor(labels, dtype=torch.float32).to(device)
    similarity_matrices = [sm.tocoo() for sm in similarity_matrices]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for multi-label classification

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        f_meta, A_meta, predictions = model(features, similarity_matrices)

        loss = loss_fn(predictions, labels)

        loss.backward()
        optimizer.step()

        micro_precision, macro_precision = compute_precisions(predictions, labels)
        print(f"Epoch {epoch}, Loss: {loss.item()}, Micro Precision: {micro_precision}, Macro Precision: {macro_precision}")

# =================================================================================================

print(f'Saving to : {data_path}\n')

features = torch.load(f'{data_path}/X_32.pt')
K = 3
N = features.shape[0]
feature_dim = features.shape[1]
output_dim = 16  # Output dimension of GNN

print(N, feature_dim)
aggregation_method = 'max' # 'mean', 'concat'

# Load the similarity matrices
similarity_matrices = []
for k in range(K):
    similarity_matrices.append(scipy.sparse.load_npz(f'{data_path}/A/sparse_matrix_{k}.npz'))
    
# Ensure features is a PyTorch tensor
features = torch.tensor(features, dtype=torch.float32)

labels = torch.load(f'{data_path}/Y.pt')
num_classes = labels.shape[1]

# =======================================================================================================
def synthetic_data():
    # Sample data generation
    N = 100  # Number of nodes
    K = 3  # Number of similarity matrices
    feature_dim = 4  # Dimension of node features
    output_dim = 16  # Output dimension of GNN
    num_classes = 10  # Number of classes for multi-label classification

    # Generate random node features
    features = torch.randn(N, feature_dim)

    # Generate random similarity matrices
    similarity_matrices = []
    for k in range(K):
        matrix = np.random.rand(N, N)
        sparse_matrix = scipy.sparse.csr_matrix(matrix)
        similarity_matrices.append(sparse_matrix)

    # Generate random labels for multi-label classification
    labels = torch.randint(0, 2, (N, num_classes)).float()

    # Ensure features is a PyTorch tensor
    features = torch.tensor(features, dtype=torch.float32)
    
    return N, K, feature_dim, output_dim, num_classes, features, similarity_matrices, labels
# N, K, feature_dim, output_dim, num_classes, features, similarity_matrices, labels = synthetic_data()
# =======================================================================================================

# Initialize the HSGNN model
hs_gnn = HSGNN(feature_dim, output_dim, K, num_classes, aggregation_method='mean')

# Train the HSGNN model
train_hsgnn(hs_gnn, features, similarity_matrices, labels)

# Forward pass to compute F_meta and A_meta
hs_gnn.eval()
with torch.no_grad():
    f_meta, A_meta, predictions = hs_gnn(features, similarity_matrices)

# print("F_meta:", f_meta)
# print(f_meta.shape)

# print("A_meta:", A_meta)
# print(A_meta.shape)

print("Predictions:", predictions)
print(predictions.shape)
