import torch
import torch.nn as nn
from torch_geometric.data import Data

from gnn_model import GNNModel
from node_label import node_labels
from node_feature_builder import feature_extracter
from ppi_graph_builder import PPIGraphBuilder
class GNNtrainer:
    def __init__(self, model, optimizer, loss_function, device= torch.device('cuda')):
        """
        Trainer for GNN node classification.

        Parameters:
            model : nn.Module
                GNN model instance
            optimizer : torch.optim.Optimizer
                Training optimizer
            loss_function (nn.Module): nn.Module
                Loss function (CrossEntropyLoss)
            device
                Device to use for training
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device

        self.model.to(self.device) # Moves model to device

    def train(self, data):
        """
        Trains GNN

        Parameters:
            data : Tensor
                Data object describing graph
        
        Returns:
            loss : Scalar
                Model loss
        """
        self.model.train()
        self.optimizer.zero_grad() # Clears previous gradient

        # Moves data to device
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        y = data.y.to(self.device)
        mask = data.train_mask.to(self.device)

        out = self.model(x, edge_index) # Forward pass, out = logits
        loss = self.loss_function(out[mask], y[mask]) # Calculates loss

        loss.backward() # Computes gradient
        self.optimizer.step() # Applies gradient updates to weights

        return loss.item() # Returns loss as scaler

# Get edgeindex of all proteins
ppi_g_build = PPIGraphBuilder()
edge_index = ppi_g_build.merge_networks()

# Extract protein feature data
fe = feature_extracter()
x = fe.get_biological_features()

# Get actual labels for proteins
y = node_labels(x) # Need to drop lables data from features prior to training!

data = Data(x=x, edge_index=edge_index, y=y)

model = GNNModel(in_channels=data.num_node_features, hidden_channels=128, num_classes=2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()

trainer = GNNtrainer(model, optimizer, loss_function)
for epoch in range(1000):
    loss = trainer.train(data)