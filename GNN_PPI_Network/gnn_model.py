import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GNNLayer(MessagePassing):
    """
    Graph Neural Network layer for risk modeling of protein-protein interactions.

    Each node represents a protein and edges represent the interactions between proteins.
    Inherits from the MessagePassing PyTorch Geometric base class to propagate information between
    nodes through the graph to determine how they influence each other.

    Inputs:
        x (Tensor): Node feature matrix of shape [num_nodes, in_channels].
        edge_index (LongTensor): Graph connectivity in COO format with shape [2, num_edges].

    Outputs:
        Tensor: Updated node embeddings of shape [num_nodes, out_channels].
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes GNN layer

        Parameters:
            in_channels : int
                Number of features of input nodes
            out_channels : int
                Number of features of output nodes
        """
        super().__init__(aggr="add")  # Add: Sums messages from neighbors (Aggregation)

        # Linear transformation applied to node features
        self.lin = Linear(in_channels, out_channels, bias=False)
        # Initializes biases of layer
        self.bias = Parameter(torch.empty(out_channels))
        # Resets weights and bias of layer
        self.reset_parameters()

    def reset_parameters(self):
        """
        Sets initial learnable parameters
        """
        self.lin.reset_parameters()  # Resets weights of linear transformation
        self.bias.data.zero_()  # Sets the bias term to zero

    def forward(self, x, edge_index):
        """
        Forward pass of GNN layer

        Parameters:
            x : Tensor
                Matrix of nodes features (Dismension are num_nodes X in_channels(features))
            edge_index : Tensor
                Connected nodes edge indices (Dimensions are 2 X num_edges)
                edge_index[0] = list of source node
                edge_index[1] = list of destination nodes

        Returns:
            Tensor
                Updated nodes
        """
        # Add self loop to include node's own features in calculation
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Performs linear transformation of feature vector
        x = self.lin(x)  # xi -> W*xi

        # Normalization calc to prevent high connectivity dominating learning
        row, col = (
            edge_index  # row = Source node [num_edges, ], col = target nodes [num_edges, ]
        )
        deg = degree(col, x.size(0), dtype=x.dtype)  # Degree of each node
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0  # Sets inf deg_inv_sqrt to 0
        norm = (
            deg_inv_sqrt[row] * deg_inv_sqrt[col]
        )  # Calc normalization value for each edge

        # Propogate messages
        # Internally class message(), aggregate(), and update()
        out = self.propagate(edge_index, x=x, norm=norm)

        # Apply bias vector
        out = out + self.bias
        return out

    def message(self, x_j, norm):
        """
        Compute messages sent along each edge.

        Parameters;
            x_j : Tensor
                Source node features for each edge. Shape [num_edges, out_channels]
            norm : Tensor
                Normalization value per edge

        Returns:
            Tensor
                Messages of shape [num_edges, out_channels]
        """
        return norm.view(-1, 1) * x_j


class GNNModel(nn.Module):
    """
    GNN Model for risk modeling of protein-protein interactions.
    """

    def __init__(self, in_channels, hidden_channels, out_classes, dropout=0.5):
        super().__init__()
        """
        Initializes GNN layers, classifier, and dropout layers

        Parameters:
            in_channels : int
                Number of features of input nodes
            hidden_channels : int
                Dimensionality of embedding
            out_classes: int
                Number of output classes
            dropout: int
                Dropout factor to be applied in dropout layer (prevents overfitting)
        """
        # Message passing layers (Defines information range: 1 layer = neighbors, 2 = neighbors of neighbors, ect.)
        self.gnn1 = GNNLayer(in_channels, hidden_channels)
        self.gnn2 = GNNLayer(hidden_channels, hidden_channels)
        self.gnn3 = GNNLayer(hidden_channels, hidden_channels)

        # Classifier: Converts embedding to class scores
        self.classifier = Linear(hidden_channels, out_classes)

        # Dropout layer: Sets some inputs to 0 to reduce overfitting
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        """
        Forward pass of GNN Model

        Parameters:
            x : Tensor
                Matrix of nodes features (Dismension are num_nodes X in_channels(features))
            edge_index : Tensor
                Connected nodes edge indices (Dimensions are 2 X num_edges)
                edge_index[0] = list of source node
                edge_index[1] = list of destination nodes

        Returns:
            logits : Tensor [num_nodes, num_classes]
                Classification (Raw score not yet a probability)
        """
        # First layer propogation
        x = self.gnn1(x, edge_index)
        x = F.relu(x)  # Activation function
        x = self.dropout(x)  # Dropout layer

        # Second layer propogation
        x = self.gnn2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Third layer propogation
        x = self.gnn3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Performs classification of each node
        logits = self.classifier(x)
        return logits
