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
        Parameters:
            in_channels : int
                Number of features of input nodes
            out_channels : int
                Number of features of output nodes

        Returns:
            Tensor
                Updated nodes
        """
        super().__init__(aggr='add') # Add: Sums messages from neighbors (Aggregation)

        # Initializes linear transforer of layer
        self.lin = Linear(in_channels, out_channels, bias=False)
        # Initializes biases of layer
        self.bias = Parameter(torch.empty(out_channels))
        # Resets weights and bias of layer
        self.reset_parameters()

    def reset_parameters(self):
        """
        Sets initial learnable parameters
        """
        self.lin.reset_parameters() # Resets weights of linear transformation
        self.bias.data.zero_() # Sets the bias term to zero

    def forward(self, x, edge_index):
        """
        Forward pass of GNN layer

        Parameters;
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
        x = self.lin(x)

        # Normalization calc to prevent high connectivity dominating learning
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Propogate messages
        out = self.propagate(edge_index, x=x, norm=norm)

        # Apply bias vector
        out = out + self.bias
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
