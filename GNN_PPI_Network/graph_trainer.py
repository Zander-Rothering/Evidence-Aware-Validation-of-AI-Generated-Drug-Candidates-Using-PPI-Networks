import torch
import torch.nn as nn

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
    