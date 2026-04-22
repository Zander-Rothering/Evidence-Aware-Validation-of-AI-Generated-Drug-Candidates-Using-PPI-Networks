import torch
import torch.nn as nn
from torch_geometric.transforms import RandomLinkSplit


class GNNtrainer:
    """
    Trainer for GNN node classification.
    """

    def __init__(self, model, optimizer, loss_function, device=None):
        """
        Initializes GNN Trainer

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
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model.to(self.device)  # Moves model to device

    def data_train_test_split(self, data, train_split=0.7, val_split=0.1):
        """
        Splits data into training, validation, and test sets

        Parameters:
            data : Tensor
                Data object describing graph
            train_split : float
                Split of data to be used for training
            val_split : float
                Split of data to be used for validation

        Returns:
            data : Tensor
                Data object with train, validation, and test split
        """
        # Get number of nodes in data
        num_nodes = data.num_nodes
        # Randomize the nodes index
        random_nodes = torch.randperm(num_nodes)

        # Get splits
        train_split_end = int(train_split * num_nodes)
        val_split_end = int((train_split + val_split) * num_nodes)

        # Assign index groups
        train_idx = random_nodes[:train_split_end]
        val_idx = random_nodes[train_split_end:val_split_end]
        test_idx = random_nodes[val_split_end:]

        # Create masks for data split
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        # Fill masks with true for each masks index group
        data.train_mask[train_idx] = (
            True  # All train index nodes have train mask of true
        )
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True

        return data

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
        self.optimizer.zero_grad()  # Clears previous gradient

        # Moves data to device
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        y = data.y.to(self.device)
        mask = data.train_mask.to(self.device)

        out = self.model(x, edge_index)  # Forward pass, out = logits
        loss = self.loss_function(out[mask], y[mask])  # Calculates loss

        loss.backward()  # Computes gradient (backpropagation)
        self.optimizer.step()  # Applies gradient updates to weights

        return loss.item()  # Returns loss as scaler

    def evaluate(self, data, eval_mask="val"):
        """
        Performs evaluation of model by calculating loss and accuracy of model

        Parameters:
            data : Tensor
                Data object describing graph
            eval_mask : str
                Split of data to use for evaluation

        Returns:
            loss : Scalar
                Model loss
            accuracy : float
                Portion of total nodes with correct prediction
        """
        # Move data to device
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        y = data.y.to(self.device)

        # Get mask to use for predictions
        if eval_mask == "train":
            mask = data.train_mask
        elif eval_mask == "val":
            mask = data.val_mask
        else:
            mask = data.test_mask

        # Move mask to device
        mask = mask.to(self.device)

        # Model predictions for each node
        score = self.model(x, edge_index)

        # Calculate loss for nodes using mask
        loss = self.loss_function(score[mask], y[mask])
        # Predicted class for each node
        pred = score.argmax(dim=1)

        # Calculate model accuracy
        correct_pred = (pred[mask] == y[mask]).sum().item()
        total_nodes = mask.sum().item()
        accuracy = correct_pred / total_nodes

        return loss.item(), accuracy

    def test_model(self, data, epochs=10000):
        """
        Performs evaluation of model by calculating loss and accuracy of model

        Parameters:
            data : Tensor
                Data object describing graph
            epochs : ints
                Number of epochs to train the data for

        Returns:
            test_loss : float
                Model test loss
            test_accuracy : float
                Model test accuracy
        """
        best_val_acc = 0.0

        # Split data into train, val, and test
        data = self.data_train_test_split(data)

        # Loop through epochs to train model
        for epoch in range(1, epochs + 1):
            # Train model on training data and return loss
            train_loss = self.train(data)
            # Evaluate validation nodes
            val_loss, val_acc = self.evaluate(data, "val")

            # If current val acc is best acc save current model state for later training
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), "best_model_state.pt")

            if epoch % 100 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f}"
                )

        # Evaluate test data
        test_loss, test_accuracy = self.evaluate(data, "test")

        print("\nFINAL TEST RESULTS")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        return test_loss, test_accuracy
