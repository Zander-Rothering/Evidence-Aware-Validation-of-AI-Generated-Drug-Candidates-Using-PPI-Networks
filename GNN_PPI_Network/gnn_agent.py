import os
import torch
import torch.nn as nn
from torch_geometric.data import Data

from gnn_model import GNNModel
from node_label import node_labeler
from node_feature_builder import feature_extracter
from ppi_graph_builder import PPIGraphBuilder
from graph_trainer import GNNtrainer


class gnn_agent:
    """
    Combines GNN components into pipeline to build and train GNN for PPI Analysis
    """

    def __init__(self, best_model_path=None, learning_rate=0.001, decay=0.3):
        """
        Initializes the GNN agent components

        Parameters:
            best_model_path : str
                File path to pre-trained model state data.
                If provided previous model weights are loaded.
        """
        # Selects device to use (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Builds PPI Graph
        if os.path.exists("features_files/edge_index.pt") and os.path.exists(
            "features_files/edge_attr.pt"
        ):
            edge_index = torch.load(
                "edge_files/edge_index.pt", map_location="cpu", weights_only=True
            )
            edge_attr = torch.load(
                "edge_files/edge_attr.pt", map_location="cpu", weights_only=True
            )
        else:
            builder = PPIGraphBuilder()
            stringdb_df = builder.get_stringdb_network()
            biogrid_df = builder.get_biogrid_network(
                "edge_files/BIOGRID-ALL-5.0.256.tab3.txt"
            )
            edges_df = builder.merge_networks(stringdb_df, biogrid_df)
            proteins, protein_mapping = builder.protein_extraction(edges_df)
            edges_df = builder.encode_edge_types(edges_df)
            edge_index = builder.build_edges(edges_df, protein_mapping)
            edge_attr = builder.edge_attr(edges_df)

        # Extracts PPI Node Features
        if os.path.exists("features_files/node_features.pt"):
            x = torch.load(
                "features_files/node_features.pt", map_location="cpu", weights_only=True
            )
        else:
            extractor = feature_extracter(proteins)
            protein_features_df = extractor.get_uniprot()
            x = extractor.node_features_encoder(protein_features_df)

        # Creates PPI Node labels
        if os.path.exists("label_files/node_labels.pt"):
            y = torch.load(
                "label_files/node_labels.pt", map_location="cpu", weights_only=True
            )
        else:
            labeler = node_labeler(proteins)
            y = labeler.node_labels()

        # Creates torch_geometric.data graph object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        self.data = data.to(self.device)

        # Initializes GNN model
        model = GNNModel(
            in_channels=data.num_node_features, hidden_channels=128, out_classes=2
        )
        self.model = model.to(self.device)
        # Loads optional pre-trained weights
        if best_model_path is not None:
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device, weights_only=True)
            )

        # Defines optimizer and loss function
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=decay
        )
        loss_function = nn.CrossEntropyLoss()

        # Initializes trainer
        trainer = GNNtrainer(model, optimizer, loss_function)
        self.trainer = trainer

    def train_model(self, epochs=10000):
        """
        Train GNN model for specified number of epochs

        Parameters
            epochs : int
                Number of training epochs
        """
        self.trainer.test_model(self.data, epochs=epochs)

    def logit_prediction(self):
        """
        Predicts logit of nodes
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index)
            return logits
