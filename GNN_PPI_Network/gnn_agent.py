import torch
import torch.nn as nn
from torch_geometric.data import Data

from gnn_model import GNNModel
from node_label import node_labeler
from node_feature_builder import feature_extracter
from ppi_graph_builder import PPIGraphBuilder
from graph_trainer import GNNtrainer

class gnn_agent():
    def __init__(self, best_model_path= None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        builder = PPIGraphBuilder()
        stringdb_df = builder.get_stringdb_network()
        biogrid_df = builder.get_biogrid_network("BIOGRID-ALL-5.0.256.tab3.txt")
        edges_df = builder.merge_networks(stringdb_df, biogrid_df)
        proteins, protein_mapping = builder.protein_extraction(edges_df) 
        edge_index = builder.build_edges(edges_df, protein_mapping)
        edge_attr = builder.edge_weights(edges_df)

        extractor = feature_extracter(proteins)
        protein_features_df = extractor.get_uniprot()
        x = extractor.node_features_encoder(protein_features_df)
        
        labeler = node_labeler(proteins)
        y = labeler.node_labels()

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        self.data = data.to(self.device)
        
        model = GNNModel(in_channels=data.num_node_features, hidden_channels=128, out_classes=2)
        if best_model_path is not None:
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_function = nn.CrossEntropyLoss()

        trainer = GNNtrainer(model, optimizer, loss_function)
        self.trainer = trainer

    def train_model(self, epochs=1000):
        self.trainer.test_model(self.data, epochs=epochs)
    
Agent = gnn_agent(best_model_path= None)
Agent.train_model(epochs= 16)
