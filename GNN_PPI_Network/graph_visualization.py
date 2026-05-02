import os
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

from node_label import node_labeler
from node_feature_builder import feature_extracter
from ppi_graph_builder import PPIGraphBuilder

# Builds PPI Graph
if os.path.exists("features_files/edge_index.pt") and os.path.exists("features_files/edge_attr.pt"):
    edge_index = torch.load("edge_files/edge_index.pt", map_location="cpu", weights_only=True)
    edge_attr = torch.load("edge_files/edge_attr.pt", map_location="cpu", weights_only=True)
else:
    builder = PPIGraphBuilder()
    stringdb_df = builder.get_stringdb_network()
    biogrid_df = builder.get_biogrid_network("edge_files/BIOGRID-ALL-5.0.256.tab3.txt")
    edges_df = builder.merge_networks(stringdb_df, biogrid_df)
    proteins, protein_mapping = builder.protein_extraction(edges_df)
    edges_df = builder.encode_edge_types(edges_df)
    edge_index = builder.build_edges(edges_df, protein_mapping)
    edge_attr = builder.edge_attr(edges_df)

# Extracts PPI Node Features
if os.path.exists("features_files/node_features.pt"):
    x = torch.load("features_files/node_features.pt", map_location="cpu", weights_only=True)
else:
    extractor = feature_extracter(proteins)
    protein_features_df = extractor.get_uniprot()
    x = extractor.node_features_encoder(protein_features_df)

# Creates PPI Node labels
if os.path.exists("label_files/node_labels.pt"):
    y = torch.load("label_files/node_labels.pt", map_location="cpu", weights_only=True)
else:
    labeler = node_labeler(proteins)
    y = labeler.node_labels()

# Creates torch_geometric.data graph object
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# Convert PyG Data to NetworkX graph
G = to_networkx(data, to_undirected=True)

# Visualize the graph
plt.figure(figsize=(10, 10))
nx.draw(G, node_size=7, pos=nx.spring_layout(G), with_labels=False)
plt.title('PPI Network: HMGCR', fontsize=15)
plt.tight_layout()
plt.show()