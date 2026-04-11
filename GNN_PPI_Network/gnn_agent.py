from torch_geometric.data import Data

from gnn_model import GNNModel
from node_label import node_labels
from node_feature_builder import feature_extracter
from ppi_graph_builder import PPIGraphBuilder

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