from gnn_agent import gnn_agent
from neighbor_risk_aggregator import RiskAggregator

# Load pre-trained model
agent = gnn_agent(best_model_path="best_model_state.pt")
# Calculate raw logit predictions from best model
logits = agent.logit_prediction()

# Aggregate node level risk scores
aggregator = RiskAggregator()
# Calculate PPI Network risk score from logits
network_risk = aggregator.aggregate(logits)

with open("gnn_files/network_risk.txt", "w") as f:
    f.write(str(network_risk))
