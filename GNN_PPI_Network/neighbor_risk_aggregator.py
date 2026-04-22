import numpy as np
import torch.nn.functional as F


class RiskAggregator:
    """
    Aggregates risk scores of nodes in Protein-Protein Interaction Network to get overall risk score
    """

    def aggregate(self, logits):
        """
        Parameters:
            logits : Tensor
                Raw risk scores for nodes in PPI network (Shape = [num_nodes, num_classes])

        Returns:
            network_risk : float
                Overall risk score of PPI
        """
        # Calculate probabilities using softmax
        probs = F.softmax(logits, dim=1)

        # Extract probability of single class 
        risk_scores = probs[:, 1]

        # Average risk score over PPI nodes
        network_risk = (risk_scores).mean()

        return network_risk.item()
