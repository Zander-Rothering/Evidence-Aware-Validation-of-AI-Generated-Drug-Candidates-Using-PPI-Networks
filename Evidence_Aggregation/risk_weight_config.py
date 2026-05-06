from dataclasses import dataclass

@dataclass
class RiskWeightConfig:
    """
    Risk weights for each portion of our pipeline
    """
    def __init__(self, weight_cme, weight_nlp, weight_gnn):
        self.weights_cme = weight_cme
        self.weights_nlp = weight_nlp
        self.weights_gnn = weight_gnn
