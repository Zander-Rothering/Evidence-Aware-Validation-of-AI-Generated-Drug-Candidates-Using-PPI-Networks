from dataclasses import dataclass

@dataclass
class RiskWeightConfig:
    """
    Risk weights for each portion of our pipeline.
    """
    def __init__(self,weight_cme: float = 0.45, weight_nlp: float = 0.25, weight_gnn: float = 0.30):
        self.weights_cme = weight_cme
        self.weights_nlp = weight_nlp
        self.weights_gnn = weight_gnn