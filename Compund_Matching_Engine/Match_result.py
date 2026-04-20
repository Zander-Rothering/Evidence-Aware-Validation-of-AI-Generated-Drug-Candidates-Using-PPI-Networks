"""Match_result.py — output container for the full Part 1 compound matching engine."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from .Filter_result import FilterResult


@dataclass
class MatchResult:
    """
    Aggregated output of Part 1 (Steps A2–A9).

    Passed downstream to:
      - SimilarityScorer nearest-neighbour lookup (drives SIDER / NLP search terms)
      - ScaffoldExtractor scaffold comparison
      - NoveltyChecker novelty flag
      - EvidenceAggregator (Part 3) as similarity_risk_score
    """

    # Input identity
    query_smiles: str = ""

    # Nearest-neighbour hit (A5)
    nn_name: str = ""
    nn_smiles: str = ""
    nn_ic50: Optional[float] = None
    tanimoto: float = 0.0           # max Tanimoto against ChEMBL reference set

    # Drug-likeness filter results (A6 / A7)
    filter_result: Optional[FilterResult] = None

    # Scaffold comparison (A8)
    shared_atoms: int = 0
    scaffold_similarity: float = 0.0
    query_scaffold: str = ""
    nn_scaffold: str = ""

    # Novelty (A9)
    is_novel: bool = True           # False if identical to a training compound

    # ML feature vector (assembled by MatchingEngine for RF/SVM)
    ml_features: dict = field(default_factory=dict)

    # Risk signal contributed to Part 3
    similarity_risk_score: float = 0.0   # 0.0 (low) – 1.0 (high)

    def to_dict(self) -> dict:
        d = {
            "query_smiles":        self.query_smiles,
            "nn_name":             self.nn_name,
            "nn_smiles":           self.nn_smiles,
            "nn_ic50":             self.nn_ic50,
            "tanimoto":            round(self.tanimoto, 4),
            "scaffold_similarity": round(self.scaffold_similarity, 4),
            "is_novel":            self.is_novel,
            "similarity_risk_score": round(self.similarity_risk_score, 4),
        }
        if self.filter_result is not None:
            d.update(self.filter_result.to_ml_features())
        return d

    def __repr__(self) -> str:
        return (
            f"MatchResult(nn={self.nn_name!r}, tanimoto={self.tanimoto:.3f}, "
            f"novel={self.is_novel}, risk={self.similarity_risk_score:.3f})"
        )

