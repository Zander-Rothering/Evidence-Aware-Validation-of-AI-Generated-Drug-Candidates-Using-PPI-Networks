"""Match_result.py — output container for the full Part 1 compound matching engine."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Any

from .Filter_result import FilterResult


@dataclass
class MatchResult:
    """
    Aggregated output of Part 1 (Steps A2-A10b).

    Passed downstream to:
      - SimilarityScorer nearest-neighbour lookup (drives SIDER / NLP search terms)
      - ScaffoldExtractor scaffold comparison
      - NoveltyChecker novelty flag
      - RulesEngine deterministic tier (A10a)
      - RiskClassifier RF/SVM ensemble probability (A10b)
      - EvidenceAggregator (Part 3) as similarity_risk_score
    """

    # Input identity
    query_smiles: str = ""

    # Pipeline-stage error (set when A2 / heavy-atom / disconnect check fails).
    error: Optional[str] = None

    # Nearest-neighbour hit (A5)
    nn_name: str = ""
    nn_smiles: str = ""
    nn_ic50: Optional[float] = None
    tanimoto: Optional[float] = None      # max Tanimoto vs reference set (None on error)

    # Convenience alias used by downstream stages / tests
    nearest_neighbor_chembl_id: Optional[str] = None

    # Drug-likeness filter results (A6 / A7)
    filter_result: Optional[FilterResult] = None

    # Scaffold comparison (A8)
    shared_atoms: int = 0
    novel_atoms: int = 0
    shared_pct: float = 0.0
    scaffold_similarity: float = 0.0
    query_scaffold: str = ""
    nn_scaffold: str = ""

    # Novelty (A9)
    is_novel: bool = True             # False if identical to a training compound

    # ML feature vector (assembled by MatchingEngine for RF/SVM)
    ml_features: dict = field(default_factory=dict)

    # A10a -- deterministic rules
    rule_tier: Optional[str] = None              # "LOW" | "MEDIUM" | "HIGH"
    evidence_trail: list = field(default_factory=list)  # list[RuleVerdict]

    # A10b -- ML risk score
    similarity_risk_score: Optional[float] = None   # 0.0 (low) - 1.0 (high)
    classifier_confidence: Optional[float] = None   # 1.0 - |rf_p - svm_p|

    # Remote merged fields populated by MatchingEngine.run() (Part 2A/2B/3 handoff)
    risk_tier: str = ""                                 # final tier after A10c rule+ML reconciliation
    sider_risks: List[str] = field(default_factory=list)
    sider_risk_records: List[dict] = field(default_factory=list)
    search_terms: str = ""
    target: str = "HMGCR"
    ml_proba: dict = field(default_factory=dict)
    ml_tier: str = ""
    confidence_flag: str = ""

    def __post_init__(self) -> None:
        # Validate only when no upstream error was recorded; an errored
        # MatchResult is allowed to be partially populated for caller inspection.
        if self.error is None:
            if not self.query_smiles or not self.query_smiles.strip():
                raise ValueError("query_smiles must be non-empty when error is None")
            if self.tanimoto is not None and not (0.0 <= self.tanimoto <= 1.0):
                raise ValueError(
                    f"tanimoto out of range [0, 1]: {self.tanimoto!r}"
                )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "query_smiles":          self.query_smiles,
            "error":                 self.error,
            "nn_name":               self.nn_name,
            "nn_smiles":             self.nn_smiles,
            "nn_ic50":               self.nn_ic50,
            "tanimoto":              round(self.tanimoto, 4) if self.tanimoto is not None else None,
            "scaffold_similarity":   round(self.scaffold_similarity, 4),
            "is_novel":              self.is_novel,
            "rule_tier":             self.rule_tier,
            "similarity_risk_score": round(self.similarity_risk_score, 4) if self.similarity_risk_score is not None else None,
            "classifier_confidence": round(self.classifier_confidence, 4) if self.classifier_confidence is not None else None,
            "nearest_neighbor_chembl_id": self.nearest_neighbor_chembl_id,
        }
        if self.filter_result is not None:
            d.update(self.filter_result.to_ml_features())
        return d

    def __repr__(self) -> str:
        if self.error is not None:
            return f"MatchResult(error={self.error!r}, query={self.query_smiles!r})"
        tani = f"{self.tanimoto:.3f}" if self.tanimoto is not None else "n/a"
        risk = (
            f"{self.similarity_risk_score:.3f}"
            if self.similarity_risk_score is not None else "n/a"
        )
        return (
            f"MatchResult(nn={self.nn_name!r}, tanimoto={tani}, "
            f"tier={self.rule_tier}, novel={self.is_novel}, risk={risk})"
        )
