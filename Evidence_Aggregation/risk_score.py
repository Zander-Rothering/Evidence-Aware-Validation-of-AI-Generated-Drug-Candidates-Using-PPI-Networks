from dataclasses import dataclass, field

@dataclass
class RiskScore:
    """Final pipeline output container for one AI-generated compound.

    Aggregates evidence from all three pipeline stages:
      - Part 1  Compound Matching  -> similarity_score
      - Part 2A NLP Literature     -> literature_risk_score
      - Part 2B PPI / GNN Network  -> network_risk_score

    Attributes
    ----------
    risk_tier : str
        Overall risk classification: HIGH / MEDIUM / LOW.
    similarity_score : float
        Tanimoto similarity from Part 1 (MatchResult).
    literature_risk_score : float
        NLP-derived risk score from Part 2A.
    network_risk_score : float
        PPI/GNN-derived risk score from Part 2B.
    combined_score : float
        Weighted combination of all evidence streams.
    flagged_proteins : list[str]
        Proteins flagged during PPI network analysis.
    top_signals : list[dict]
        Highest-confidence literature signals.
    ml_proba : dict
        ML model class probabilities.
    target : str
        Primary protein target (default HMGCR).
    search_terms : str
        PubMed query terms used by NLP agent.
    confidence_flag : str
        Data-quality confidence indicator.
    evidence_level : str
        Strength of supporting evidence.
    pmids : list[str]
        PubMed IDs cited as evidence.
    """

    # TODO: risk_tier —> final HIGH/MEDIUM/LOW classification based on combined_score (not yet implemented).
    risk_tier: str = ""
    similarity_score: float = 0.0
    literature_risk_score: float = 0.0
    network_risk_score: float = 0.0
    # TODO: combined_score —> weighted aggregation of similarity, literature, and network scores (not yet implemented).
    combined_score: float = 0.0

    # TODO: flagged_proteins —> extract flagged protein names from GNN node-level logits (not yet implemented).
    flagged_proteins: list[str] = field(default_factory=list)
    top_signals: list[dict] = field(default_factory=list)
    ml_proba: dict = field(default_factory=dict)

    target: str = "HMGCR"
    search_terms: str = ""
    confidence_flag: str = ""
    evidence_level: str = ""
    pmids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a plain dict for reporting, JSON export, or notebooks."""
        return {
            "risk_tier": self.risk_tier,
            "similarity_score": round(self.similarity_score, 4),
            "literature_risk_score": round(self.literature_risk_score, 4),
            "network_risk_score": round(self.network_risk_score, 4),
            "combined_score": round(self.combined_score, 4),
            "flagged_proteins": self.flagged_proteins,
            "top_signals": self.top_signals,
            "ml_proba": self.ml_proba,
            "target": self.target,
            "search_terms": self.search_terms,
            "confidence_flag": self.confidence_flag,
            "evidence_level": self.evidence_level,
            "pmids": self.pmids,
        }

    def __repr__(self) -> str:
        return (
            f"RiskScore(tier={self.risk_tier!r}, combined={self.combined_score:.3f}, "
            f"similarity={self.similarity_score:.3f}, literature={self.literature_risk_score:.3f}, "
            f"network={self.network_risk_score:.3f})"
        )