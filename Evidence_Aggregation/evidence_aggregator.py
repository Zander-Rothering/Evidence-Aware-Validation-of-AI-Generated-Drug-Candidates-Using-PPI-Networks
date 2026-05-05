from pathlib import Path

from .risk_score import RiskScore
from .risk_weight_config import RiskWeightConfig


class EvidenceAggregator:
    """Combines Part 1, 2A, and 2B evidence streams into a final RiskScore."""

    def __init__(self, config: RiskWeightConfig):
        self.config = config

    # Temporary Part 2B helper: read the MVP float from GNN_PPI_Network/gnn_files/network_risk.txt until `NetworkResult` is implemented.
    def extract_network_inputs(self, network_result=None, network_risk: float | None = None, flagged_proteins: list[str] | None = None) -> tuple[float, list[str]]:
        """Normalize Part 2B inputs from future NetworkResult, direct float, or current network_risk.txt fallback."""
        if network_result is not None:
            return (float(network_result.network_risk_score), list(getattr(network_result, "flagged_proteins", []) or []))

        if network_risk is None:
            path = Path(__file__).resolve().parents[1] / "GNN_PPI_Network" / "gnn_files" / "network_risk.txt"
            with open(path) as f:
                network_risk = float(f.read().strip())

        return float(network_risk), list(flagged_proteins or [])

    def aggregate(self, match_result, literature_result, network_result=None, network_risk: float | None = None, flagged_proteins: list[str] | None = None) -> RiskScore:
        network_risk, flagged_proteins = self.extract_network_inputs(network_result=network_result, network_risk=network_risk, flagged_proteins=flagged_proteins)
        # Part 1 / Compound Matching: risk-oriented similarity signal from MatchResult.
        # The raw Tanimoto score is still stored separately as RiskScore.similarity_score.
        similarity_risk = match_result.similarity_risk_score

        # Part 3 / Evidence Aggregation: weighted combination of the three independent streams:
        # Part 1 similarity risk, Part 2A literature risk, and Part 2B network risk.
        weighted_scores = {
            "compound_matching": self.config.weights_cme * similarity_risk,
            "literature": self.config.weights_nlp * literature_result.literature_risk_score,
            "network": self.config.weights_gnn * network_risk
        }
        combined_risk_signal = sum(weighted_scores.values())

        tier = self._classify(combined_risk_signal, similarity_risk, literature_result.literature_risk_score, network_risk)

        return RiskScore(
            risk_tier=tier,                                         # Part 3 final HIGH / MEDIUM / LOW classification.
            similarity_score=match_result.tanimoto,                 # Part 1 raw A5 Tanimoto similarity, kept for reporting.
            literature_risk_score=literature_result.literature_risk_score,  # Part 2A NLP literature risk score.
            network_risk_score=network_risk,                        # Part 2B PPI/GNN network risk score.
            combined_score=combined_risk_signal,                    # Part 3 weighted final output score.
            flagged_proteins=flagged_proteins,                      # Part 2B flagged network proteins.
            top_signals=literature_result.top_signals,              # Part 2A top extracted literature signals.
            ml_proba=match_result.ml_proba,                         # Part 1 A10b ML risk probabilities.
            target=match_result.target,                             # Shared target context across Part 1/2A/2B (default HMGCR).
            search_terms=match_result.search_terms,                 # Part 1 PubMed query string forwarded to Part 2A.
            confidence_flag=match_result.confidence_flag,           # Part 1 A10c rule-vs-ML agreement flag.
            evidence_level=literature_result.evidence_level,        # Part 2A evidence confidence level.
            pmids=literature_result.pmids                           # Part 2A supporting PubMed IDs.
        )

    def _classify(self, combined_risk_signal: float, similarity_risk: float, literature_risk: float, network_risk: float) -> str:
        """Classify final risk using weighted score plus stream-level agreement."""
        stream_scores = [similarity_risk, literature_risk, network_risk]
        high_streams = sum(score >= 0.7 for score in stream_scores)

        if high_streams == 3:
            return "HIGH"

        if combined_risk_signal >= 0.7:
            return "HIGH"

        if high_streams >= 1:
            return "MEDIUM"

        if combined_risk_signal >= 0.4:
            return "MEDIUM"

        return "LOW"
