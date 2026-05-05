"""Literature Risk Scorer — Part 2A, Activity B4.

Converts signal list (B3) into a numeric risk score 0-1.
Score is down-weighted if evidence came from class-level
fallback rather than direct compound literature.

Some lines of codes in the main function need to be 
refactored after having a working pipeline to avoid code duplicaition 
(needed as of now)

Developed with AI assistance for syntax support.
"""

import numpy as np

try:
    from .ner_extractor import NERExtractor
    from .pubmed_searcher import PubMedSearcher
    from .signal_classifier import SignalClassifier
except ImportError:
    from ner_extractor import NERExtractor
    from pubmed_searcher import PubMedSearcher
    from signal_classifier import SignalClassifier


# Evidence level weights — direct compound hit scores higher than fallback
EVIDENCE_WEIGHTS = {
    "level_0": 1.00,  # compound + target
    "level_1": 0.80,  # scaffold + target
    "level_2": 0.60,  # drug class + target
    "level_3": 0.40,  # target only
}

DEFAULT_EVIDENCE_WEIGHT = 0.40  # unknown / missing B1 tag -> treat like broadest fallback (level_3)
EFFICACY_DISCOUNT = 0.35  # X raw_efficacy, then subtract from raw_safety


class LiteratureRiskScorer:
    """Convert B3 signals into a single literature_risk_score 0-1."""

    def __init__(self) -> None:
        # Copy defaults into instance variables for potential future customization
        self.evidence_weights = dict(EVIDENCE_WEIGHTS)
        self.default_evidence_weight = DEFAULT_EVIDENCE_WEIGHT
        self.efficacy_discount = EFFICACY_DISCOUNT

    @staticmethod
    def signal_weights_array(signals: list[dict], signal_type: str) -> np.array:
        """Return one signal type's weights as a float array."""
        return np.array(
            [float(s["weight"]) for s in signals if s.get("signal_type") == signal_type],
            dtype=float,
        )

    def score(self, signals: list[dict], evidence_source: str | None = None) -> dict:
        safety_weights = self.signal_weights_array(signals, "SAFETY_FLAG")
        efficacy_weights = self.signal_weights_array(signals, "EFFICACY")

        # Means of NER weights — empty type => 0
        raw_safety = float(safety_weights.mean()) if safety_weights.size > 0 else 0.0
        raw_efficacy = float(efficacy_weights.mean()) if efficacy_weights.size > 0 else 0.0
        adjusted_risk = max(0.0, raw_safety - (self.efficacy_discount * raw_efficacy))  # efficacy softens headline risk
        tier = float(self.evidence_weights.get(evidence_source or "", self.default_evidence_weight))  # PubMed query tier from B1
        literature_risk_score = float(np.clip(adjusted_risk * tier, 0.0, 1.0))  # down-weight if evidence is weak/broad

        return {
            "literature_risk_score": round(literature_risk_score, 4),
            "evidence_confidence": round(tier, 4),
        }


if __name__ == "__main__":
    _QUERY = "atorvastatin LDL cholesterol hypercholesterolemia"
    _TARGET = "HMGCR"
    _MAX_RESULTS = 10

    searcher = PubMedSearcher(max_results=_MAX_RESULTS, email="example@berkeley.edu")
    hit = searcher.search(_QUERY, target=_TARGET)

    extractor = NERExtractor()
    entities = extractor.predict(hit["abstracts"], target_context=_TARGET)

    classifier = SignalClassifier()
    signals = classifier.classify(entities)

    scorer = LiteratureRiskScorer()
    result = scorer.score(signals, evidence_source=hit.get("evidence_source"))

    print("evidence_source:", hit.get("evidence_source"))
    print("n_signals:", len(signals))
    print(result)
