"""LiteratureResult — Part 2A output container.

Packs B1-B4 outputs into one small object for downstream aggregation.

The LiteratureResult class only stores final NLP literature features. When this
file is run directly, it builds one LiteratureResult from the real B1-B4 pipeline.

Developed with AI assistance for syntax support.
"""

import argparse
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LiteratureResult:
    """Container for the final Part 2A literature evidence output."""
    literature_risk_score: float
    evidence_level: str
    top_signals: list[dict]
    evidence_confidence: float = 0.0
    search_terms: str = ""
    target: str = ""
    pmids: list[str] = field(default_factory=list)

    @classmethod
    def from_pipeline_outputs(
        cls,
        score_output: dict,
        evidence_source: str,
        signals: list[dict],
        search_terms: str = "",
        target: str = "",
        pmids: Optional[list[str]] = None,
        top_k: Optional[int] = 10
    ) -> "LiteratureResult":
        # Used by NLPAgent.run() to package B1-B4 outputs into a LiteratureResult.
        """Build a LiteratureResult from B1-B4 outputs."""
        sorted_signals = sorted(signals, key=lambda signal: signal.get("weight", 0), reverse=True)
        top_signals = sorted_signals if top_k is None else sorted_signals[:top_k]

        return cls(
            literature_risk_score=float(score_output.get("literature_risk_score", 0.0)),
            evidence_confidence=float(score_output.get("evidence_confidence", 0.0)),
            evidence_level=evidence_source,
            top_signals=top_signals,
            search_terms=search_terms,
            target=target,
            pmids=list(pmids or [])
        )

    def to_dict(self) -> dict:
        """Return a plain dict for printing, JSON export, or Part 3 handoff."""
        return {
            "literature_risk_score": self.literature_risk_score,
            "evidence_confidence": self.evidence_confidence,
            "evidence_level": self.evidence_level,
            "top_signals": self.top_signals,
            "search_terms": self.search_terms,
            "target": self.target,
            "pmids": self.pmids
        }


def build_literature_result(
    query: str,
    target: str = "HMGCR",
    max_results: int = 10,
    email: str = "example@berkeley.edu",
    sider_risks: Optional[list[str]] = None,
) -> LiteratureResult:
    """Run B1-B4 and return a LiteratureResult.

    This is useful when another file wants a real literature result without
    manually wiring PubMedSearcher, NERExtractor, SignalClassifier, and scorer.
    """
    # Local imports keep this module lightweight when only the `LiteratureResult` container is needed.
    from ner_extractor import NERExtractor
    from pubmed_searcher import PubMedSearcher
    from signal_classifier import SignalClassifier
    from literature_risk_scorer import LiteratureRiskScorer

    searcher = PubMedSearcher(max_results=max_results, email=email)
    hit = searcher.search(query, sider_risks=sider_risks, target=target)
    entities = NERExtractor().predict(hit["abstracts"], target_context=target)
    signals = SignalClassifier().classify(entities)
    score = LiteratureRiskScorer().score(signals, hit.get("evidence_source"))

    return LiteratureResult.from_pipeline_outputs(
        score_output=score,
        evidence_source=hit.get("evidence_source", "unknown"),
        signals=signals,
        search_terms=query,
        target=target,
        pmids=hit.get("pmids", []),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build LiteratureResult from the real B1-B4 pipeline")
    parser.add_argument(
        "--query",
        default="atorvastatin LDL cholesterol hypercholesterolemia",
        help="PubMed search query for this LiteratureResult demo",
    )
    parser.add_argument("--target", default="HMGCR", help="Target protein context for PubMed and NER")
    parser.add_argument("--max-results", type=int, default=10, help="Max PubMed papers to fetch")
    args = parser.parse_args()

    result = build_literature_result(
        query=args.query,
        target=args.target,
        max_results=args.max_results,
    )

    print("LiteratureResult")
    print(f"search_terms:          {result.search_terms}")
    print(f"target:                {result.target}")
    print(f"literature_risk_score: {result.literature_risk_score}")
    print(f"evidence_confidence:   {result.evidence_confidence}")
    print(f"evidence_level:        {result.evidence_level}")
    print(f"pmids:                 {', '.join(result.pmids)}")
    print(f"top_signals ({len(result.top_signals)}):")
    for signal in result.top_signals:
        print(f"  {signal['signal_type']:15s} | {signal['entity']:25s} | {signal.get('weight', 0):.4f}")
