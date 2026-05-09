"""B3 signal classifier for presentation MVP.

Usage (from this folder):
    python signal_classifier_mvp.py
    python signal_classifier_mvp.py --query "drug or topic keywords"

`--query` overrides the default PubMed search string; omit it to use the built-in demo query.

Maps extracted entities into downstream signal types:
  - EFFICACY for LDL / cholesterol lowering mentions
  - SAFETY_FLAG for safety/tolerability-adverse cues

Developed with AI assistance for syntax support.
"""
import argparse

try:
    from .ner_extractor import NERExtractor
    from .pubmed_searcher import PubMedSearcher
except ImportError:
    from ner_extractor import NERExtractor
    from pubmed_searcher import PubMedSearcher

from rich.console import Console
from rich.table import Table


class SignalClassifier:
    # Rule-based mapper: B2 entity dict -> {entity, signal_type, weight} or None.
    def __init__(self) -> None:
        # Substrings on normalized NER span text (model often splits words oddly).
        self.efficacy_terms = {
            "choterol",
            "dyslipidemia",
            "hypercholesterolemia",
            "hyperchoterol",
            "ldl-c",
            "ldl - c",
            "ldl-c reduction",
            "ldl-c lowering",
            "reduced ldl-c",
            "lowers ldl-c",
            "ldl cholesterol",
            "ldl - cholesterol",
            "ldl reduction",
            "ldl lowering",
            "reduced ldl",
            "lowers ldl",
            "density lipoprotein",
            "low-density lipoprotein",
            "low density lipoprotein",
            "low-density lipoprotein cholesterol",
            "low density lipoprotein cholesterol",
            "lipoprotein - cholesterol",
            "lipoprotein cholesterol",
            "cholesterol reduction",
            "cholesterol lowering",
            "plasma cholesterol",
            "serum cholesterol",
            "hyperlipidemia",
            "lipid lowering",
            "lipid-lowering",
        }
        # Conservative vocabulary guard for entity->signal mapping:
        # ADVERSE_EFFECT/DISEASE labels become SAFETY_FLAG only when text matches
        # known safety/tolerability terms.
        self.safety_flag_terms = {
            "adverse effect",
            "adverse effects",
            "adverse event",
            "adverse events",
            "side effect",
            "side effects",
            "creatine kinase",
            "creatine phosphokinase",
            "ck elevation",
            "ck elevations",
            "hepatotoxicity",
            "hepatic",
            "hepatotoxic",
            "transaminase",
            "aminotransferase",
            "liver",
            "liver enzyme",
            "liver enzymes",
            "enzyme elevations",
            "muscle",
            "muscle toxicity",
            "muscle symptoms",
            "muscular",
            "muscular toxicity",
            "myalgia",
            "myopathy",
            "myositis",
            "rash",
            "pruritus",
            "urticaria",
            "pain",
            "renal",
            "renal dysfunction",
            "renal impairment",
            "acute renal failure",
            "kidney",
            "rhabdomyolysis",
            "dose-dependent",
            "discontinuation",
            "serious adverse",
            "clinically important adverse",
            "toxicity",
        }

    def normalize_text(self, text: str) -> str:
        return text.strip().lower()

    def is_efficacy_text(self, text: str) -> bool:
        return any(term in text for term in self.efficacy_terms)

    def is_safety_flag_text(self, text: str) -> bool:
        return any(term in text for term in self.safety_flag_terms)

    def classify_one(self, entity: dict) -> dict | None:
        label = entity.get("label", "")
        text = self.normalize_text(entity.get("entity", ""))
        score = float(entity.get("score", 0.0))  # NER confidence; we pass through as weight.

        # Efficacy wins over label: any span that looks lipid-related -> EFFICACY.
        if self.is_efficacy_text(text):
            return {
                "entity": entity.get("entity", ""),
                "signal_type": "EFFICACY",
                "weight": score,
            }

        match label:
            case "ADVERSE_EFFECT" | "DISEASE":
                if not self.is_safety_flag_text(text):
                    return None
                return {
                    "entity": entity.get("entity", ""),
                    "signal_type": "SAFETY_FLAG",
                    "weight": score,
                }
            case _:  # everything else (e.g., DRUG, PROTEIN): no signal
                return None

    def classify(self, entities: list[dict]) -> list[dict]:
        signals = []
        for entity in entities:
            signal = self.classify_one(entity)
            if signal is not None:
                signals.append(signal)
        return signals


if __name__ == "__main__":
    # PubMed -> NER -> signals. Change the query with --query; other settings are fixed below.
    _TARGET = "HMGCR"  # passed into search() + ner `target_context` (protein hints in B2).
    _MAX_RESULTS = 10  # Max PubMed papers to fetch (more = slower, usually more entities).

    parser = argparse.ArgumentParser(description="B3 MVP: PubMed query -> NER -> signal tables")
    parser.add_argument(
        "--query",
        default="atorvastatin LDL cholesterol hypercholesterolemia",
        help="PubMed search query for the MVP demo",
    )
    args = parser.parse_args()

    searcher = PubMedSearcher(max_results=_MAX_RESULTS, email="example@berkeley.edu")
    hit = searcher.search(args.query, target=_TARGET)  # tier in hit["evidence_source"] = which fallback query hit.
    extractor = NERExtractor()
    entities = extractor.predict(hit["abstracts"], target_context=_TARGET)
    signals = SignalClassifier().classify(entities)

    console = Console()
    console.print(
        f"Query: {args.query}\n"
        f"PubMed tier: {hit.get('evidence_source')}  |  "
        f"NER entities (B2): {len(entities)}  |  "
        f"Signals (B3): {len(signals)}\n"
    )

    for name in ("EFFICACY", "SAFETY_FLAG"):
        rows = sorted((s for s in signals if s["signal_type"] == name), key=lambda s: s["weight"], reverse=True)
        t = Table(title=name)
        t.add_column("entity")
        t.add_column("weight", justify="right")
        (rows and list(map(lambda s: t.add_row(s["entity"], f"{s['weight']:.4f}"), rows))) or t.add_row("—", "—")
        console.print(t)
