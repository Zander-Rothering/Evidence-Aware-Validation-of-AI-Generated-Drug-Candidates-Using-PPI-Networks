"""NER Extractor — Pre-trained biomedical NER, no fine-tuning.

Part 2A (NLP Agent), Activity B2.

Model : d4data/biomedical-ner-all
          DistilBERT fine-tuned on MACCROBAT (41 biomedical entity types)
          https://huggingface.co/d4data/biomedical-ner-all

Remaps model output -> pipeline labels: DRUG, PROTEIN, DISEASE, ADVERSE_EFFECT.

Not perfect.
Will refactor later as well overall. / Or completely swap out later.

Developed with AI assistance (Claude, Anthropic) for syntax support.
"""

import re
import torch
from transformers import pipeline as hf_pipeline

try:
    from .pubmed_searcher import PubMedSearcher
except ImportError:
    from pubmed_searcher import PubMedSearcher

MODEL_NAME = "d4data/biomedical-ner-all"

# --- Rules / config (post-processing and label mapping) ---

# Model entity_group -> pipeline label.
LABEL_MAP = {
    "Medication": "DRUG",
    "Drug": "DRUG",
    "Chemical": "DRUG",
    "Gene_protein": "PROTEIN",
    "Disease_disorder": "DISEASE",
    "Sign_symptom": "ADVERSE_EFFECT",
    "Therapeutic_procedure": "DISEASE",
    "Biological_structure": "DISEASE",
    "Diagnostic_procedure": "DISEASE",
}

# Normalized surface -> label override (after map_entity, before DA blocklist).
SURFACE_RELABEL: dict[str, str] = {
    "ldl - c": "DISEASE",
    "ldl-c": "DISEASE",
    "ldl cholesterol": "DISEASE",
    "lipoprotein choterol": "DISEASE",  # typo for cholesterol / lipoprotein cholesterol
}

# If has_drug flips DISEASE -> ADVERSE_EFFECT, keep DISEASE for these normalized surfaces.
PROTECTED_TERMS = frozenset({
    "hmgcr",
    "hmg-coa",
    "hmg coa",
    "hmg - coa",
    "reductase",
    "hmg-coa reductase",
    "hmg coa reductase",
    "3 - hydroxy - 3 - methylglutaryl - coenzyme a reductase",
    "3-hydroxy-3-methylglutaryl-coenzyme a reductase",
})

DRUG_BLOCKLIST = frozenset({
    "sinam",  # clinical trial / ID-style fragment, not a drug name
})

DISEASE_ADVERSE_BLOCKLIST = frozenset({
    "studies",
    "tests",
    "pd",
    "trib",
    "pca",
    "inverse variance weighted",
    "crisp",
    "sinam",
    "anti",
    "girdle",
    "pathway enrichment",
    "copsta",
    "mouse",
    "compound",
    "molecular docking",
    "disease",
    "hyper",
    "hmgc",
    "aldolase",
    "ot - autoimmune",
    "mediated necrotizing myopathy",
    "reductase",
    "hmgcr",
})

DISEASE_ADVERSE_LABELS = frozenset({"ADVERSE_EFFECT", "DISEASE"})

SCORE_THRESHOLD = 0.9
MIN_DRUG_ENTITY_LEN = 4
# Drop ADVERSE_EFFECT / DISEASE when len(entity.strip()) <= this.
MAX_SHORT_FRAGMENT_CHARS = 3


def normalize_entity(text: str) -> str:
    """Collapse whitespace; lowercase. Use for blocklists and relabel keys."""
    return re.sub(r"\s+", " ", text).strip().lower()


class NERExtractor:
    def __init__(self):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        try:
            self.pipe = hf_pipeline("ner", model=MODEL_NAME, aggregation_strategy="simple", device=device)
        except Exception:
            self.pipe = hf_pipeline("ner", model=MODEL_NAME, aggregation_strategy="simple", device="cpu")

        self._postprocessors = [
            self.filter_by_score,
            self.drop_subword_artifacts,
            self.filter_entities,
            self.apply_surface_rules,
            self.drop_blocklisted_disease_adverse,
            self.drop_contained_drug_fragments,
            self.deduplicate_by_entity_label,
        ]

    @staticmethod
    def merge_tokens(entities: list[dict]) -> list[dict]:
        """Merge WordPiece fragments; conservative score = min of merged pieces."""
        merged = []
        for e in entities:
            word = e.get("word", "")
            if word.startswith("##") and merged:
                merged[-1]["word"] = merged[-1]["word"] + word[2:]
                merged[-1]["score"] = min(merged[-1]["score"], e["score"])
            else:
                merged.append(dict(e))
        return merged

    @staticmethod
    def map_model_label(entity_group: str) -> str | None:
        return LABEL_MAP.get(entity_group)

    @staticmethod
    def apply_context_rules(label: str, entity_text: str, has_drug: bool) -> str:
        """Co-occurring drug: treat many DISEASE spans as signs/symptoms unless protected."""
        if label == "DISEASE" and has_drug:
            if normalize_entity(entity_text) not in PROTECTED_TERMS:
                return "ADVERSE_EFFECT"
        return label

    def map_entity(self, entity: dict, has_drug: bool) -> dict | None:
        """One HF aggregated entity -> our row; None if entity type is unused."""
        label = self.map_model_label(entity["entity_group"])
        if label is None:
            return None
        label = self.apply_context_rules(label, entity["word"], has_drug)
        return {
            "entity": entity["word"],
            "label": label,
            "score": round(float(entity["score"]), 4),
        }

    @staticmethod
    def collect_protein_matches(text: str, target_context: str) -> list[dict]:
        """Regex fallback when the model misses target protein symbols."""
        proteins = [t.strip() for t in target_context.split(",") if t.strip()]
        out: list[dict] = []
        for p in proteins:
            for m in re.finditer(re.escape(p), text, re.IGNORECASE):
                out.append({"entity": m.group(), "label": "PROTEIN", "score": 1.0})
        return out

    @staticmethod
    def filter_by_score(items: list[dict]) -> list[dict]:
        return [x for x in items if float(x["score"]) >= SCORE_THRESHOLD]

    @staticmethod
    def drop_subword_artifacts(items: list[dict]) -> list[dict]:
        # Unmerged "##" pieces are junk spans.
        return [x for x in items if "##" not in x["entity"]]

    @staticmethod
    def filter_entities(items: list[dict]) -> list[dict]:
        """DRUG length + DRUG blocklist + short AE + short DISEASE (single pass)."""
        out: list[dict] = []
        for x in items:
            label = x["label"]
            stripped = x["entity"].strip()
            norm = normalize_entity(x["entity"])
            if label == "DRUG":
                if len(stripped) < MIN_DRUG_ENTITY_LEN:
                    continue
                if norm in DRUG_BLOCKLIST:
                    continue
            if label == "ADVERSE_EFFECT" and len(stripped) <= MAX_SHORT_FRAGMENT_CHARS:
                continue
            if label == "DISEASE" and len(stripped) <= MAX_SHORT_FRAGMENT_CHARS:
                continue
            out.append(x)
        return out

    @staticmethod
    def apply_surface_rules(items: list[dict]) -> list[dict]:
        """SURFACE_RELABEL overrides after coarse label mapping."""
        relabeled: list[dict] = []
        for item in items:
            row = dict(item)
            row["label"] = SURFACE_RELABEL.get(normalize_entity(item["entity"]), item["label"])
            relabeled.append(row)
        return relabeled

    @staticmethod
    def drop_blocklisted_disease_adverse(items: list[dict]) -> list[dict]:
        out: list[dict] = []
        for x in items:
            if x["label"] not in DISEASE_ADVERSE_LABELS:
                out.append(x)
                continue
            if normalize_entity(x["entity"]) in DISEASE_ADVERSE_BLOCKLIST:
                continue
            out.append(x)
        return out

    @staticmethod
    def drop_contained_drug_fragments(items: list[dict]) -> list[dict]:
        """Longest non-contained DRUG surface wins per abstract-sized batch."""
        drug_i = [i for i, x in enumerate(items) if x["label"] == "DRUG"]
        norm = {i: normalize_entity(items[i]["entity"]) for i in drug_i}
        kept: list[str] = []
        ok: set[int] = set()
        for i in sorted(drug_i, key=lambda j: len(norm[j]), reverse=True):
            s = norm[i]
            if s and not any(len(t) > len(s) and s in t for t in kept):
                kept.append(s)
                ok.add(i)
        return [x for i, x in enumerate(items) if x["label"] != "DRUG" or i in ok]

    @staticmethod
    def deduplicate_by_entity_label(items: list[dict]) -> list[dict]:
        # Key matches legacy behavior: strip + lower only (no internal space collapse).
        best: dict[tuple[str, str], dict] = {}
        for x in items:
            key = (x["entity"].strip().lower(), x["label"])
            prev = best.get(key)
            if prev is None or float(x["score"]) > float(prev["score"]):
                best[key] = dict(x)
        return list(best.values())

    def postprocess(self, items: list[dict]) -> list[dict]:
        for fn in self._postprocessors:
            items = fn(items)
        return items

    def predict(self, abstract_texts: list[str], target_context: str = "HMGCR") -> list[dict]:
        all_raw = self.pipe(abstract_texts)
        all_raw = [self.merge_tokens(raw) for raw in all_raw]

        results: list[dict] = []
        for text, raw in zip(abstract_texts, all_raw):
            has_drug = any(self.map_model_label(e["entity_group"]) == "DRUG" for e in raw)
            mapped = [self.map_entity(e, has_drug) for e in raw]
            results.extend(m for m in mapped if m is not None)
            results.extend(self.collect_protein_matches(text, target_context))

        return self.postprocess(results)


if __name__ == "__main__":
    searcher = PubMedSearcher(max_results=10, email="example@berkeley.edu")
    result = searcher.search("atorvastatin")
    print("evidence_source:", result.get("evidence_source"), "| n_abstracts:", len(result.get("abstracts", [])),)

    extractor = NERExtractor()
    results = extractor.predict(result["abstracts"], target_context="HMGCR")

    for hit in results:
        label = hit["label"]
        entity = hit["entity"]
        score = hit["score"]
        print(f"  {label:20s} | {entity:30s} | {score:.4f}")
