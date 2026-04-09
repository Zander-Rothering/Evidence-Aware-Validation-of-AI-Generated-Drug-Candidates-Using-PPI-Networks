"""NER Extractor — Pre-trained biomedical NER, no fine-tuning.

Part 2A (NLP Agent), Activity B2.

Model : d4data/biomedical-ner-all
          DistilBERT fine-tuned on MACCROBAT (41 biomedical entity types)
          https://huggingface.co/d4data/biomedical-ner-all

Remaps model output -> pipeline labels: DRUG, PROTEIN, DISEASE, ADVERSE_EFFECT.

Not perfect.
Will add a method to parse better later but works for now.
Will refactor later as well overall. / Or completely swap out later.

Developed with AI assistance (Claude, Anthropic) for syntax support.
"""

import re
import torch
from transformers import pipeline as hf_pipeline

MODEL_NAME = "d4data/biomedical-ner-all"

# d4data/biomedical-ner-all uses 41 entity types (e.g. "Medication", "Sign_symptom").
# Our pipeline expects only 4 labels: DRUG, PROTEIN, DISEASE, ADVERSE_EFFECT.
# This dict translates the model's labels into ours.
LABEL_MAP = {
    "Medication":            "DRUG",
    "Drug":                  "DRUG",
    "Chemical":              "DRUG",
    "Gene_protein":          "PROTEIN",
    "Disease_disorder":      "DISEASE",
    "Sign_symptom":          "ADVERSE_EFFECT",
    "Therapeutic_procedure": "DISEASE",
    "Biological_structure":  "DISEASE",
    "Diagnostic_procedure":  "DISEASE",
}


class NERExtractor:
    def __init__(self):
        # this is for my macbook setting (might need to change for other settings)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        # first run downloads the model (66MB) from HuggingFace, cached after that
        try:
            self.pipe = hf_pipeline("ner", model=MODEL_NAME, aggregation_strategy="simple", device=device)
        except Exception:
            self.pipe = hf_pipeline("ner", model=MODEL_NAME, aggregation_strategy="simple", device="cpu")

    def remap(self, entity: dict, has_drug: bool) -> dict | None:
        """Translate one model entity -> our label. Returns None if irrelevant."""
        label = LABEL_MAP.get(entity["entity_group"])
        if label is None:
            return None
        if label == "DISEASE" and has_drug:
            label = "ADVERSE_EFFECT"
        return {"entity": entity["word"], "label": label,
                "score": round(float(entity["score"]), 4)}

    def predict(self, abstract_texts: list[str], target_context: str = "HMGCR") -> list[dict]:
        # batch all abstracts at once instead of one-by-one
        all_raw = self.pipe(abstract_texts)

        results = []
        for text, raw in zip(abstract_texts, all_raw):
            has_drug = any(LABEL_MAP.get(e["entity_group"]) == "DRUG" for e in raw)

            mapped = [self.remap(e, has_drug) for e in raw]
            results.extend([m for m in mapped if m is not None])

            # fallback: model might miss specific proteins, so we string-match target names
            proteins = [t.strip() for t in target_context.split(",") if t.strip()]
            results.extend([
                {"entity": m.group(), "label": "PROTEIN", "score": 1.0}
                for p in proteins
                for m in re.finditer(re.escape(p), text, re.IGNORECASE)
            ])

        return results


if __name__ == "__main__":
    from pubmed_searcher import SAMPLE_RESULT # from pubmed_searcher.py

    extractor = NERExtractor()
    results = extractor.predict(
        SAMPLE_RESULT["abstracts"],
        target_context="HMGCR",
    )
    for hit in results:
        label = hit['label']
        entity = hit['entity']
        score = hit['score']
        print(f"  {label:20s} | {entity:30s} | {score:.4f}")
