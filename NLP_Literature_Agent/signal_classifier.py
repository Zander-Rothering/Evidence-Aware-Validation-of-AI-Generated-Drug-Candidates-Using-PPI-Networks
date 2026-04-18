"""`SignalClassifier` is a simple rule-based classifier that identifies whether an entity is
 a "signal" based on its NER label and text content.
 """

import re

class SignalClassifier:
    def __init__(self, target_context: str = "HMGCR") -> None:
        # Store normalized target names for protein matching
        self.target_terms = {self._normalize_text(t) for t in target_context.split(",") if t.strip()}

    def normalize_text(text: str) -> str:
        return text.strip().lower()

    def classify_one(self, entity: dict) -> dict | None:
        # Read the NER label and entity text
        label = entity.get("label", "")
        text = self.normalize_text(entity.get("entity", ""))

        pass