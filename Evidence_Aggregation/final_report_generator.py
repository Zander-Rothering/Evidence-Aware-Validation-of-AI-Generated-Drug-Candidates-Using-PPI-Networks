"""Report helpers for final RiskScore outputs.
    Developed with AI assistance for syntax support.
"""

import csv
import json

try:
    from .risk_score import RiskScore
except ImportError:
    from risk_score import RiskScore


CSV_FIELDS = [
    "query_smiles",
    "risk_tier",
    "combined_score",
    "similarity_score",
    "literature_risk_score",
    "network_risk_score",
    "target",
    "confidence_flag",
    "evidence_level",
]

ANNOTATED_CSV_FIELDS = [
    "query_smiles",
    "is_pubchem_novel",
    "risk_tier",
    "combined_score",
    "similarity_score",
    "literature_risk_score",
    "network_risk_score",
    "target",
    "confidence_flag",
    "evidence_level",
]


def _as_list(results) -> list[RiskScore]:
    """Accept one RiskScore or many RiskScore objects."""
    if isinstance(results, RiskScore):
        return [results]
    return list(results)


def save_csv_results(results, output_path: str = "validation_results.csv") -> None:
    """Save batch-friendly summary rows for sorting/filtering in spreadsheets."""
    rows = [result.to_dict() for result in _as_list(results)]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})

    print(f"Saved CSV results: {output_path}")


def save_json_results(results, output_path: str = "validation_results.json") -> None:
    """Save full nested evidence packages for reproducibility/audit trails."""
    rows = [result.to_dict() for result in _as_list(results)]

    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Saved JSON results: {output_path}")


def save_annotated_csv(rows: list[dict], output_path: str = "validation_results.csv") -> None:
    """Save validation summary rows with one PubChem novelty field as CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATED_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in ANNOTATED_CSV_FIELDS})

    print(f"Saved CSV results: {output_path}")


def save_annotated_json(rows: list[dict], output_path: str = "validation_results.json") -> None:
    """Save validation summary rows with the same fields as the annotated CSV."""
    rows = [
        {field: row.get(field, "") for field in ANNOTATED_CSV_FIELDS}
        for row in rows
    ]
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Saved JSON results: {output_path}")
