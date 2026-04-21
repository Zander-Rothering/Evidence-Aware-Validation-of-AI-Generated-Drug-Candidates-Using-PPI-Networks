"""
fetch_chembl402.py

Pulls ALL activity records for target CHEMBL402 (HMG-CoA reductase, HMGCR)
from ChEMBL via chembl_webresource_client, normalizes column names to
lowercase snake_case, verifies the schema expected by
EDA/filter_chembl_statins.py, prints summary stats, and writes a
tab-separated file `chembl402_activities.tsv` in the current directory.

Required output columns (checked against the filter script):
    canonical_smiles, standard_type, standard_relation,
    standard_units, standard_value, assay_type, molecule_chembl_id
"""

import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from chembl_webresource_client.new_client import new_client

TARGET_CHEMBL_ID = "CHEMBL402"
OUTPUT_TSV = Path("chembl402_activities.tsv")

REQUIRED_COLS = [
    "canonical_smiles",
    "standard_type",
    "standard_relation",
    "standard_units",
    "standard_value",
    "assay_type",
    "molecule_chembl_id",
]


def to_snake_case(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    return name.lower()


def fetch_activities(target_id: str) -> pd.DataFrame:
    activity = new_client.activity
    qs = activity.filter(target_chembl_id=target_id)

    try:
        total = len(qs)
        print(f"[Fetch] target={target_id}  total activity rows reported: {total}")
    except Exception:
        total = None
        print(f"[Fetch] target={target_id}  total count unavailable, streaming...")

    rows = []
    start = time.time()
    last_print = start
    for i, rec in enumerate(qs):
        rows.append(rec)
        now = time.time()
        if now - last_print > 5.0:
            pct = f"{100.0 * (i + 1) / total:5.1f}%" if total else "   ?%"
            print(f"  progress: {i + 1:>7d} rows  ({pct})  elapsed {now - start:6.1f}s",
                  flush=True)
            last_print = now
    elapsed = time.time() - start
    print(f"[Fetch] pulled {len(rows)} rows in {elapsed:.1f}s")
    return pd.DataFrame(rows)


def main():
    df = fetch_activities(TARGET_CHEMBL_ID)

    if df.empty:
        print("ERROR: no activity rows returned.", file=sys.stderr)
        sys.exit(1)

    df.columns = [to_snake_case(c) for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"WARNING: missing required columns: {missing}", file=sys.stderr)
        print(f"         available columns: {sorted(df.columns)}", file=sys.stderr)
    else:
        print("[Schema] all 7 required columns present.")

    print("\n===== Summary =====")
    print(f"total rows:       {len(df)}")
    if "molecule_chembl_id" in df.columns:
        print(f"unique molecules: {df['molecule_chembl_id'].nunique()}")
    if "standard_type" in df.columns:
        print("\nstandard_type distribution (top 20):")
        print(df["standard_type"].fillna("<NA>").value_counts().head(20).to_string())
    if "assay_type" in df.columns:
        print("\nassay_type distribution:")
        print(df["assay_type"].fillna("<NA>").value_counts().to_string())
    print("===================\n")

    df.to_csv(OUTPUT_TSV, sep="\t", index=False)
    size_mb = os.path.getsize(OUTPUT_TSV) / (1024 * 1024)
    print(f"[Save] wrote {OUTPUT_TSV} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
