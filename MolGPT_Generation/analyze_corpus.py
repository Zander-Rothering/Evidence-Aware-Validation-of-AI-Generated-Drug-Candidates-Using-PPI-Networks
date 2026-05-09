"""
analyze_corpus.py — one-off diagnostic for the CHEMBL402 statin corpus.

Answers:
  (a) The 35 surviving compounds (SMILES + ChEMBL id + best potency)
  (b) Drop counts at each layer (L1/L2/L4 + export dedup)
  (c) Projected survivor count under a relaxed L1:
        IC50/Ki/Kd, relation in {=, <, <=}, value <= 10_000 nM, assay B/F
  (d) Size of the "Inhibition > 50%" secondary pool
"""

import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "EDA"))

# Reuse the exact filter logic so we don't drift from what the pipeline actually does.
from filter_chembl_statins import (  # type: ignore
    layer2_pharmacophore,
    layer4_properties,
    canonicalise_for_molgpt,
)

TSV = Path(__file__).parent / "data" / "chembl402_activities.tsv"


def strict_layer1(df: pd.DataFrame) -> pd.DataFrame:
    """Current pipeline's Layer 1."""
    df = df[df["standard_type"].isin(["IC50", "Ki", "Kd"])]
    df = df[df["standard_relation"] == "="]
    df = df[df["standard_units"] == "nM"]
    df = df[pd.to_numeric(df["standard_value"], errors="coerce").notna()].copy()
    df["standard_value"] = df["standard_value"].astype(float)
    df = df[df["standard_value"] <= 1000.0]
    df = df[df["assay_type"].isin(["B", "F"])]
    df = df.dropna(subset=["canonical_smiles"])
    df = (df.sort_values("standard_value", ascending=True)
            .drop_duplicates(subset=["molecule_chembl_id"], keep="first"))
    return df.reset_index(drop=True)


def relaxed_layer1(df: pd.DataFrame) -> pd.DataFrame:
    """Proposed relaxed Layer 1: <=10 uM, accept =/< /<=."""
    df = df[df["standard_type"].isin(["IC50", "Ki", "Kd"])]
    df = df[df["standard_relation"].isin(["=", "<", "<="])]
    df = df[df["standard_units"] == "nM"]
    df = df[pd.to_numeric(df["standard_value"], errors="coerce").notna()].copy()
    df["standard_value"] = df["standard_value"].astype(float)
    df = df[df["standard_value"] <= 10000.0]
    df = df[df["assay_type"].isin(["B", "F"])]
    df = df.dropna(subset=["canonical_smiles"])
    df = (df.sort_values("standard_value", ascending=True)
            .drop_duplicates(subset=["molecule_chembl_id"], keep="first"))
    return df.reset_index(drop=True)


def run_pipeline(df_raw: pd.DataFrame, layer1_fn, label: str):
    n_raw_rows = len(df_raw)
    n_raw_cmpd = df_raw["molecule_chembl_id"].nunique()

    df1 = layer1_fn(df_raw.copy())
    n1 = len(df1)

    df2 = layer2_pharmacophore(df1.copy())
    n2 = len(df2)

    df4 = layer4_properties(df2.copy())
    n4 = len(df4)

    # Replicate the export-time dedup (non-isomeric canonical SMILES)
    df4 = df4.copy()
    df4["smi_canon"] = df4["canonical_smiles"].apply(canonicalise_for_molgpt)
    df4 = df4[df4["smi_canon"] != ""]
    df_final = df4.drop_duplicates(subset=["smi_canon"]).reset_index(drop=True)
    n_final = len(df_final)

    print(f"\n===== {label} =====")
    print(f"  raw rows / unique compounds:   {n_raw_rows} / {n_raw_cmpd}")
    print(f"  after Layer 1 (activity):      {n1}   (dropped {n_raw_cmpd - n1} compounds; {n_raw_rows - n1} rows)")
    print(f"  after Layer 2 (pharmacophore): {n2}   (dropped {n1 - n2})")
    print(f"  after Layer 4 (properties):    {n4}   (dropped {n2 - n4})")
    print(f"  after final dedup by canon SMI:{n_final}  (dropped {n4 - n_final})")
    return df_final


def main():
    df_raw = pd.read_csv(TSV, sep="\t", low_memory=False)

    # --- strict (current) pipeline ---
    df_strict = run_pipeline(df_raw, strict_layer1, "STRICT (current)")

    # --- (a) enumerate the 35 surviving compounds ---
    print("\n===== (a) 35 surviving compounds (strict pipeline) =====")
    cols = ["molecule_chembl_id", "molecule_pref_name",
            "standard_type", "standard_relation",
            "standard_value", "standard_units", "smi_canon"]
    with pd.option_context("display.max_colwidth", 120, "display.width", 200):
        print(df_strict[cols].sort_values("standard_value").to_string(index=False))

    # --- (c) relaxed pipeline ---
    df_relax = run_pipeline(df_raw, relaxed_layer1, "RELAXED (<=10 uM, =/</<=)")

    # --- (d) Inhibition > 50% pool ---
    inh = df_raw[df_raw["standard_type"] == "Inhibition"].copy()
    inh["val"] = pd.to_numeric(inh["standard_value"], errors="coerce")
    inh50 = inh[inh["val"] > 50.0]
    n_rows = len(inh50)
    n_cmpd = inh50["molecule_chembl_id"].nunique()
    n_cmpd_with_smi = inh50.dropna(subset=["canonical_smiles"])["molecule_chembl_id"].nunique()
    print(f"\n===== (d) Inhibition > 50% secondary pool =====")
    print(f"  rows:               {n_rows}")
    print(f"  unique compounds:   {n_cmpd}")
    print(f"  ... with SMILES:    {n_cmpd_with_smi}")

    print("\nDone.")


if __name__ == "__main__":
    main()
