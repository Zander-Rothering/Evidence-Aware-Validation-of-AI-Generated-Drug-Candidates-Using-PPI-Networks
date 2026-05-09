"""
validation_pipeline.py — Steps 3–5: Filter, Validate, and Report for
MolGPT-generated HMGCR inhibitor candidates.

Requires: rdkit, pandas, requests
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors, DataStructs
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# ---------------------------------------------------------------------------
# SAS scorer — use RDKit's bundled Contrib copy
# ---------------------------------------------------------------------------
_SASCORER = None


def _get_sascorer():
    """Lazy-import sascorer from RDKit Contrib (conda layout)."""
    global _SASCORER
    if _SASCORER is not None:
        return _SASCORER

    # Try several known locations
    try:
        import sascorer as _mod
        _SASCORER = _mod
        return _SASCORER
    except ImportError:
        pass

    import importlib.util
    import os

    import rdkit as _rdkit

    rdkit_file = _rdkit.__file__
    if rdkit_file is None:
        raise ImportError("Cannot determine rdkit install path")

    search_dirs = [
        Path(rdkit_file).parent.parent / "share" / "RDKit" / "Contrib" / "SA_Score",
        Path(rdkit_file).resolve().parents[2] / "share" / "RDKit" / "Contrib" / "SA_Score",
    ]
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        search_dirs.append(Path(conda_prefix) / "share" / "RDKit" / "Contrib" / "SA_Score")

    for sa_dir in search_dirs:
        sa_path = sa_dir / "sascorer.py"
        if sa_path.exists():
            spec = importlib.util.spec_from_file_location("sascorer", str(sa_path))
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _SASCORER = mod
            return _SASCORER

    raise ImportError(
        "Could not locate sascorer.py. Install RDKit via conda or place "
        "sascorer.py (from RDKit Contrib/SA_Score) on PYTHONPATH."
    )


def compute_sa_score(mol: Chem.Mol) -> float:
    """Return the Synthetic Accessibility score (1 = easy, 10 = hard)."""
    return _get_sascorer().calculateScore(mol)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Filter candidates
# ═══════════════════════════════════════════════════════════════════════════

def _build_pains_catalog() -> FilterCatalog:
    """Build a combined PAINS_A + PAINS_B + PAINS_C filter catalog."""
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog(params)


def filter_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply drug-likeness filters to a DataFrame with a 'smiles' column.

    Filters applied (in order):
      1. Valid SMILES parse
      2. Lipinski Rule of Five (MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10)
      3. QED ≥ 0.4
      4. SAS ≤ 6
      5. No PAINS (A/B/C) alerts

    Returns a copy of the DataFrame with computed property columns added
    and only rows that pass all filters retained.
    """
    pains_catalog = _build_pains_catalog()

    records = []
    for _, row in df.iterrows():
        smi = row["smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        qed = QED.qed(mol)
        sas = compute_sa_score(mol)
        pains_hit = pains_catalog.GetFirstMatch(mol) is not None

        # Lipinski
        if mw > 500 or logp > 5 or hbd > 5 or hba > 10:
            continue
        if qed < 0.4:
            continue
        if sas > 6:
            continue
        if pains_hit:
            continue

        rec = row.to_dict()
        rec.update({
            "MW": round(mw, 2),
            "LogP": round(logp, 2),
            "HBD": hbd,
            "HBA": hba,
            "QED": round(qed, 3),
            "SAS": round(sas, 3),
            "PAINS_clean": True,
        })
        records.append(rec)

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Validate against BindingDB
# ═══════════════════════════════════════════════════════════════════════════

BINDINGDB_URL = (
    "https://bindingdb.org/axis2/services/BDBService/"
    "getLigandsByUniprots"
)
HMGCR_UNIPROT = "P04035"


def fetch_bindingdb_compounds(
    uniprot_id: str = HMGCR_UNIPROT,
) -> pd.DataFrame:
    """
    Fetch HMGCR ligands from BindingDB REST API.

    Returns a DataFrame with columns: smiles, affinity_nM
    """
    params = {"uniprot": uniprot_id, "response": "json"}
    try:
        resp = requests.get(BINDINGDB_URL, params=params, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[WARN] BindingDB fetch failed: {exc}")
        print("       Continuing with empty BindingDB set.")
        return pd.DataFrame(columns=["smiles", "affinity_nM"])

    # BindingDB JSON wraps results in an "affinities" → "affinity" list
    data = resp.json()
    affinities = data.get("affinities", {}).get("affinity", [])
    if isinstance(affinities, dict):
        affinities = [affinities]

    records = []
    for entry in affinities:
        smi = entry.get("query_ligand_smiles") or entry.get("ligand_smiles", "")
        if not smi:
            continue
        # Prefer Ki, then IC50, then Kd (in nM)
        aff = entry.get("Ki_nM") or entry.get("IC50_nM") or entry.get("Kd_nM") or ""
        try:
            aff_val = float(str(aff).replace(">", "").replace("<", "").strip())
        except (ValueError, TypeError):
            aff_val = None

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        # Canonicalize
        records.append({
            "smiles": Chem.MolToSmiles(mol),
            "affinity_nM": aff_val,
        })

    result = pd.DataFrame(records).drop_duplicates(subset=["smiles"])
    print(f"  Fetched {len(result)} unique BindingDB compounds for {uniprot_id}")
    return result


def _compute_fingerprints(smiles_list: list[str]) -> list[tuple[str, object]]:
    """Return list of (canonical_smiles, morgan_fp) tuples."""
    gen = GetMorganGenerator(radius=2, fpSize=2048)
    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        results.append((Chem.MolToSmiles(mol), gen.GetFingerprint(mol)))
    return results


def validate_against_bindingdb(
    filtered_df: pd.DataFrame,
    bindingdb_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Tanimoto similarity of each filtered candidate against BindingDB
    compounds.  Adds columns: max_tanimoto_bdb, mean_tanimoto_bdb,
    most_similar_bdb_smiles.
    """
    if bindingdb_df.empty:
        filtered_df = filtered_df.copy()
        filtered_df["max_tanimoto_bdb"] = None
        filtered_df["mean_tanimoto_bdb"] = None
        filtered_df["most_similar_bdb_smiles"] = None
        return filtered_df

    bdb_fps = _compute_fingerprints(bindingdb_df["smiles"].tolist())
    gen = GetMorganGenerator(radius=2, fpSize=2048)

    max_tans, mean_tans, best_smis = [], [], []
    for _, row in filtered_df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            max_tans.append(None)
            mean_tans.append(None)
            best_smis.append(None)
            continue

        qfp = gen.GetFingerprint(mol)
        sims = [
            (DataStructs.TanimotoSimilarity(qfp, bfp), bsmi)
            for bsmi, bfp in bdb_fps
        ]
        scores = [s for s, _ in sims]
        best_sim, best_smi = max(sims, key=lambda x: x[0])
        max_tans.append(round(best_sim, 4))
        mean_tans.append(round(sum(scores) / len(scores), 4))
        best_smis.append(best_smi)

    result = filtered_df.copy()
    result["max_tanimoto_bdb"] = max_tans
    result["mean_tanimoto_bdb"] = mean_tans
    result["most_similar_bdb_smiles"] = best_smis
    return result


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Report
# ═══════════════════════════════════════════════════════════════════════════

def flag_memorized(
    df: pd.DataFrame,
    training_smiles: list[str],
    threshold: float = 0.8,
) -> pd.DataFrame:
    """
    Flag compounds with Tanimoto > threshold vs the ChEMBL training set as
    potentially memorized.  Adds columns: max_tanimoto_train, memorized_flag.
    """
    train_fps = _compute_fingerprints(training_smiles)
    gen = GetMorganGenerator(radius=2, fpSize=2048)

    max_tans = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None or not train_fps:
            max_tans.append(None)
            continue
        qfp = gen.GetFingerprint(mol)
        best = max(
            DataStructs.TanimotoSimilarity(qfp, tfp) for _, tfp in train_fps
        )
        max_tans.append(round(best, 4))

    df = df.copy()
    df["max_tanimoto_train"] = max_tans
    df["memorized_flag"] = df["max_tanimoto_train"].apply(
        lambda x: x is not None and x > threshold
    )
    return df


def flag_bindingdb_hits(
    df: pd.DataFrame,
    threshold: float = 0.6,
) -> pd.DataFrame:
    """Flag compounds with max Tanimoto > threshold vs BindingDB as validated hits."""
    df = df.copy()
    df["bindingdb_hit"] = df["max_tanimoto_bdb"].apply(
        lambda x: x is not None and x > threshold
    )
    return df


def generate_report(
    df: pd.DataFrame,
    total_input: int,
    output_path: str,
) -> None:
    """Write validation_results.csv and print a summary."""
    df.to_csv(output_path, index=False)

    passed = len(df)
    bdb_hits = int(df["bindingdb_hit"].sum()) if "bindingdb_hit" in df.columns else 0
    memorized = int(df["memorized_flag"].sum()) if "memorized_flag" in df.columns else 0
    if "bindingdb_hit" in df.columns and "memorized_flag" in df.columns:
        novel_hits = int((df["bindingdb_hit"] & (df["memorized_flag"] == False)).sum())  # noqa: E712
    else:
        novel_hits = 0

    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Total candidates in:          {total_input}")
    print(f"  Passed filters (Step 3):      {passed}")
    print(f"  BindingDB hits (Tc > 0.6):    {bdb_hits}")
    print(f"  Memorized flagged (Tc > 0.8): {memorized}")
    print(f"  Novel validated hits:         {novel_hits}")
    print(f"  Results written to:           {output_path}")
    print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(
    input_csv: str,
    training_csv: str,
    output_csv: str,
) -> pd.DataFrame:
    """Execute the full Steps 3–5 pipeline and return the final DataFrame."""
    # --- Load inputs ---
    print("[Step 3] Loading candidates...")
    candidates = pd.read_csv(input_csv)
    total_input = len(candidates)
    print(f"  Loaded {total_input} candidates from {input_csv}")

    # --- Step 3: Filter ---
    print("[Step 3] Applying filters (Lipinski, QED, SAS, PAINS)...")
    filtered = filter_candidates(candidates)
    print(f"  {len(filtered)} / {total_input} passed all filters")

    if filtered.empty:
        print("[WARN] No candidates passed filtering. Exiting.")
        generate_report(filtered, total_input, output_csv)
        return filtered

    # --- Step 4: BindingDB validation ---
    print("[Step 4] Fetching BindingDB compounds for HMGCR (P04035)...")
    bdb = fetch_bindingdb_compounds()
    print("[Step 4] Computing Tanimoto similarity vs BindingDB...")
    validated = validate_against_bindingdb(filtered, bdb)

    # --- Step 5: Memorization check & report ---
    print("[Step 5] Checking for memorized compounds vs training set...")
    training_df = pd.read_csv(training_csv)
    training_smiles = training_df["smiles"].dropna().tolist()
    print(f"  Loaded {len(training_smiles)} training SMILES from {training_csv}")

    result = flag_memorized(validated, training_smiles)
    result = flag_bindingdb_hits(result)

    generate_report(result, total_input, output_csv)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validation pipeline (Steps 3-5) for MolGPT HMGCR candidates"
    )
    parser.add_argument(
        "--input", required=True,
        help="CSV with MolGPT-generated candidates (must have 'smiles' column)",
    )
    parser.add_argument(
        "--training_set", required=True,
        help="CSV with ChEMBL training SMILES (must have 'smiles' column)",
    )
    parser.add_argument(
        "--output", default="validation_results.csv",
        help="Output CSV path (default: validation_results.csv)",
    )
    args = parser.parse_args()

    run_pipeline(args.input, args.training_set, args.output)
