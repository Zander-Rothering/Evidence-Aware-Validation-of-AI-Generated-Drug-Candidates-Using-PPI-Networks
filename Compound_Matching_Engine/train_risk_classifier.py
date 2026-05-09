"""train_risk_classifier.py — fit the A10b RF + RBF-SVM ensemble.

Class assignment:
    LOW  (y=0) -- ChEMBL CHEMBL402 actives loaded by CompoundLoader.
    HIGH (y=1) -- preferred:  ZINC250K compounds with Tanimoto < 0.40
                              vs the active set, taken from a local CSV.
                  fallback:   ChEMBL inactives with IC50 > 10 uM
                              (only available when CompoundLoader hits the
                              live API — cached / fallback libraries do not
                              expose inactives).

If neither source is usable, the script prints exactly which files are
missing rather than attempting to train on a single class.

Feature vector (per compound, 2057 features total):
    2048 Morgan radius-2 fingerprint bits
    +  9 scalar features:
       tanimoto_nn, scaffold_tanimoto, mw, logp, hbd, hba, qed, tpsa, rot_bonds

Output:
    Compund_Matching_Engine/models/risk_classifier.pkl
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs

from .compound_loader import CompoundLoader
from .fingerprint_encoder import FingerprintEncoder
from .Drug_likeness_filter import DrugLikenessFilter
from .Scaffold_extractor import ScaffoldExtractor
from .risk_classifier import RiskClassifier


HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
MODEL_PATH = MODELS_DIR / "risk_classifier.pkl"

# Locations the script will probe for a ZINC250K SMILES list. First match wins.
# Each file must have a `smiles` column (header) or be a single-column .smi file.
ZINC_CANDIDATE_PATHS = [
    HERE.parent / "datasets" / "zinc250k.csv",
    HERE.parent / "EDA" / "zinc250k.csv",
    HERE.parent / "EDA" / "250k_rndm_zinc_drugs_clean_3.csv",
]


# ---------------------------------------------------------------------------
# Feature-vector helper
# ---------------------------------------------------------------------------
def _build_feature_vector(
    mol: Chem.Mol,
    *,
    encoder: FingerprintEncoder,
    filter_engine: DrugLikenessFilter,
    scaffold_engine: ScaffoldExtractor,
    reference_fps: list,
    nn_mol_lookup: dict,
) -> np.ndarray | None:
    """Return the 2057-dim feature vector for `mol`, or None if rdkit dies."""
    fp = encoder.encode(mol)
    if fp is None:
        return None
    bits = encoder.to_numpy(fp)

    sims = DataStructs.BulkTanimotoSimilarity(fp, reference_fps)
    if not sims:
        return None
    nn_idx = int(np.argmax(sims))
    tanimoto_nn = float(sims[nn_idx])

    nn_mol = nn_mol_lookup.get(nn_idx)
    scaffold_tanimoto = 0.0
    if nn_mol is not None:
        sc_res = scaffold_engine.extract(mol, nn_mol)
        scaffold_tanimoto = float(sc_res.scaffold_similarity)

    fr = filter_engine.filter(mol)
    scalars = np.asarray(
        [
            tanimoto_nn,
            scaffold_tanimoto,
            fr.mw,
            fr.logp,
            fr.hbd,
            fr.hba,
            fr.qed,
            fr.tpsa,
            fr.rot_bonds,
        ],
        dtype=np.float32,
    )
    return np.concatenate([bits.astype(np.float32), scalars])


# ---------------------------------------------------------------------------
# Training-set builders
# ---------------------------------------------------------------------------
def _load_actives(target: str = "CHEMBL402") -> tuple[list, list, list]:
    """Return parallel lists (mols, smiles, fingerprints) of LOW-risk actives."""
    loader = CompoundLoader(target_chembl_id=target)
    library = loader.load_reference_library()
    mols, smiles_list, fps = [], [], []
    for name, smi, ic50, fp in library:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mols.append(mol)
        smiles_list.append(smi)
        fps.append(fp)
    return mols, smiles_list, fps


def _zinc_path() -> Path | None:
    for p in ZINC_CANDIDATE_PATHS:
        if p.is_file():
            return p
    return None


def _load_zinc_negatives(active_fps: list, encoder: FingerprintEncoder,
                         max_compounds: int = 2000,
                         tanimoto_cutoff: float = 0.40) -> list[Chem.Mol]:
    """Read the local ZINC CSV and keep compounds with Tanimoto < cutoff vs actives."""
    path = _zinc_path()
    if path is None:
        return []
    print(f"[zinc] reading {path}")
    df = pd.read_csv(path)
    smi_col = "smiles" if "smiles" in df.columns else df.columns[0]

    negatives: list[Chem.Mol] = []
    for smi in df[smi_col].dropna().astype(str):
        smi = smi.strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumHeavyAtoms() < 10:
            continue
        fp = encoder.encode(mol)
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, active_fps)
        if sims and max(sims) < tanimoto_cutoff:
            negatives.append(mol)
        if len(negatives) >= max_compounds:
            break
    print(f"[zinc] kept {len(negatives)} negatives below Tanimoto {tanimoto_cutoff}")
    return negatives


def _load_chembl_inactives(target: str = "CHEMBL402",
                            ic50_threshold_nM: float = 10_000.0,
                            max_compounds: int = 2000) -> list[Chem.Mol]:
    """Live-API fallback: fetch ChEMBL records with IC50 > 10 uM as HIGH-risk."""
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        return []

    print(f"[chembl] fetching inactives (IC50 > {ic50_threshold_nM:.0f} nM) for {target}")
    try:
        results = new_client.activity.filter(
            target_chembl_id=target,
            standard_type="IC50",
            assay_type="B",
        ).only(["canonical_smiles", "standard_value"])
    except Exception as exc:
        print(f"[chembl] live API unreachable: {exc}")
        return []

    inactives: list[Chem.Mol] = []
    seen: set[str] = set()
    for rec in results:
        smi = rec.get("canonical_smiles")
        val = rec.get("standard_value")
        if not smi or not val or smi in seen:
            continue
        try:
            ic50 = float(val)
        except (ValueError, TypeError):
            continue
        if ic50 <= ic50_threshold_nM:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumHeavyAtoms() < 10:
            continue
        seen.add(smi)
        inactives.append(mol)
        if len(inactives) >= max_compounds:
            break
    print(f"[chembl] kept {len(inactives)} inactives")
    return inactives


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    encoder = FingerprintEncoder()
    filter_engine = DrugLikenessFilter()
    scaffold_engine = ScaffoldExtractor()

    print("[actives] loading ChEMBL CHEMBL402 reference set")
    active_mols, _, active_fps = _load_actives()
    if not active_mols:
        print(
            "ERROR: no actives loaded. Check ChEMBL connectivity or that\n"
            "       Compund_Matching_Engine/chembl_cache.pkl exists."
        )
        return 2
    print(f"[actives] {len(active_mols)} compounds")

    high_mols = _load_zinc_negatives(active_fps, encoder)
    if not high_mols:
        zinc_path = _zinc_path()
        if zinc_path is None:
            tried = "\n  ".join(str(p) for p in ZINC_CANDIDATE_PATHS)
            print(
                "[zinc] no local ZINC250K CSV found. Tried:\n  " + tried + "\n"
                "Download from:\n"
                "  https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
                "master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv\n"
                "and save as `datasets/zinc250k.csv` (CSV with a `smiles` column)."
            )
        print("[fallback] trying ChEMBL inactives (IC50 > 10 uM) ...")
        high_mols = _load_chembl_inactives()

    if not high_mols:
        print(
            "\nERROR: no HIGH-class training data available.\n"
            "Cannot train a binary classifier on a single class.\n\n"
            "To proceed, provide ONE of:\n"
            "  1. datasets/zinc250k.csv  -- single-column or CSV with header `smiles`\n"
            "  2. Working network access to ChEMBL so the live API can supply\n"
            "     inactives (IC50 > 10000 nM) for CHEMBL402.\n"
        )
        return 1

    print(f"\n[features] building feature matrix (2048 + 9 = 2057 features)")
    nn_mol_lookup = {i: m for i, m in enumerate(active_mols)}

    rows: list[np.ndarray] = []
    labels: list[int] = []
    for mol in active_mols:
        vec = _build_feature_vector(
            mol,
            encoder=encoder,
            filter_engine=filter_engine,
            scaffold_engine=scaffold_engine,
            reference_fps=active_fps,
            nn_mol_lookup=nn_mol_lookup,
        )
        if vec is not None:
            rows.append(vec)
            labels.append(0)
    for mol in high_mols:
        vec = _build_feature_vector(
            mol,
            encoder=encoder,
            filter_engine=filter_engine,
            scaffold_engine=scaffold_engine,
            reference_fps=active_fps,
            nn_mol_lookup=nn_mol_lookup,
        )
        if vec is not None:
            rows.append(vec)
            labels.append(1)

    X = np.vstack(rows)
    y = np.asarray(labels, dtype=np.int64)
    print(f"[features] X.shape={X.shape}  positives={int((y==1).sum())}  "
          f"negatives={int((y==0).sum())}")

    if int((y == 1).sum()) < 5 or int((y == 0).sum()) < 5:
        print(
            "ERROR: each class needs at least 5 examples for a stratified 80/20 "
            f"split. Got LOW={int((y==0).sum())} HIGH={int((y==1).sum())}."
        )
        return 1

    print("\n[fit] training Random Forest + RBF-SVM ensemble")
    clf = RiskClassifier()
    metrics = clf.fit(X, y, test_size=0.2)
    print("[fit] held-out metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:14s} {v:.4f}")
        else:
            print(f"    {k:14s} {v}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    clf.save(str(MODEL_PATH))
    print(f"\n[save] {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
