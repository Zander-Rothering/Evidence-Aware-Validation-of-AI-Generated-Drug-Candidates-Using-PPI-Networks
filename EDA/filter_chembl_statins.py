"""
filter_chembl_statins.py

Layered filter to reduce CHEMBL1781 (HMGCR) down to genuine statin-like
inhibitors suitable for MolGPT fine-tuning.

Layers:
  1. Activity   -- potent binding/functional assays only, IC50/Ki/Kd <= 1 uM
  2. Pharmacophore -- require HMG-like warhead (open dihydroxy acid OR delta-lactone)
  3. Scaffold sanity -- Butina cluster and report; manual review hook provided
  4. Property window -- MW / LogP / HBD / HBA / RotB typical for statins

Usage:
    python filter_chembl_statins.py \
        --input  chembl1781_activities.tsv \
        --output statin_filtered.csv \
        --report filter_report.txt

Input expectations:
    A ChEMBL activity export TSV with at least these columns:
        canonical_smiles, standard_type, standard_relation,
        standard_units, standard_value, assay_type, molecule_chembl_id
    (ChEMBL web interface: Download -> TSV from the target's Activities tab.)

Author: Girish
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from rdkit import DataStructs

RDLogger.DisableLog("rdApp.*")  # silence RDKit parse warnings


# ---------------------------------------------------------------------------
# Pharmacophore SMARTS -- the statin HMG-mimetic warhead
# ---------------------------------------------------------------------------
# Open 3,5-dihydroxy pentanoic acid tail -- matches atorva, prava (saturated
# linker) AND rosuva, fluva, pita, ceriva (vinyl linker) because the warhead
# itself begins at the first CH(OH).  Also accepts deprotonated carboxylate
# for ChEMBL salt entries.
SMARTS_OPEN_ACID = "[CX4;H1]([OH])[CH2][CX4;H1]([OH])[CH2]C(=O)[O;H1,-1]"
# Delta-valerolactone with beta-hydroxy (simva, lova -- prodrug/closed form)
SMARTS_LACTONE   = "O=C1O[CX4;H1,H2][CX4;H2][CX4;H1]([OH])[CX4;H2]1"

PHARMACOPHORE_PATTERNS = [
    Chem.MolFromSmarts(SMARTS_OPEN_ACID),
    Chem.MolFromSmarts(SMARTS_LACTONE),
]


# ---------------------------------------------------------------------------
# Property window for statin-like molecules
# ---------------------------------------------------------------------------
PROP_WINDOW = {
    "MW":   (380.0, 600.0),   # 20% buffer on 400-560
    "LogP": (1.0,   6.0),     # widened ceiling to admit atorvastatin (clogP 5.7)
    "HBD":  (1,     5),
    "HBA":  (4,     10),
    "RotB": (6,     14),
}


# ---------------------------------------------------------------------------
# Layer 1 -- Activity filter
# ---------------------------------------------------------------------------
def layer1_activity(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["standard_type"].isin(["IC50", "Ki", "Kd"])]
    df = df[df["standard_relation"] == "="]
    df = df[df["standard_units"] == "nM"]
    df = df[pd.to_numeric(df["standard_value"], errors="coerce").notna()].copy()
    df["standard_value"] = df["standard_value"].astype(float)
    df = df[df["standard_value"] <= 1000.0]          # <= 1 uM
    df = df[df["assay_type"].isin(["B", "F"])]
    df = df.dropna(subset=["canonical_smiles"])
    # keep the MOST POTENT measurement per compound (lowest IC50/Ki/Kd)
    df = (df.sort_values("standard_value", ascending=True)
            .drop_duplicates(subset=["molecule_chembl_id"], keep="first"))
    print(f"[Layer 1] activity filter: {before} -> {len(df)}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Layer 2 -- Pharmacophore filter
# ---------------------------------------------------------------------------
def has_statin_warhead(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return any(mol.HasSubstructMatch(p) for p in PHARMACOPHORE_PATTERNS if p is not None)


def layer2_pharmacophore(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    mask = df["canonical_smiles"].apply(has_statin_warhead)
    df = df[mask]
    print(f"[Layer 2] pharmacophore (HMG warhead): {before} -> {len(df)}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Layer 3 -- Scaffold sanity (Butina cluster report)
# ---------------------------------------------------------------------------
def compute_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaf) if scaf is not None else ""


def butina_cluster(smiles_list, cutoff: float = 0.4):
    from rdkit.Chem import rdFingerprintGenerator
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps  = [fpgen.GetFingerprint(m) for m in mols]
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - x for x in sims])
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    return clusters


def layer3_scaffold_report(df: pd.DataFrame, report_path: Path) -> pd.DataFrame:
    if len(df) == 0:
        return df
    df = df.copy()
    df["scaffold"] = df["canonical_smiles"].apply(compute_scaffold)
    clusters = butina_cluster(df["canonical_smiles"].tolist(), cutoff=0.4)
    with report_path.open("a") as fh:
        fh.write("\n[Layer 3] Butina scaffold clusters (cutoff 0.4 on Morgan2 fps)\n")
        fh.write(f"  total clusters: {len(clusters)}\n")
        fh.write("  top 10 clusters by size (idx: size, exemplar SMILES):\n")
        sorted_clusters = sorted(clusters, key=len, reverse=True)
        for i, c in enumerate(sorted_clusters[:10]):
            exemplar = df.iloc[c[0]]["canonical_smiles"]
            fh.write(f"    cluster {i}: size={len(c)}  ex={exemplar}\n")
    print(f"[Layer 3] scaffold cluster report written to {report_path}")
    # This layer does NOT drop -- manual review is the hook.
    # If a reviewer wants to drop specific cluster indices, they can do it
    # downstream using the cluster label saved below.
    cluster_of = {}
    for cid, c in enumerate(sorted_clusters):
        for idx in c:
            cluster_of[idx] = cid
    df["scaffold_cluster"] = [cluster_of.get(i, -1) for i in range(len(df))]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Layer 4 -- Property window
# ---------------------------------------------------------------------------
def compute_props(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MW":   Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBD":  Lipinski.NumHDonors(mol),
        "HBA":  Lipinski.NumHAcceptors(mol),
        "RotB": Lipinski.NumRotatableBonds(mol),
    }


def in_window(props) -> bool:
    if props is None:
        return False
    for k, (lo, hi) in PROP_WINDOW.items():
        if not (lo <= props[k] <= hi):
            return False
    return True


def layer4_properties(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    prop_rows = df["canonical_smiles"].apply(compute_props)
    prop_df = pd.DataFrame(prop_rows.tolist())
    df = pd.concat([df.reset_index(drop=True), prop_df.reset_index(drop=True)], axis=1)
    df = df[df.apply(lambda r: in_window({k: r[k] for k in PROP_WINDOW}), axis=1)]
    print(f"[Layer 4] property window: {before} -> {len(df)}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Final export -- strip stereochem, drop disconnected, MolGPT-ready CSV
# ---------------------------------------------------------------------------
def canonicalise_for_molgpt(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    smi = Chem.MolToSmiles(mol, isomericSmiles=False)
    if "." in smi:          # drop disconnected fragments
        return ""
    return smi


def export_molgpt_csv(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    from rdkit.Chem import QED
    df = df.copy()
    df["smiles"] = df["canonical_smiles"].apply(canonicalise_for_molgpt)
    df = df[df["smiles"] != ""]
    df = df.drop_duplicates(subset=["smiles"]).reset_index(drop=True)

    # QED + scaffold + source columns (MolGPT required schema)
    def _qed(s):
        m = Chem.MolFromSmiles(s)
        return QED.qed(m) if m is not None else 0.0
    def _scaf(s):
        m = Chem.MolFromSmiles(s)
        if m is None:
            return ""
        sc = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc, isomericSmiles=False) if sc else ""

    df["qed"]            = df["smiles"].apply(_qed)
    df["scaffold_smiles"]= df["smiles"].apply(_scaf)

    # 90/10 train/val split (use "val" not "test" -- MolGPT requirement)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_val = max(1, int(0.1 * len(df)))
    df["source"] = ["val"] * n_val + ["train"] * (len(df) - n_val)

    out_df = df[["smiles", "qed", "scaffold_smiles", "source"]]
    out_df.to_csv(out_path, index=False)
    print(f"[Export] wrote {len(out_df)} compounds -> {out_path}")
    return out_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="ChEMBL activities TSV")
    ap.add_argument("--output", default="statin_filtered.csv")
    ap.add_argument("--report", default="filter_report.txt")
    ap.add_argument("--drop-clusters", default="",
                    help="Comma-separated scaffold_cluster IDs to drop after manual review")
    args = ap.parse_args()

    report = Path(args.report)
    report.write_text(f"Filter report for {args.input}\n" + "=" * 60 + "\n")

    df = pd.read_csv(args.input, sep="\t", low_memory=False)
    print(f"[Load] {args.input}: {len(df)} rows")

    df = layer1_activity(df)
    df = layer2_pharmacophore(df)
    df = layer3_scaffold_report(df, report)

    if args.drop_clusters:
        drop_ids = {int(x) for x in args.drop_clusters.split(",") if x.strip()}
        before = len(df)
        df = df[~df["scaffold_cluster"].isin(drop_ids)].reset_index(drop=True)
        print(f"[Manual] dropped clusters {sorted(drop_ids)}: {before} -> {len(df)}")

    df = layer4_properties(df)

    if len(df) == 0:
        print("WARNING: no compounds survived filtering.", file=sys.stderr)
        sys.exit(1)

    export_molgpt_csv(df, Path(args.output))

    with report.open("a") as fh:
        fh.write(f"\nFinal surviving compounds: {len(df)}\n")
    print("Done.")


if __name__ == "__main__":
    main()
