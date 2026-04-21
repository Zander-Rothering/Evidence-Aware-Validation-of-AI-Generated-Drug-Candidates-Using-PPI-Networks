"""
validate_statins_v2.py

Run the evidence-aware validation pipeline on the v2 generated candidates:
    1. HMG warhead pharmacophore (open-acid OR lactone SMARTS)
    2. Lipinski Rule of Five (pass = <=1 violation)
    3. PAINS filter (RDKit canonical PAINS catalog)
    4. Tanimoto similarity vs:
         (a) datasets/statin_filtered.csv (the 51-compound CHEMBL402 reference)
         (b) 7 marketed statins (hardcoded SMILES)
       Reports MAX Tanimoto and the name of the nearest marketed statin.

Output:
    generation_run/v2_validated.csv   ranked by tanimoto_max_marketed (desc)
"""

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.*")

ROOT       = Path(__file__).parent
GEN_DIR    = ROOT / "generation_run"
CHEMBL_CSV = ROOT / "datasets" / "statin_filtered.csv"

# ---------------------------------------------------------------------------
# Pharmacophore SMARTS -- HMG-mimetic warhead
# (Same patterns as EDA/filter_chembl_statins.py; reproduced here so this
# script is self-contained.)
# ---------------------------------------------------------------------------
SMARTS_OPEN_ACID = "[CX4;H1]([OH])[CH2][CX4;H1]([OH])[CH2]C(=O)[O;H1,-1]"
SMARTS_LACTONE   = "O=C1O[CX4;H1,H2][CX4;H2][CX4;H1]([OH])[CX4;H2]1"
PHARMACOPHORE = [Chem.MolFromSmarts(SMARTS_OPEN_ACID),
                 Chem.MolFromSmarts(SMARTS_LACTONE)]

# ---------------------------------------------------------------------------
# Marketed statins (canonical, no stereo -- Morgan FP ignores stereo by default
# so this is fine for similarity scoring vs the no-stereo generated set)
# ---------------------------------------------------------------------------
MARKETED = {
    "atorvastatin":  "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccc(F)cc2)n(CCC(O)CC(O)CC(=O)O)c1-c1ccccc1",
    "rosuvastatin":  "CC(C)c1nc(N(C)S(C)(=O)=O)nc(-c2ccc(F)cc2)c1C=CC(O)CC(O)CC(=O)O",
    "simvastatin":   "CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12",
    "pravastatin":   "CCC(C)C(=O)OC1CC(O)C=C2C=CC(C)C(CCC(O)CC(O)CC(=O)O)C12",
    "fluvastatin":   "OC(=O)CC(O)CC(O)C=Cc1c(C(C)C)n(Cc2ccc(F)cc2)c2ccccc12",
    "pitavastatin":  "OC(=O)CC(O)CC(O)C=Cc1c(-c2ccccc2)c2ccccc2nc1C1CC1",
    "lovastatin":    "CCC(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C12",
}


def warhead_match(mol: Chem.Mol) -> bool:
    return any(mol.HasSubstructMatch(p) for p in PHARMACOPHORE if p is not None)


def lipinski_check(mol: Chem.Mol):
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = Lipinski.NumHDonors(mol)
    hba  = Lipinski.NumHAcceptors(mol)
    viol = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return mw, logp, hbd, hba, viol, viol <= 1


def build_pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv",  type=Path, default=GEN_DIR / "v2_candidates.csv")
    ap.add_argument("--out_csv", type=Path, default=GEN_DIR / "v2_validated.csv")
    return ap.parse_args()


def main():
    args = parse_args()
    in_csv  = args.in_csv
    out_csv = args.out_csv

    df = pd.read_csv(in_csv)
    print(f"[load] {in_csv}: {len(df)} candidates")

    # Reference fingerprints
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    chembl_smiles = []
    if CHEMBL_CSV.exists():
        cdf = pd.read_csv(CHEMBL_CSV)
        chembl_smiles = cdf["smiles"].dropna().tolist()
    chembl_fps = []
    for s in chembl_smiles:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            chembl_fps.append(fpgen.GetFingerprint(m))
    print(f"[ref] CHEMBL filtered set: {len(chembl_fps)} valid fingerprints")

    marketed_fps = {}
    for name, smi in MARKETED.items():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            print(f"WARN: marketed SMILES failed to parse: {name}")
            continue
        marketed_fps[name] = fpgen.GetFingerprint(m)
    print(f"[ref] marketed statins:    {len(marketed_fps)} fingerprints")

    pains = build_pains_catalog()

    # Score every candidate
    rows = []
    for _, r in df.iterrows():
        smi = r["smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        wh = warhead_match(mol)
        mw, logp, hbd, hba, viol, lip_pass = lipinski_check(mol)
        pains_clean = not pains.HasMatch(mol)
        cand_fp = fpgen.GetFingerprint(mol)

        tan_chembl = 0.0
        if chembl_fps:
            sims = DataStructs.BulkTanimotoSimilarity(cand_fp, chembl_fps)
            tan_chembl = max(sims)

        tan_market_max = 0.0
        nearest = ""
        for name, fp in marketed_fps.items():
            t = DataStructs.TanimotoSimilarity(cand_fp, fp)
            if t > tan_market_max:
                tan_market_max = t
                nearest = name

        rows.append({
            "smiles": smi,
            "warhead_match": wh,
            "lipinski_pass": lip_pass,
            "pains_clean":   pains_clean,
            "qed":  r["qed"],
            "mw":   round(mw, 2),
            "logp": round(logp, 3),
            "tanimoto_max_chembl":   round(tan_chembl, 4),
            "tanimoto_max_marketed": round(tan_market_max, 4),
            "nearest_marketed_statin": nearest,
        })

    out = (pd.DataFrame(rows)
             .sort_values("tanimoto_max_marketed", ascending=False)
             .reset_index(drop=True))
    out.to_csv(out_csv, index=False)
    print(f"[write] {out_csv}: {len(out)} validated rows")

    # Headline numbers
    n = len(out)
    n_wh   = int(out["warhead_match"].sum())
    n_lip  = int(out["lipinski_pass"].sum())
    n_pa   = int(out["pains_clean"].sum())
    n_pass = int((out["warhead_match"] & out["lipinski_pass"] & out["pains_clean"]).sum())

    tm = out["tanimoto_max_marketed"]
    print()
    print("=" * 62)
    print("VALIDATION REPORT")
    print("=" * 62)
    print(f"Total validated:           {n}")
    print(f"Pass warhead_match:        {n_wh}  ({100 * n_wh / n:.1f}%)")
    print(f"Pass lipinski (<=1 viol):  {n_lip}  ({100 * n_lip / n:.1f}%)")
    print(f"Pass PAINS (clean):        {n_pa}  ({100 * n_pa / n:.1f}%)")
    print(f"Pass ALL 4 filters:        {n_pass}  ({100 * n_pass / n:.1f}%)")
    print()
    print("tanimoto_max_marketed distribution:")
    print(f"  mean   {tm.mean():.4f}")
    print(f"  median {tm.median():.4f}")
    print(f"  max    {tm.max():.4f}")
    print(f"  count > 0.4   {int((tm > 0.4).sum())}")
    print(f"  count > 0.5   {int((tm > 0.5).sum())}")
    print()
    print("Top 5 by tanimoto_max_marketed:")
    cols = ["smiles", "tanimoto_max_marketed", "nearest_marketed_statin",
            "tanimoto_max_chembl", "warhead_match", "lipinski_pass",
            "pains_clean", "qed", "mw", "logp"]
    for i, row in out.head(5).iterrows():
        print(f"\n  #{i + 1}  T={row['tanimoto_max_marketed']:.3f} "
              f"vs {row['nearest_marketed_statin']}  T_chembl={row['tanimoto_max_chembl']:.3f}")
        print(f"    SMILES: {row['smiles']}")
        print(f"    warhead={row['warhead_match']}  lipinski_pass={row['lipinski_pass']}  "
              f"pains_clean={row['pains_clean']}")
        print(f"    qed={row['qed']:.3f}  mw={row['mw']}  logp={row['logp']}")


if __name__ == "__main__":
    main()
