"""
augment_smiles.py — SMILES-randomization augmentation for the filtered
statin corpus.

Input:  datasets/statin_filtered.csv   (MolGPT schema: smiles, qed, scaffold_smiles, source)
Output: datasets/statin_augmented.csv  (same schema, ~15x larger)

For each compound we generate 20 random non-canonical SMILES and keep
up to 15 unique ones. qed and scaffold_smiles are molecule-level
invariants (they don't change under SMILES reordering) so we copy them
from the original row. The train/val 90/10 split is re-done at the
MOLECULE level — all variants of a compound end up in the same split
to prevent signal leakage between train and validation.
"""

import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).parent.parent
INPUT  = PROJECT_ROOT / "datasets" / "statin_filtered.csv"
OUTPUT = PROJECT_ROOT / "datasets" / "statin_augmented.csv"

VARIANTS_PER_CMPD = 15       # target unique random SMILES per compound
ATTEMPTS          = 20       # max random draws per compound
VAL_FRAC          = 0.10
SEED              = 42


def randomize_smiles(mol: Chem.Mol, n_attempts: int, target_unique: int) -> list[str]:
    out: set[str] = set()
    for _ in range(n_attempts):
        smi = Chem.MolToSmiles(mol, isomericSmiles=False, doRandom=True, canonical=False)
        if smi:
            out.add(smi)
        if len(out) >= target_unique:
            break
    return list(out)


def main():
    random.seed(SEED)

    df_in = pd.read_csv(INPUT)
    n_in = len(df_in)
    print(f"[Load] {INPUT}: {n_in} compounds")

    # 1) generate variants per compound
    records: list[dict] = []
    variant_counts: list[int] = []
    shortfalls: list[tuple[str, int]] = []

    for _, row in df_in.iterrows():
        smi_canon = row["smiles"]
        mol = Chem.MolFromSmiles(smi_canon)
        if mol is None:
            print(f"  WARNING: could not parse {smi_canon}")
            continue
        variants = randomize_smiles(mol, ATTEMPTS, VARIANTS_PER_CMPD)
        # ensure the canonical form is in the set (augmentation should include the original)
        variants = list(set(variants) | {smi_canon})
        # cap at target count (canonical + up to VARIANTS_PER_CMPD-1 random); keep canonical
        if len(variants) > VARIANTS_PER_CMPD:
            rest = [v for v in variants if v != smi_canon]
            random.shuffle(rest)
            variants = [smi_canon] + rest[: VARIANTS_PER_CMPD - 1]
        variant_counts.append(len(variants))
        if len(variants) < VARIANTS_PER_CMPD:
            shortfalls.append((smi_canon, len(variants)))
        for v in variants:
            records.append({
                "molecule_key":    smi_canon,     # groups variants of the same compound
                "smiles":          v,
                "qed":             row["qed"],
                "scaffold_smiles": row["scaffold_smiles"],
            })

    # 2) molecule-level 90/10 train/val split
    mol_keys = df_in["smiles"].tolist()
    random.shuffle(mol_keys)
    n_val = max(1, round(VAL_FRAC * len(mol_keys)))
    val_set = set(mol_keys[:n_val])
    train_set = set(mol_keys[n_val:])

    df_out = pd.DataFrame(records)
    df_out["source"] = df_out["molecule_key"].apply(lambda k: "val" if k in val_set else "train")
    df_out = df_out[["smiles", "qed", "scaffold_smiles", "source"]]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT, index=False)

    # 3) summary
    print()
    print(f"[Output] {OUTPUT}: {len(df_out)} SMILES rows")
    print(f"  input compounds:          {n_in}")
    print(f"  mean variants/compound:   {sum(variant_counts) / len(variant_counts):.2f}")
    print(f"  min / max variants:       {min(variant_counts)} / {max(variant_counts)}")
    print(f"  compounds at target ({VARIANTS_PER_CMPD}): "
          f"{sum(1 for c in variant_counts if c == VARIANTS_PER_CMPD)}")
    if shortfalls:
        print(f"  compounds under target:   {len(shortfalls)}")
        for smi, n in shortfalls[:5]:
            print(f"    {n:2d} variants : {smi}")
        if len(shortfalls) > 5:
            print(f"    ... and {len(shortfalls) - 5} more")
    print(f"  train molecules:          {len(train_set)}")
    print(f"  val molecules:            {len(val_set)}")
    print(f"  train rows:               {(df_out['source'] == 'train').sum()}")
    print(f"  val rows:                 {(df_out['source'] == 'val').sum()}")


if __name__ == "__main__":
    main()
