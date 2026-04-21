"""
render_v2_top16.py

Render the top-N v2 candidates (deduped by canonical SMILES, ranked by
tanimoto_max_marketed) as a 4x4 grid with annotated badges.

Default reads the full validated CSV; pass --in_csv to point at the
novel-only set. Output filename auto-uses the actual surviving count.
"""

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

ROOT    = Path(__file__).parent
GEN_DIR = ROOT / "generation_run"


def badge(name: str, passed: bool) -> str:
    return f"{name}:{'YES' if passed else 'NO'}"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv",     type=Path, default=GEN_DIR / "v2_validated.csv")
    ap.add_argument("--out_prefix", type=str,  default="v2_top",
                    help="Output PNG basename: actual N is appended (e.g. 'v2_top' -> v2_top14.png)")
    ap.add_argument("--require_pass", action="store_true",
                    help="If set, restrict to rows passing warhead_match AND lipinski_pass AND pains_clean")
    ap.add_argument("--max_n", type=int, default=16)
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.in_csv)
    if args.require_pass:
        df = df[df["warhead_match"] & df["lipinski_pass"] & df["pains_clean"]]
    df = (df.drop_duplicates(subset=["smiles"], keep="first")
            .sort_values("tanimoto_max_marketed", ascending=False)
            .head(args.max_n)
            .reset_index(drop=True))
    n = len(df)
    out_png = GEN_DIR / f"{args.out_prefix}{n}.png"
    print(f"[load] top-{n} deduped from {args.in_csv}")

    mols, legends = [], []
    for i, r in df.iterrows():
        m = Chem.MolFromSmiles(r["smiles"])
        if m is None:
            continue
        mols.append(m)
        line1 = f"#{i + 1}  T={r['tanimoto_max_marketed']:.3f}  ~{r['nearest_marketed_statin']}"
        line2 = (f"{badge('W', r['warhead_match'])}  "
                 f"{badge('L', r['lipinski_pass'])}  "
                 f"{badge('P', r['pains_clean'])}")
        legends.append(line1 + "\n" + line2)

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=4,
        subImgSize=(360, 360),
        legends=legends,
        useSVG=False,
    )
    # MolsToGridImage returns a PIL.Image (when useSVG=False); save it.
    img.save(out_png)
    print(f"[write] {out_png}  ({len(mols)} molecules in 4x{(len(mols) + 3) // 4} grid)")
    print()
    print(f"Top {n} (deduped) summary:")
    for i, r in df.iterrows():
        print(f"  #{i + 1:2d}  T={r['tanimoto_max_marketed']:.3f}  "
              f"~{r['nearest_marketed_statin']:<14s}  "
              f"W={int(r['warhead_match'])} L={int(r['lipinski_pass'])} "
              f"P={int(r['pains_clean'])}  qed={r['qed']:.2f}  "
              f"mw={r['mw']:.0f}  logp={r['logp']:.2f}")


if __name__ == "__main__":
    main()
