"""
build_v2_summary.py

Post-processing for the v2 validated set:
    1. Filter v2_validated.csv to NOVEL only (drop tanimoto_max_chembl == 1.000),
       re-rank by tanimoto_max_marketed, save v2_validated_novel.csv
    2. Write a 2-panel summary figure (training loss + Tanimoto histogram)
       to generation_run/v2_summary_figure.png
    3. Write a markdown summary to generation_run/v2_summary.md
       (paste-ready for the report)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT       = Path(__file__).parent
GEN_DIR    = ROOT / "generation_run"
TRAIN_LOG  = ROOT / "molgpt"   / "training_log.csv"
VAL_CSV    = GEN_DIR / "v2_validated.csv"
NOVEL_CSV  = GEN_DIR / "v2_validated_novel.csv"
SUMMARY_PNG = GEN_DIR / "v2_summary_figure.png"
SUMMARY_MD  = GEN_DIR / "v2_summary.md"

# Source-of-truth numbers from the training run + earlier reports
BEST_EPOCH      = 37
BEST_VAL_LOSS   = 0.5997
TRAIN_WALL_SEC  = 3326          # sum of per-epoch wall_time_s
N_GENERATED     = 5000          # total samples
N_VALID         = 283           # parsed-OK with the gen-time filters
N_UNIQUE        = 279
N_NOVEL_GEN     = 276           # vs the 51 filtered training compounds


def filter_novel(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["tanimoto_max_chembl"] < 1.0].copy()
    df = (df.sort_values("tanimoto_max_marketed", ascending=False)
            .reset_index(drop=True))
    print(f"[novel] {before} -> {len(df)} after dropping tanimoto_max_chembl == 1.000")
    return df


def write_summary_figure(novel_df: pd.DataFrame):
    log = pd.read_csv(TRAIN_LOG)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: training curves
    ax_l.plot(log["epoch"], log["train_loss"], label="train_loss", color="#1f77b4", lw=1.6)
    ax_l.plot(log["epoch"], log["val_loss"],   label="val_loss",   color="#d62728", lw=1.6)
    ax_l.axvline(BEST_EPOCH, ls="--", color="gray", lw=1.0,
                 label=f"best @ epoch {BEST_EPOCH} (val={BEST_VAL_LOSS:.4f})")
    ax_l.set_xlabel("Epoch")
    ax_l.set_ylabel("Cross-entropy loss")
    ax_l.set_title("MolGPT v2 fine-tuning (40 epochs, statin_augmented)")
    ax_l.legend(loc="upper right")
    ax_l.grid(alpha=0.3)

    # Right: Tanimoto histogram, color-coded by warhead_match
    tm_yes = novel_df.loc[novel_df["warhead_match"], "tanimoto_max_marketed"]
    tm_no  = novel_df.loc[~novel_df["warhead_match"], "tanimoto_max_marketed"]
    bins   = np.linspace(0, max(0.6, novel_df["tanimoto_max_marketed"].max() + 0.02), 31)
    ax_r.hist([tm_no, tm_yes], bins=bins, stacked=True,
              color=["#bbbbbb", "#2ca02c"],
              label=[f"warhead=False (n={len(tm_no)})",
                     f"warhead=True  (n={len(tm_yes)})"])
    mean = novel_df["tanimoto_max_marketed"].mean()
    med  = novel_df["tanimoto_max_marketed"].median()
    ax_r.axvline(mean, ls="--", color="black", lw=1.2, label=f"mean = {mean:.3f}")
    ax_r.axvline(med,  ls=":",  color="black", lw=1.2, label=f"median = {med:.3f}")
    ax_r.set_xlabel("Tanimoto vs nearest marketed statin")
    ax_r.set_ylabel("Count")
    ax_r.set_title(f"Novel candidate similarity (n={len(novel_df)})")
    ax_r.legend(loc="upper right")
    ax_r.grid(alpha=0.3)

    fig.suptitle("Evidence-aware validation: MolGPT v2 statin candidates", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(SUMMARY_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {SUMMARY_PNG}")


def write_summary_md(novel_df: pd.DataFrame):
    n        = len(novel_df)
    n_wh     = int(novel_df["warhead_match"].sum())
    n_lip    = int(novel_df["lipinski_pass"].sum())
    n_pa     = int(novel_df["pains_clean"].sum())
    n_pass4  = int((novel_df["warhead_match"] & novel_df["lipinski_pass"] & novel_df["pains_clean"]).sum())
    tm       = novel_df["tanimoto_max_marketed"]

    top5 = novel_df.head(5)

    lines = []
    lines.append("# MolGPT v2 — statin candidate generation summary")
    lines.append("")
    lines.append("## Training")
    lines.append("")
    lines.append(f"- Best val loss: **{BEST_VAL_LOSS:.4f}** at epoch **{BEST_EPOCH}** (max_epochs=40, early-stop did not trigger)")
    lines.append(f"- Total wall time: **{TRAIN_WALL_SEC // 60} min {TRAIN_WALL_SEC % 60} s** (~83 s / epoch on Apple MPS)")
    lines.append("- Architecture: 4-layer / 4-head / 256-embd MolGPT, vocab 94, block 78")
    lines.append(f"- Warm-started from `cond_gpt/weights/statin_model.pt` (CHEMBL402 statin checkpoint)")
    lines.append("")
    lines.append("## Generation (T=1.0, 5000 samples)")
    lines.append("")
    lines.append(f"- Total generated: **{N_GENERATED}**")
    lines.append(f"- Valid (parseable, connected, ≥10 heavy atoms): **{N_VALID} ({100 * N_VALID / N_GENERATED:.1f}%)**")
    lines.append(f"- Unique canonical: **{N_UNIQUE}**")
    lines.append(f"- Novel vs 51 training compounds: **{N_NOVEL_GEN}**")
    lines.append("")
    lines.append("## Validation (novel-only, after dropping training-set duplicates)")
    lines.append("")
    lines.append(f"- Novel candidates evaluated: **{n}**")
    lines.append(f"- Pass HMG warhead pharmacophore: **{n_wh} ({100 * n_wh / n:.1f}%)**")
    lines.append(f"- Pass Lipinski (≤1 violation): **{n_lip} ({100 * n_lip / n:.1f}%)**")
    lines.append(f"- Pass PAINS (RDKit canonical, Baell+Holloway A+B+C): **{n_pa} ({100 * n_pa / n:.1f}%)**")
    lines.append(f"- Pass **all 4** filters: **{n_pass4} ({100 * n_pass4 / n:.1f}%)**")
    lines.append("")
    lines.append("## Tanimoto vs marketed statins (Morgan r=2, 2048 bits)")
    lines.append("")
    lines.append(f"- mean: **{tm.mean():.4f}**")
    lines.append(f"- median: **{tm.median():.4f}**")
    lines.append(f"- max: **{tm.max():.4f}**")
    lines.append(f"- count > 0.4: **{int((tm > 0.4).sum())}**")
    lines.append(f"- count > 0.5: **{int((tm > 0.5).sum())}**")
    lines.append("")
    lines.append("## Top 5 novel candidates by tanimoto_max_marketed")
    lines.append("")
    lines.append("| # | T_marketed | nearest statin | T_chembl | warhead | Lip | PAINS | QED | MW | LogP | SMILES |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in top5.iterrows():
        lines.append(
            f"| {i + 1} | {r['tanimoto_max_marketed']:.3f} | {r['nearest_marketed_statin']} | "
            f"{r['tanimoto_max_chembl']:.3f} | "
            f"{'✓' if r['warhead_match'] else '✗'} | "
            f"{'✓' if r['lipinski_pass'] else '✗'} | "
            f"{'✓' if r['pains_clean'] else '✗'} | "
            f"{r['qed']:.2f} | {r['mw']:.0f} | {r['logp']:.2f} | "
            f"`{r['smiles']}` |"
        )
    lines.append("")

    SUMMARY_MD.write_text("\n".join(lines))
    print(f"[md] wrote {SUMMARY_MD}")


def main():
    full = pd.read_csv(VAL_CSV)
    novel = filter_novel(full)
    novel.to_csv(NOVEL_CSV, index=False)
    print(f"[write] {NOVEL_CSV}: {len(novel)} novel candidates")

    write_summary_figure(novel)
    write_summary_md(novel)


if __name__ == "__main__":
    main()
