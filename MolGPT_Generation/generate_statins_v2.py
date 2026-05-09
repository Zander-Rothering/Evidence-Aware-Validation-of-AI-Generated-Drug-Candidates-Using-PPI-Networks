"""
generate_statins_v2.py

Sample 5000 candidate SMILES from the v2 fine-tuned MolGPT
(cond_gpt/weights/statin_model_v2.pt) and filter at generation time.

Outputs:
    generation_run/v2_candidates.csv   smiles, qed, lipinski_violations
    generation_run/v2_valid.smi        canonical SMILES, one per line
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, QED

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "molgpt" / "train"))
from model import GPT, GPTConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Config (matches the augmented training corpus)
# ---------------------------------------------------------------------------
CKPT_PATH   = ROOT / "cond_gpt" / "weights" / "statin_model_v2.pt"
STOI_PATH   = ROOT / "molgpt"   / "statin_stoi.json"
TRAIN_CSV   = ROOT / "datasets" / "statin_filtered.csv"   # 51 filtered training compounds
OUT_DIR     = ROOT / "generation_run"

VOCAB_SIZE  = 94
BLOCK_SIZE  = 78
N_LAYER     = 4
N_HEAD      = 4
N_EMBD      = 256

BATCH_SIZE  = 128
TOP_K       = None
START_TOKEN = "C"
MIN_HEAVY   = 10

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Same regex MolGPT uses for SMILES tokenization
SMILES_REGEX = re.compile(
    r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


def load_model() -> GPT:
    with open(STOI_PATH) as fh:
        stoi = json.load(fh)
    assert len(stoi) == VOCAB_SIZE, f"vocab size {len(stoi)} != expected {VOCAB_SIZE}"

    cfg = GPTConfig(
        vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
        num_props=0, scaffold=False, scaffold_maxlen=0,
        lstm=False, lstm_layers=0,
    )
    model = GPT(cfg)

    state = torch.load(CKPT_PATH, map_location="cpu")
    # Drop attention-mask buffers (they're recreated by the new model with the
    # correct shape; old ckpt was sized for the augmented training run).
    for k in [k for k in list(state.keys()) if "mask" in k]:
        del state[k]
    # Also drop any shape-mismatched keys (e.g. pos_emb if block sizes ever drift).
    msd = model.state_dict()
    for k in list(state.keys()):
        if k in msd and state[k].shape != msd[k].shape:
            print(f"  shape mismatch dropped: {k}  ckpt {tuple(state[k].shape)} != model {tuple(msd[k].shape)}")
            del state[k]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] loaded {len(state)} keys, missing={len(missing)}, unexpected={len(unexpected)}")

    model.eval().to(DEVICE)
    return model, stoi


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--n_samples",   type=int,   default=5000)
    ap.add_argument("--out_prefix",  type=str,   default="v2",
                    help="Output filename prefix (e.g. 'v2' -> v2_candidates.csv, v2_valid.smi)")
    return ap.parse_args()


@torch.no_grad()
def sample_batch(model: GPT, stoi: dict, batch: int, steps: int,
                 temperature: float, top_k):
    itos = {v: k for k, v in stoi.items()}
    start_idx = stoi[START_TOKEN]
    x = torch.full((batch, 1), start_idx, dtype=torch.long, device=DEVICE)

    for _ in range(steps - 1):
        x_cond = x if x.size(1) <= BLOCK_SIZE else x[:, -BLOCK_SIZE:]
        logits, _, _ = model(x_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, ix), dim=1)

    out = []
    for row in x.cpu().tolist():
        s = "".join(itos[i] for i in row)
        s = s.split("<")[0]   # truncate at first pad/EOS
        out.append(s)
    return out


def filter_candidate(smi: str):
    """Return (canonical_smiles, qed, lipinski_violations) or None."""
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    canon = Chem.MolToSmiles(mol, isomericSmiles=False)
    if "." in canon:
        return None
    mol = Chem.MolFromSmiles(canon)
    if mol is None or mol.GetNumHeavyAtoms() < MIN_HEAVY:
        return None
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = Lipinski.NumHDonors(mol)
    hba  = Lipinski.NumHAcceptors(mol)
    viol = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    qed  = QED.qed(mol)
    return canon, qed, viol


def main():
    args = parse_args()
    n_samples   = args.n_samples
    temperature = args.temperature
    out_csv     = OUT_DIR / f"{args.out_prefix}_candidates.csv"
    out_smi     = OUT_DIR / f"{args.out_prefix}_valid.smi"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[device] {DEVICE}")
    print(f"[ckpt] {CKPT_PATH}")
    print(f"[stoi] {STOI_PATH}")
    print(f"[target] {n_samples} samples in batches of {BATCH_SIZE}, "
          f"temperature={temperature}")

    t0 = time.time()
    model, stoi = load_model()

    # Generate
    raw = []
    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE
    for b in range(n_batches):
        bs = min(BATCH_SIZE, n_samples - len(raw))
        out = sample_batch(model, stoi, bs, BLOCK_SIZE, temperature, TOP_K)
        raw.extend(out)
        print(f"  batch {b + 1}/{n_batches}: total raw = {len(raw)}  "
              f"(t={time.time() - t0:.1f}s)")
    raw = raw[:n_samples]

    # Filter
    rows = []
    for s in raw:
        r = filter_candidate(s)
        if r is not None:
            rows.append(r)

    # Novelty vs the 51 filtered training compounds
    train_set = set()
    if TRAIN_CSV.exists():
        td = pd.read_csv(TRAIN_CSV)
        for s in td["smiles"]:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                train_set.add(Chem.MolToSmiles(m, isomericSmiles=False))
    n_train = len(train_set)

    canon_list = [r[0] for r in rows]
    n_total = len(raw)
    n_valid = len(rows)
    n_unique = len(set(canon_list))
    n_novel = sum(1 for s in set(canon_list) if s not in train_set)

    # Save candidates CSV (one row per *valid* generation, duplicates kept)
    df = pd.DataFrame(rows, columns=["smiles", "qed", "lipinski_violations"])
    df.to_csv(out_csv, index=False)

    # Save unique canonical SMILES for downstream tools
    with open(out_smi, "w") as fh:
        for s in sorted(set(canon_list)):
            fh.write(s + "\n")

    wall = time.time() - t0

    # Report
    print()
    print("=" * 62)
    print("GENERATION REPORT")
    print("=" * 62)
    print(f"Total generated:     {n_total}")
    print(f"Valid (parseable):   {n_valid}  ({100 * n_valid / n_total:.1f}%)")
    print(f"Unique canonical:    {n_unique}")
    print(f"Novel vs training:   {n_novel}  (training set size: {n_train})")
    print(f"Wall time:           {wall:.1f}s")
    print()
    print("Top-10 most-frequent canonical SMILES:")
    for smi, c in Counter(canon_list).most_common(10):
        print(f"  {c:5d}  {smi}")
    print()
    if n_valid:
        # MW / LogP need explicit computation since CSV stores only QED + viol
        mws, logps = [], []
        for s in canon_list:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                mws.append(Descriptors.MolWt(m))
                logps.append(Descriptors.MolLogP(m))
        print(f"Mean QED across valid:   {df['qed'].mean():.3f}")
        print(f"Mean MW across valid:    {sum(mws) / len(mws):.1f}")
        print(f"Mean LogP across valid:  {sum(logps) / len(logps):.2f}")
    print()
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_smi}  ({n_unique} unique SMILES)")


if __name__ == "__main__":
    main()
