# MolGPT Fine-Tuning on HMGCR Inhibitors

> **Project:** Evidence-Aware Validation of AI-Generated Drug Candidates  
> **Target:** HMGCR — HMG-CoA Reductase · ChEMBL target CHEMBL1781  
> **Hardware:** Mac M-series GPU (Apple Metal / MPS)

---

MolGPT was fine-tuned on 665 HMGCR inhibitor SMILES over 50 epochs, with training loss converging from 1.44 to 0.80. An initial generation of 1,000 candidates produced 13 non-trivial valid compounds. To increase the candidate pool, a second generation run of 10,000 candidates was performed, producing 125 non-trivial valid compounds. Of these, 101 scored above Tanimoto 0.40 and 70 scored above 0.60 against the ChEMBL HMGCR reference set, with 99 passing both Lipinski and PAINS filters. The mean Tanimoto of 0.659 is nearly 3x higher than ZINC250K (0.225), which produced only 9 hits above 0.60 from 100,000 candidates.

Training compounds: 665 (after cleaning)
Epochs: 50
Training loss: 1.44 to 0.80
Generation run 1: 1,000 candidates → 13 non-trivial valid compounds
Generation run 2: 10,000 candidates → 125 non-trivial valid compounds
Hits above Tanimoto 0.40: 101 / 125 (81%)
Hits above Tanimoto 0.60: 70 / 125 (56%)
Lipinski pass: 101 / 101 (100%)
PAINS clean: 99 / 101 (98%)
Clean candidates: 99
Mean Tanimoto vs ChEMBL: 0.659 vs 0.225 for ZINC250K
ZINC hits above 0.60: 9 / 100,000 (0.009%)

## Setup

```bash
git clone https://github.com/devalab/molgpt.git
cd molgpt
mkdir -p ../cond_gpt/weights

pip install torch numpy pandas tqdm wandb six
pip install rdkit
pip install git+https://github.com/molecularsets/moses.git
pip install chembl-webresource-client
```

---

## Steps

### 1 — Prepare training data

```python
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd

activity = new_client.activity
acts = activity.filter(
    target_chembl_id="CHEMBL1781",
    standard_type="IC50"
).only("canonical_smiles")
chembl_smiles = [a["canonical_smiles"] for a in acts if a["canonical_smiles"]]

rows = []
for smi in chembl_smiles:
    mol = Chem.MolFromSmiles(smi.strip())
    if mol is None: continue
    clean_smi = Chem.MolToSmiles(mol, isomericSmiles=False)  # strip stereochemistry
    if "." in clean_smi: continue                             # filter disconnected SMILES
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except:
        scaffold = ""
    if "." in scaffold: scaffold = ""
    rows.append({"smiles": clean_smi, "qed": round(QED.qed(mol), 3), "scaffold_smiles": scaffold})

df = pd.DataFrame(rows)
split = int(len(df) * 0.9)
df["source"] = "val"             # must be "val" not "test"
df.loc[:split-1, "source"] = "train"
df.to_csv("datasets/statin.csv", index=False)
```

### 2 — Reconstruct vocabulary file

Training never saves `statin_stoi.json` automatically. Reconstruct from the hardcoded vocab in `train.py`:

```python
import json

whole_string = [
    "#", "%10", "%11", "%12", "(", ")", "-", "1", "2", "3", "4", "5",
    "6", "7", "8", "9", "<", "=", "B", "Br", "C", "Cl", "F", "I",
    "N", "O", "P", "S", "[B-]", "[BH-]", "[BH2-]", "[BH3-]", "[B]",
    "[C+]", "[C-]", "[CH+]", "[CH-]", "[CH2+]", "[CH2]", "[CH]",
    "[F+]", "[H]", "[I+]", "[IH2]", "[IH]", "[N+]", "[N-]", "[NH+]",
    "[NH-]", "[NH2+]", "[NH3+]", "[N]", "[O+]", "[O-]", "[OH+]", "[O]",
    "[P+]", "[PH+]", "[PH2+]", "[PH]", "[S+]", "[S-]", "[SH+]", "[SH]",
    "[Se+]", "[SeH+]", "[SeH]", "[Se]", "[Si-]", "[SiH-]", "[SiH2]",
    "[SiH]", "[Si]", "[b-]", "[bH-]", "[c+]", "[c-]", "[cH+]", "[cH-]",
    "[n+]", "[n-]", "[nH+]", "[nH]", "[o+]", "[s+]", "[sH+]", "[se+]",
    "[se]", "b", "c", "n", "o", "p", "s"
]
stoi = {c: i for i, c in enumerate(whole_string)}
with open("statin_stoi.json", "w") as f:
    json.dump(stoi, f)
print(f"Vocab size: {len(stoi)}")  # should print 94
```

### 3 — Train

```bash
export KMP_DUPLICATE_LIB_OK=TRUE

python train/train.py \
    --run_name statin_model \
    --data_name statin \
    --n_layer 4 --n_head 4 --n_embd 256 \
    --max_epochs 50 --batch_size 32 --learning_rate 0.0001
```

Select **3** (offline) when wandb prompts.

### 4 — Fix checkpoint mismatch

```python
import torch
ckpt = torch.load("../cond_gpt/weights/statin_model.pt")
for k in [k for k in ckpt if "mask" in k]:
    del ckpt[k]
torch.save(ckpt, "../cond_gpt/weights/statin_model_fixed.pt")
```

### 5 — Fix CUDA to Apple Metal

```bash
find generate/ -name "*.py" -exec sed -i "" \
    "s/\.cuda()/.to('mps')/g; s/to('cuda')/to('mps')/g" {} \;
```

### 6 — Generate

```bash
cp datasets/statin.csv statin.csv
export KMP_DUPLICATE_LIB_OK=TRUE

python generate/generate.py \
    --model_weight ../cond_gpt/weights/statin_model_fixed.pt \
    --csv_name generated_statins --data_name statin \
    --n_layer 4 --n_head 4 --n_embd 256 \
    --gen_size 1000 --batch_size 32 --vocab_size 94 --block_size 151
```

---

## Challenges and Fixes

**1. pip install molsets fails — pomegranate C++ error on Python 3.13**  
Fix: Download MOSES CSV directly from GitHub  
Reference: https://github.com/jmschrei/pomegranate/issues

**2. git clone moses fails — git-lfs not installed**  
Fix: `brew install git-lfs && git lfs install`  
Reference: https://git-lfs.com

**3. Wrong moses installed — PyPI proxy library shares same name**  
Fix: Uninstall; reinstall from molecularsets GitHub  
Reference: https://github.com/molecularsets/moses

**4. rdkit.six import error — removed in newer RDKit**  
Fix: Patch `sascorer.py` with `from six import iteritems`  
Reference: https://github.com/benjaminp/six · https://www.rdkit.org/docs/Changelog.html

**5. DataFrame.append removed in pandas 2.0**  
Fix: Replace with `DataFrame._append()` in `utils.py`  
Reference: https://pandas.pydata.org/docs/whatsnew/v2.0.0.html

**6. MolGPT not a pip package**  
Fix: Clone repo; run scripts directly  
Reference: https://github.com/devalab/molgpt

**7. learning_rate type=int rejects float values**  
Fix: Patch `train.py` line 49 — `type=int` to `type=float`  
Reference: https://docs.python.org/3/library/argparse.html#type · https://github.com/devalab/molgpt/blob/main/train/train.py

**8. Training CSV missing columns**  
Fix: Add `smiles`, `qed`, `scaffold_smiles`, `source` columns  
Reference: https://github.com/devalab/molgpt/blob/main/train/train.py

**9. source column uses test not val**  
Fix: Replace `test` with `val`  
Reference: https://github.com/devalab/molgpt/blob/main/train/train.py L78

**10. Stereochemistry tokens not in 94-token vocab**  
Fix: `MolToSmiles(isomericSmiles=False)`  
Reference: https://www.rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html

**11. Disconnected SMILES crash — dot notation not handled**  
Fix: Filter all SMILES containing `.` before saving CSV  
Reference: https://github.com/molecularsets/moses/issues

**12. Weights directory missing**  
Fix: `mkdir -p ../cond_gpt/weights`

**13. pos_emb shape 151 vs attention mask shape 253**  
Fix: Remove mask keys from checkpoint; load with `strict=False`  
Reference: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict

**14. CUDA not available on Mac**  
Fix: Replace `.cuda()` with `.to('mps')` across generate/ folder  
Reference: https://pytorch.org/docs/stable/notes/mps.html

**15. statin_stoi.json never saved by training**  
Fix: Reconstruct from `whole_string` in `train.py` (94 tokens)  
Reference: https://github.com/devalab/molgpt/blob/main/train/train.py L123

---

## References

### Papers

1. Bagal S et al. MolGPT. *J Chem Inf Model.* 2022. https://arxiv.org/abs/2004.01507
2. Polykovskiy D et al. MOSES. *Front Pharmacol.* 2020. https://arxiv.org/abs/1811.12823
3. Gaulton A et al. ChEMBL. *Nucleic Acids Res.* 2012. https://doi.org/10.1093/nar/gkr777
4. Irwin JJ et al. ZINC. *J Chem Inf Model.* 2012. https://doi.org/10.1021/ci3001277
5. Rogers D, Hahn M. ECFP Fingerprints. *J Chem Inf Model.* 2010. https://doi.org/10.1021/ci100050t
6. Bickerton GR et al. QED. *Nat Chem.* 2012. https://doi.org/10.1038/nchem.1243
7. Baell JB, Holloway GA. PAINS. *J Med Chem.* 2010. https://doi.org/10.1021/jm901137j
8. Vaswani A et al. Attention is all you need. *NeurIPS.* 2017. https://arxiv.org/abs/1706.03762
9. Paszke A et al. PyTorch. *NeurIPS.* 2019. https://arxiv.org/abs/1912.01703

### GitHub Repositories

- MolGPT — https://github.com/devalab/molgpt (MIT)
- MOSES — https://github.com/molecularsets/moses (MIT)
- RDKit — https://github.com/rdkit/rdkit (BSD-3)
- PyTorch — https://github.com/pytorch/pytorch (BSD-3)
- wandb — https://github.com/wandb/wandb (MIT)
- chembl-webresource-client — https://github.com/chembl/chembl_webresource_client (MIT)
- six — https://github.com/benjaminp/six (MIT)
- pandas — https://github.com/pandas-dev/pandas (BSD-3)

### Databases

- ChEMBL HMGCR target (CHEMBL1781) — https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL1781
- ChEMBL API docs — https://chembl.gitbook.io/chembl-interface-documentation
- ZINC250K CSV — https://github.com/aspuru-guzik-group/chemical_vae/blob/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv
- MOSES benchmark data — https://github.com/molecularsets/moses/tree/master/data

### Documentation

- RDKit docs — https://rdkit.org/docs
- RDKit BulkTanimotoSimilarity (C++ API) — https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html
- RDKit DataStructs C++ source — https://github.com/rdkit/rdkit/blob/master/Code/DataStructs/DataStructs.cpp
- PyTorch MPS — https://pytorch.org/docs/stable/notes/mps.html
- pandas 2.0 migration — https://pandas.pydata.org/docs/whatsnew/v2.0.0.html
- argparse type param — https://docs.python.org/3/library/argparse.html#type
- PyTorch load_state_dict — https://pytorch.org/docs/stable/generated/torch.nn.Module.html
- Transformer explainer — https://poloclub.github.io/transformer-explainer
- Illustrated transformer — https://jalammar.github.io/illustrated-transformer

### BulkTanimotoSimilarity — C++ Implementation

`BulkTanimotoSimilarity` is used in the EDA scoring pipeline for fast similarity computation:

```python
from rdkit.DataStructs import BulkTanimotoSimilarity
scores = BulkTanimotoSimilarity(query_fp, ref_fps)  # single C++ call
```

Unlike looping `TanimotoSimilarity` in Python, `BulkTanimotoSimilarity` computes similarity against an entire list of fingerprints in a single C++ loop with no Python overhead per comparison — typically 10–20× faster.

| Resource | URL |
|---|---|
| RDKit DataStructs Python API | https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html |
| RDKit C++ API | https://www.rdkit.org/docs/cppapi/namespacerdkit_1_1DataStructs.html |
| RDKit DataStructs source | https://github.com/rdkit/rdkit/blob/master/Code/DataStructs/DataStructs.cpp |

**Cite as:**
> Landrum G. RDKit: Open-source cheminformatics. rdkit.org. 2006. `BulkTanimotoSimilarity` implemented in C++ via Boost.Python bindings (`rdkit.DataStructs`).
