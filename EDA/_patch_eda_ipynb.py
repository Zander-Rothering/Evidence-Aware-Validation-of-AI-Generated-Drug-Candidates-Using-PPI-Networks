"""
_patch_eda_ipynb.py

Conservative patch for EDA/EDA.ipynb:
    1. Cell 3: CHEMBL1781 -> CHEMBL402, add LOCAL_PATH fallback to
       datasets/statin_filtered.csv
    2. Cell 12: replace hardcoded known_statins list with import from
       EDA/marketed_statins.py (single source of truth, PubChem-verified)
    3. Insert section-header markdown cells (top + 6 inline)

No "bonus" cleanups (cell 2 MPS check, cell 4 redundant re-encode,
cell 6 logic bug etc.) — those are deliberately left alone per user's
"minimal diff, low risk" instruction.
"""

from pathlib import Path
import nbformat as nbf

NB_PATH = Path(__file__).parent / "EDA.ipynb"
nb = nbf.read(NB_PATH, as_version=4)


def md(text: str):
    return nbf.v4.new_markdown_cell(text.lstrip("\n"))


# -------- Issue 1: cell 3 (data loading) --------
NEW_CELL_3 = '''# Step 1 — load ZINC250K candidates
url        = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
df         = pd.read_csv(url)
candidates = df["smiles"].tolist()[:100000]
print(f"Candidates: {len(candidates)}")

# Step 2 — load ChEMBL reference (HMG-CoA reductase, CHEMBL402)
# Falls back to the local filtered set if the ChEMBL API is unreachable.
LOCAL_PATH = "../datasets/statin_filtered.csv"
try:
    activity      = new_client.activity
    acts          = activity.filter(
                        target_chembl_id="CHEMBL402",
                        standard_type="IC50"
                    ).only("canonical_smiles")
    chembl_smiles = [a["canonical_smiles"] for a in acts if a["canonical_smiles"]]
    print(f"Reference (ChEMBL CHEMBL402): {len(chembl_smiles)}")
except Exception as e:
    print(f"ChEMBL API unavailable ({type(e).__name__}); "
          f"falling back to {LOCAL_PATH}")
    chembl_smiles = pd.read_csv(LOCAL_PATH)["smiles"].dropna().tolist()
    print(f"Reference (local fallback):  {len(chembl_smiles)}")

# Step 3 — fingerprint generator
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def encode(smi):
    mol = Chem.MolFromSmiles(smi.strip())
    if mol is None: return None
    return gen.GetFingerprint(mol)

# Step 4 — encode reference once
ref_fps = [fp for fp in [encode(s) for s in chembl_smiles] if fp is not None]
print(f"Valid reference fps: {len(ref_fps)}")

# Step 5 — encode all candidates in bulk first
print("Encoding candidates...")
candidate_fps = [(smi, encode(smi)) for smi in candidates]
candidate_fps = [(smi, fp) for smi, fp in candidate_fps if fp is not None]
print(f"Valid candidates: {len(candidate_fps)}")

# Step 6 — score in batches using numpy for max speed
print("Scoring...")
BATCH = 1000
results = []

for i in range(0, len(candidate_fps), BATCH):
    batch = candidate_fps[i:i+BATCH]
    for smi, fp in batch:
        scores = BulkTanimotoSimilarity(fp, ref_fps)
        results.append({"smiles": smi, "tanimoto": max(scores)})
    print(f"  processed {min(i+BATCH, len(candidate_fps))}/{len(candidate_fps)}")

df_results = pd.DataFrame(results)
hits       = df_results[df_results["tanimoto"] > 0.60].copy()
print(f"Scored: {len(df_results)}  Hits >0.60: {len(hits)}")
'''

# -------- Issue 2: cell 12 (marketed-statin comparison) --------
NEW_CELL_12 = '''# Score ChEMBL compounds against the 7 marketed statins.
# Canonical SMILES verified against PubChem PUG REST -- see
# EDA/marketed_statins.py (single source of truth across the project).
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))   # EDA/ when notebook runs from there
from marketed_statins import MARKETED_STATINS

known_statins = list(MARKETED_STATINS.items())
statin_fps = [(name, encode(smi)) for name, smi in known_statins]
statin_fps = [(name, fp) for name, fp in statin_fps if fp is not None]
print(f"Marketed-statin reference fingerprints: {len(statin_fps)}")

scores = []
for smi in chembl_smiles:
    fp = encode(smi)
    if fp is None: continue
    best = max(BulkTanimotoSimilarity(fp, [sfp for _, sfp in statin_fps]))
    scores.append(best)

import pandas as pd
s = pd.Series(scores)
print(s.describe())
print(f"Statin-like >0.70: {(s > 0.70).sum()}")
print(f"Statin-like >0.60: {(s > 0.60).sum()}")
print(f"Statin-like >0.40: {(s > 0.40).sum()}")
'''

# Find the original code cells by searching their distinctive text (robust to
# index shifts if the notebook is patched again later).
def find_cell_index(needle: str) -> int:
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if needle in src and c["cell_type"] == "code":
            return i
    raise RuntimeError(f"could not locate cell containing: {needle!r}")


idx_step1 = find_cell_index("# Step 1 — load candidates")
idx_known = find_cell_index("known_statins = [")

print(f"Modifying code cell {idx_step1} (Step 1 / data loading)")
nb["cells"][idx_step1] = nbf.v4.new_code_cell(NEW_CELL_3)
nb["cells"][idx_step1]["execution_count"] = None
nb["cells"][idx_step1]["outputs"] = []

print(f"Modifying code cell {idx_known} (known_statins -> marketed_statins import)")
nb["cells"][idx_known] = nbf.v4.new_code_cell(NEW_CELL_12)
nb["cells"][idx_known]["execution_count"] = None
nb["cells"][idx_known]["outputs"] = []

# -------- Issue 3: insert section markdowns --------
# Build the new cell list by walking the existing cells and inserting markdowns
# at the right anchors (use distinctive content to find anchors so this stays
# stable across re-runs).

TOP_MD = """# Baseline EDA — ZINC250K vs HMGCR reference

This notebook is the **CPU-only baseline** for the project: it asks how a
generic drug-like compound library (**ZINC250K**) compares to known HMG-CoA
reductase inhibitors (**ChEMBL CHEMBL402**) by Tanimoto similarity, with no
generative model in the loop.

It exists so the MolGPT v2 results in `EDA_MolGPT.ipynb` can be benchmarked
against a "what would random drugs look like" floor. For the GPU/MPS variant
of similarity scoring see `EDA_GPU.ipynb`.
"""

ANCHORS = [
    # (markdown header, distinctive substring of the cell it should go BEFORE)
    ("## Data loading",                "# Step 1 — load ZINC250K candidates"),
    ("## Fingerprint generation",      "# feature extraction for candidates"),
    ("## Similarity scoring",          "#plot similarity distribution"),
    ("## Descriptor analysis",         "# compute descriptors on hits"),
    ("## Filter funnel",               "# STEP 10: Visualisations"),
    ("## Marketed statin comparison",  "# Score ChEMBL compounds against the 7 marketed statins"),
]

new_cells = [md(TOP_MD)]
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    for header, needle in ANCHORS:
        if needle in src:
            new_cells.append(md(header))
            break
    new_cells.append(c)

nb["cells"] = new_cells

nbf.write(nb, NB_PATH)
print()
print(f"Wrote {NB_PATH}")
print(f"  cells: {len(nb['cells'])}  "
      f"(md={sum(c['cell_type']=='markdown' for c in nb['cells'])}, "
      f"code={sum(c['cell_type']=='code' for c in nb['cells'])})")
