# MolGPT v2 — statin candidate generation summary

## Training

- Best val loss: **0.5997** at epoch **37** (max_epochs=40, early-stop did not trigger)
- Total wall time: **55 min 26 s** (~83 s / epoch on Apple MPS)
- Architecture: 4-layer / 4-head / 256-embd MolGPT, vocab 94, block 78
- Warm-started from `cond_gpt/weights/statin_model.pt` (CHEMBL402 statin checkpoint)

## Generation (T=1.0, 5000 samples)

- Total generated: **5000**
- Valid (parseable, connected, ≥10 heavy atoms): **283 (5.7%)**
- Unique canonical: **279**
- Novel vs 51 training compounds: **276**

## Validation (novel-only, after dropping training-set duplicates)

- Novel candidates evaluated: **276**
- Pass HMG warhead pharmacophore: **61 (22.1%)**
- Pass Lipinski (≤1 violation): **260 (94.2%)**
- Pass PAINS (RDKit canonical, Baell+Holloway A+B+C): **276 (100.0%)**
- Pass **all 4** filters: **59 (21.4%)**

## Tanimoto vs marketed statins (Morgan r=2, 2048 bits)

- mean: **0.3102**
- median: **0.3054**
- max: **0.5507**
- count > 0.4: **46**
- count > 0.5: **3**

## Top 5 novel candidates by tanimoto_max_marketed

| # | T_marketed | nearest statin | T_chembl | warhead | Lip | PAINS | QED | MW | LogP | SMILES |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.551 | rosuvastatin | 0.597 | ✓ | ✓ | ✓ | 0.62 | 401 | 4.23 | `Cc1c(C=CC(O)CC(O)CC(=O)O)nc(-c2ccc(F)cc2)c(C)c1C(C)C` |
| 2 | 0.537 | atorvastatin | 0.514 | ✓ | ✓ | ✓ | 0.38 | 475 | 4.42 | `CC(=O)c1c(F)c(-c2ccc(F)cc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O` |
| 3 | 0.514 | rosuvastatin | 0.815 | ✓ | ✓ | ✓ | 0.59 | 404 | 3.99 | `CC(C)c1nc(-c2ccc(F)cc2)n(C=CC(O)CC(O)CC(=O)O)c1C(C)C` |
| 4 | 0.500 | rosuvastatin | 0.613 | ✓ | ✓ | ✓ | 0.51 | 424 | 3.80 | `Cc1ccc(-c2nc(-c3ccc(F)cc3)n(C)c2C=CC(O)CC(O)CC(=O)O)cc1` |
| 5 | 0.494 | rosuvastatin | 0.550 | ✓ | ✓ | ✓ | 0.38 | 517 | 3.60 | `CC(c1cccc(-c2nn(C)c(-c3ccc(F)cc3)c2C=CC(O)CC(O)CC(=O)O)c1)S(C)(=O)=O` |
