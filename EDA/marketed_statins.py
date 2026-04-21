"""
marketed_statins.py

Single source of truth for the 7 marketed-statin SMILES used in similarity
scoring throughout this project (validate_statins_v2.py, EDA/EDA.ipynb,
EDA/EDA_GPU.ipynb, EDA/EDA_MolGPT.ipynb, etc.).

All SMILES are verified against PubChem PUG REST `ConnectivitySMILES`
(stereo-stripped) and RDKit-canonicalized to the no-stereo form, matching
the `isomericSmiles=False` convention used elsewhere in the project (the
generated candidates also have no stereochem, and Morgan fingerprints
ignore stereo by default).

Re-run `EDA/_verify_marketed_statins.py` to regenerate the verification.
"""

# Map: drug name -> RDKit-canonical SMILES (no stereo)
MARKETED_STATINS = {
    "atorvastatin":  "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
    "rosuvastatin":  "CC(C)c1nc(N(C)S(C)(=O)=O)nc(-c2ccc(F)cc2)c1C=CC(O)CC(O)CC(=O)O",
    "simvastatin":   "CCC(C)(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C21",
    "pravastatin":   "CCC(C)C(=O)OC1CC(O)C=C2C=CC(C)C(CCC(O)CC(O)CC(=O)O)C21",
    "fluvastatin":   "CC(C)n1c(C=CC(O)CC(O)CC(=O)O)c(-c2ccc(F)cc2)c2ccccc21",
    "pitavastatin":  "O=C(O)CC(O)CC(O)C=Cc1c(C2CC2)nc2ccccc2c1-c1ccc(F)cc1",
    "lovastatin":    "CCC(C)C(=O)OC1CC(C)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C21",
}

# PubChem CIDs (for citation / re-verification)
PUBCHEM_CIDS = {
    "atorvastatin":  60823,
    "rosuvastatin":  446157,
    "simvastatin":   54454,
    "pravastatin":   54687,
    "fluvastatin":   1548972,
    "pitavastatin":  5282452,
    "lovastatin":    53232,
}
