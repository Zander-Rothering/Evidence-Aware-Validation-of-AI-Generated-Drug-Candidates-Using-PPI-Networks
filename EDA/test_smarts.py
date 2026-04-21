"""Verify the patched SMARTS against the 8 marketed statins and negative controls."""
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

SMARTS_OPEN_ACID = "[CX4;H1]([OH])[CH2][CX4;H1]([OH])[CH2]C(=O)[O;H1,-1]"
SMARTS_LACTONE   = "O=C1O[CX4;H1,H2][CX4;H2][CX4;H1]([OH])[CX4;H2]1"

p_acid = Chem.MolFromSmarts(SMARTS_OPEN_ACID)
p_lact = Chem.MolFromSmarts(SMARTS_LACTONE)

STATINS = {
    "atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
    "rosuvastatin": "CC(C)c1nc(N(C)S(C)(=O)=O)nc(-c2ccc(F)cc2)c1/C=C/[C@@H](O)C[C@@H](O)CC(=O)O",
    "simvastatin":  "CCC(C)(C)C(=O)O[C@H]1C[C@@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]12",
    "pravastatin":  "CC[C@H](C)C(=O)O[C@H]1C[C@H](O)C=C2C=C[C@H](C)[C@H](CC[C@@H](O)C[C@@H](O)CC(=O)O)[C@@H]12",
    "fluvastatin":  "CC(C)n1c(/C=C/[C@@H](O)C[C@@H](O)CC(=O)O)c(-c2ccc(F)cc2)c2ccccc21",
    "pitavastatin": "OC(=O)C[C@H](O)C[C@H](O)/C=C/c1c(C2CC2)nc2ccccc2c1-c1ccc(F)cc1",
    "lovastatin":   "CCC(C)C(=O)O[C@H]1C[C@@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]12",
    "cerivastatin": "COC[C@@H](c1c(C(C)C)nc(C(C)C)c(/C=C/[C@@H](O)C[C@@H](O)CC(=O)O)c1-c1ccc(F)cc1)",
}

NEGATIVES = {
    "ibuprofen":       "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "aspirin":         "CC(=O)Oc1ccccc1C(=O)O",
    "glucose":         "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "mevalonic_acid":  "OC(=O)C[C@@](O)(C)CCO",
    "camptothecin":    "CC[C@@]1(O)C(=O)OCc2c1cc1n(c2=O)Cc2cc3ccccc3nc2-1",
    "topotecan":       "CC[C@@]1(O)C(=O)OCc2c1cc1n(c2=O)Cc2cc3c(CN(C)C)c(O)ccc3nc2-1",
}

print(f"{'compound':<18} {'acid':<6} {'lact':<6} {'match'}")
print("-" * 45)

all_statins_ok = True
for name, smi in STATINS.items():
    m = Chem.MolFromSmiles(smi)
    a = m.HasSubstructMatch(p_acid) if m else False
    l = m.HasSubstructMatch(p_lact) if m else False
    ok = a or l
    if not ok:
        all_statins_ok = False
    print(f"{name:<18} {str(a):<6} {str(l):<6} {'PASS' if ok else 'FAIL'}")

print()
all_negs_ok = True
for name, smi in NEGATIVES.items():
    m = Chem.MolFromSmiles(smi)
    a = m.HasSubstructMatch(p_acid) if m else False
    l = m.HasSubstructMatch(p_lact) if m else False
    ok = a or l
    should_be = "FAIL (correct)" if not ok else "MATCH (false positive!)"
    if ok:
        all_negs_ok = False
    print(f"{name:<18} {str(a):<6} {str(l):<6} {should_be}")

print()
print(f"All 8 statins match:        {all_statins_ok}")
print(f"All 6 negatives rejected:   {all_negs_ok}")
