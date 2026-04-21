"""
_verify_marketed_statins.py

Re-runnable verifier for `marketed_statins.py`.

For each statin in `MARKETED_STATINS`:
    1. Parse it with RDKit (sanity)
    2. Fetch the canonical SMILES from PubChem via PUG REST
    3. RDKit-canonicalize both (no stereo) and compare
    4. Print a side-by-side table with PubChem CID and match status

Exits non-zero if any drug fails to parse or fails to match PubChem.

Run:
    python EDA/_verify_marketed_statins.py
"""

import json
import subprocess
import sys
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).parent))
from marketed_statins import MARKETED_STATINS, PUBCHEM_CIDS  # noqa: E402


def rdkit_canon_no_stereo(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m, isomericSmiles=False)


def fetch_pubchem(name: str):
    """Use curl (more robust than urllib here) to fetch PubChem properties."""
    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           f"{name}/property/SMILES,ConnectivitySMILES,MolecularWeight/JSON")
    try:
        r = subprocess.run(["curl", "-sS", "--max-time", "20", url],
                           capture_output=True, text=True, check=True)
        prop = json.loads(r.stdout)["PropertyTable"]["Properties"][0]
        return {
            "cid":    prop.get("CID"),
            "smi":    prop.get("SMILES", ""),
            "conn":   prop.get("ConnectivitySMILES", ""),
            "mw":     prop.get("MolecularWeight", ""),
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    print(f"{'name':<14}  {'CID':>8}  {'MW':>6}  {'formula':<14}  {'match':<6}")
    print("=" * 72)
    all_ok = True
    for name, mine in MARKETED_STATINS.items():
        mine_canon = rdkit_canon_no_stereo(mine)
        if mine_canon is None:
            print(f"{name:<14}  parse FAIL ({mine})")
            all_ok = False
            continue

        m = Chem.MolFromSmiles(mine_canon)
        mw = Descriptors.MolWt(m)
        formula = rdMolDescriptors.CalcMolFormula(m)

        pc = fetch_pubchem(name)
        if "error" in pc:
            print(f"{name:<14}  PubChem fetch FAIL: {pc['error']}")
            all_ok = False
            continue

        pc_canon = rdkit_canon_no_stereo(pc["conn"] or pc["smi"])
        match = mine_canon == pc_canon
        if not match:
            all_ok = False

        # Sanity-check the CID matches the one we declared
        cid_ok = (pc["cid"] == PUBCHEM_CIDS.get(name))
        cid_flag = "" if cid_ok else f" (CID drift from {PUBCHEM_CIDS.get(name)})"

        print(f"{name:<14}  {pc['cid']:>8}  {mw:6.1f}  {formula:<14}  "
              f"{'YES' if match else 'NO':<6}{cid_flag}")
        if not match:
            print(f"  mine:    {mine_canon}")
            print(f"  pubchem: {pc_canon}")

    print()
    if all_ok:
        print("All 7 marketed statins verified against PubChem.")
        return 0
    else:
        print("VERIFICATION FAILED — see mismatches above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
