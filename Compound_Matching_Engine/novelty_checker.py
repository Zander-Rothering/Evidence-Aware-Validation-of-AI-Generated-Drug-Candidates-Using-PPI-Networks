"""Novelty Checker — Verify whether AI-generated SMILES are present in PubChem.

This module is a standalone Part 1 utility. It is not automatically called by
MatchingEngine in the current MVP pipeline. Instead, downstream orchestrators
such as Evidence_Aggregation/validation_pipeline.py may import NoveltyChecker
to annotate final CSV/JSON outputs with PubChem-level novelty metadata.

Note:
- Girish's *_validated_novel.csv files remove exact ChEMBL duplicates using
  tanimoto_max_chembl < 1.0.
- NoveltyChecker performs a stricter PubChem InChIKey lookup and reports whether
  the candidate is already catalogued in PubChem.

Developed with AI assistance for syntax support.
"""

import requests
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey


class NoveltyChecker:
    """Check whether a SMILES string represents a novel compound not found in PubChem."""

    def __init__(self, check_novelty: bool = False) -> None:
        """Set novelty checking mode.

        check_novelty=False  ->  skip API call, assume novel (fast/dev mode)
        check_novelty=True   ->  query PubChem to verify novelty
        """
        self.check_novelty = check_novelty

    def _smiles_to_inchikey(self, smiles: str) -> str | None:
        """Convert a SMILES string to an InChIKey via RDKit.

        InChIKey guarantees a unique 1:1 mapping for each molecule,
        regardless of how the SMILES is written.
        Returns the InChIKey string, or None if conversion fails.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        inchi = MolToInchi(mol)
        if inchi is None:
            return None
        return InchiToInchiKey(inchi)

    def _query_pubchem(self, inchikey: str) -> bool:
        """Query PubChem REST API to check if a compound exists.

        Returns True if the compound is found (status 200),
        False if not found (status 404).
        On any error or timeout, returns False (fail-open: assume novel).
        """
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
            f"compound/inchikey/{inchikey}/JSON"
        )
        try:
            resp = requests.get(url, timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def is_novel(self, smiles: str) -> dict:
        """Check whether a single SMILES is novel.

        Returns a dict with keys:
          is_novel  — True if the compound is not in PubChem (or check skipped)
          inchikey  — the InChIKey string (or "N/A")
          source    — "pubchem_api", "skipped", or "rdkit_failed"
        """
        if not self.check_novelty:
            return {"is_novel": True, "inchikey": "N/A", "source": "skipped"}

        inchikey = self._smiles_to_inchikey(smiles)
        if inchikey is None:
            print(f"  [NoveltyChecker] RDKit could not convert SMILES: {smiles[:60]}...")
            return {"is_novel": True, "inchikey": "N/A", "source": "rdkit_failed"}

        found_in_pubchem = self._query_pubchem(inchikey)

        return {
            "is_novel": not found_in_pubchem,
            "inchikey": inchikey,
            "source": "pubchem_api",
        }

    def filter_novel(self, smiles_list: list[str]) -> list[str]:
        """Filter a list of SMILES, returning only novel compounds.

        Prints a summary of how many were filtered out.
        """
        novel = []
        known = 0

        for smiles in smiles_list:
            result = self.is_novel(smiles)
            if result["is_novel"]:
                novel.append(smiles)
            else:
                known += 1
                print(f"  [filtered] {smiles[:60]}... "
                      f"(InChIKey: {result['inchikey']})")

        total = len(smiles_list)
        print(f"\nNovelty filter: {len(novel)} novel / "
              f"{known} known / {total} total")
        return novel


if __name__ == "__main__":
    # --- Test 1: known drug (atorvastatin) should be found in PubChem ---
    checker = NoveltyChecker(check_novelty=True)

    atorvastatin = (
        "CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)"
        "C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4"
    )
    print("Test 1 — atorvastatin (known drug, expect is_novel=False)")
    result = checker.is_novel(atorvastatin)
    print(f"  is_novel : {result['is_novel']}")
    print(f"  inchikey : {result['inchikey']}")
    print(f"  source   : {result['source']}")

    # --- Test 2: intentionally weird SMILES unlikely to be in PubChem ---
    novel_smiles = "C1CC(CC(C1)N(C)C(=O)c2cc(F)c(Br)c(OC)c2)S(=O)(=O)N"
    print("\nTest 2 — random novel SMILES (expect is_novel=True)")
    result2 = checker.is_novel(novel_smiles)
    print(f"  is_novel : {result2['is_novel']}")
    print(f"  inchikey : {result2['inchikey']}")
    print(f"  source   : {result2['source']}")

    # --- Test 3: filter_novel on a mixed list ---
    print("\nTest 3 — filter_novel() on mixed list")
    mixed = [atorvastatin, novel_smiles, "CC(=O)Oc1ccccc1C(=O)O"]  # aspirin
    novel_only = checker.filter_novel(mixed)
    print(f"Novel SMILES kept: {len(novel_only)}")
    for s in novel_only:
        print(f"  {s[:60]}...")

    # --- Test 4: fast mode (check_novelty=False) ---
    print("\nTest 4 — fast mode (check_novelty=False)")
    fast_checker = NoveltyChecker(check_novelty=False)
    result4 = fast_checker.is_novel(atorvastatin)
    print(f"  is_novel : {result4['is_novel']}  (always True in fast mode)")
    print(f"  source   : {result4['source']}")
