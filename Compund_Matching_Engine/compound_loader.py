"""Activity 4 minimal implementation only so far:
   Build the Reference Library from ChEMBL."""

from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Fallback statins (name, SMILES, IC50 in nM)
FALLBACK_STATINS = [
    ("atorvastatin",
     "CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
     8.0),
    ("rosuvastatin",
     "CC(C)C1=NC(=NC(=C1/C=C/[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)N(C)S(=O)(=O)C",
     5.0),
    ("simvastatin",
     "CCC(C)(C)C(=O)O[C@H]1C[C@H](C=C2[C@H]1[C@H]([C@H](C=C2)C)CC[C@@H]3C[C@H](CC(=O)O3)O)C",
     11.2),
    ("pravastatin",
     "CC[C@H](C)C(=O)O[C@H]1C[C@@H](C=C2[C@H]1[C@H]([C@H](C=C2)C)CC[C@H](C[C@H](CC(=O)O)O)O)O",
     44.1),
    ("fluvastatin",
     "CC(C)N1C2=CC=CC=C2C(=C1/C=C/[C@H](C[C@H](CC(=O)O)O)O)C3=CC=C(C=C3)F",
     28.0),
    ("lovastatin",
     "CC[C@H](C)C(=O)O[C@H]1C[C@H](C=C2[C@H]1[C@H]([C@H](C=C2)C)CC[C@@H]3C[C@H](CC(=O)O3)O)C",
     37.0),
    ("pitavastatin",
     "C1CC1C2=NC3=CC=CC=C3C(=C2/C=C/[C@H](C[C@H](CC(=O)O)O)O)C4=CC=C(C=C4)F",
     6.8),
]


class CompoundLoader:
    """Loads known HMGCR inhibitors from ChEMBL or a fallback statin list."""

    def __init__(self, target_chembl_id="CHEMBL1781"):
        """Set ChEMBL target ID (default: HMGCR human)."""
        self.target_chembl_id = target_chembl_id
        self.library = []

    def load_reference_library(self):
        """Try ChEMBL first; fall back to built-in statins on failure."""
        try:
            self.library = self._fetch_from_chembl()
            if not self.library:
                raise ValueError("no results")
            print(f"Loaded {len(self.library)} compounds from ChEMBL")

        except Exception as e:
            print(f"ChEMBL failed ({e}), using 7 fallback statins")
            self.library = self._load_fallback_statins()

        return self.library

    def _fetch_from_chembl(self):
        """Query ChEMBL for IC50 binding assays against the target protein."""
        from chembl_webresource_client.new_client import new_client

        results = new_client.activity.filter(
            target_chembl_id=self.target_chembl_id,
            standard_type="IC50",
        ).only(["molecule_chembl_id", "canonical_smiles", "standard_value"])

        compounds = []
        for rec in results:
            if len(compounds) >= 200:
                break

            smiles = rec.get("canonical_smiles")
            val = rec.get("standard_value")
            name = rec.get("molecule_chembl_id", "unknown")

            if not smiles or not val:
                continue
            try:
                ic50 = float(val)
            except (ValueError, TypeError):
                continue

            fingerprint = self._make_fingerprint(smiles)
            if fingerprint is None:
                continue

            compounds.append((name, smiles, ic50, fingerprint))

        return compounds

    def _load_fallback_statins(self):
        """Return all 7 built-in statins with fingerprints."""
        compounds = []
        for name, smiles, ic50 in FALLBACK_STATINS:
            fingerprint = self._make_fingerprint(smiles)
            if fingerprint:
                compounds.append((name, smiles, ic50, fingerprint))
        print(f"Loaded {len(compounds)} fallback statins")
        return compounds

    fingerprint_generator = GetMorganGenerator(radius=2, fpSize=2048)

    def _make_fingerprint(self, smiles):
        """Generate ECFP4 Morgan fingerprint (2048 bits) from a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return self.fingerprint_generator.GetFingerprint(mol)


if __name__ == "__main__":
    loader = CompoundLoader()
    lib = loader.load_reference_library()
    print(f"\nLoaded {len(lib)} compounds total\n")
    for name, smiles, ic50, fingerprint in lib:
        print(f"  {name}: IC50={ic50} nM, bits={fingerprint.GetNumOnBits()}")

