"""Activity 4 — Build the Reference Library from ChEMBL.
   Activity 9 — Load SIDER inherited adverse-effect profiles.
   Developed with AI assistance for syntax support.
"""

import os
import pickle
import requests
import pandas as pd
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

    def __init__(self, target_chembl_id="CHEMBL402"):
        """Set ChEMBL target ID (default: HMGCR human)."""
        self.target_chembl_id = target_chembl_id
        self.library = []
        self.fingerprint_generator = GetMorganGenerator(radius=2, fpSize=2048)

    # Cache file lives next to compound_loader.py:
    # Compound_Matching_Engine/chembl_cache.pkl
    _CACHE_FILE = os.path.join(os.path.dirname(__file__), "chembl_cache.pkl")

    def load_reference_library(self):
        """Load the reference compound library with pickle caching.

        ChEMBL is unreliable (frequent 500 errors), and during deep-learning
        training loops the library is loaded repeatedly. Calling the API on
        every iteration is unnecessary because the reference compounds never
        change between runs.

        Caching strategy:
          1. If chembl_cache.pkl exists  -> load from cache, skip API entirely.
          2. If no cache                 -> fetch from ChEMBL, save to cache.
          3. If ChEMBL fails             -> use fallback statins, do NOT cache
             (so the next run retries ChEMBL when the server is back).

        Cache location: Compound_Matching_Engine/chembl_cache.pkl
        (same directory as this file, excluded from git).
        """
        # 1. Try loading from cache
        if os.path.isfile(self._CACHE_FILE):
            try:
                with open(self._CACHE_FILE, "rb") as f:
                    self.library = pickle.load(f)
                print(f"Loaded {len(self.library)} compounds from cache "
                      f"({self._CACHE_FILE})")
                return self.library
            except Exception as e:
                print(f"Cache load failed ({e}), fetching from ChEMBL")

        # 2. Fetch from ChEMBL
        try:
            self.library = self._fetch_from_chembl()
            if not self.library:
                raise ValueError("no results")
            print(f"Loaded {len(self.library)} compounds from ChEMBL")

            # Save to cache for future runs
            with open(self._CACHE_FILE, "wb") as f:
                pickle.dump(self.library, f)
            print(f"Cached to {self._CACHE_FILE}")

        except Exception as e:
            # 3. Fallback — do NOT cache so we retry ChEMBL next time
            print(f"ChEMBL failed ({e}), using 7 fallback statins")
            self.library = self._load_fallback_statins()

        return self.library

    def _fetch_from_chembl(self):
        """Query ChEMBL for IC50 binding assays against the target protein."""
        from chembl_webresource_client.new_client import new_client

        results = new_client.activity.filter(
            target_chembl_id=self.target_chembl_id,
            standard_type="IC50",
            assay_type="B",
        ).only(["molecule_chembl_id", "canonical_smiles", "standard_value"])

        compounds = []
        seen_smiles: set[str] = set()
        for rec in results:
            if len(compounds) >= 200:  # increase this number if more compounds are needed (delete chembl_cache.pkl after changing)
                break

            smiles = rec.get("canonical_smiles")
            val = rec.get("standard_value")
            name = rec.get("molecule_chembl_id", "unknown")

            if not smiles or not val:
                continue
            if smiles in seen_smiles:
                continue
            seen_smiles.add(smiles)
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

    def _make_fingerprint(self, smiles):
        """Generate ECFP4 Morgan fingerprint (2048 bits) from a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return self.fingerprint_generator.GetFingerprint(mol)

    # ----------
    # Activity 9 — SIDER inherited adverse-effect lookup 

    # Tier 4 fallback: known statin-class adverse effects.
    # Used only when drug_names.tsv, PubChem CID paths, and SIDER file join fail.
    _SIDER_FALLBACK = {
        "atorvastatin":  ["myopathy", "hepatotoxicity", "rhabdomyolysis",
                          "diabetes mellitus", "peripheral neuropathy"],
        "rosuvastatin":  ["myopathy", "hepatotoxicity", "rhabdomyolysis",
                          "proteinuria", "haematuria"],
        "simvastatin":   ["myopathy", "hepatotoxicity", "rhabdomyolysis",
                          "diabetes mellitus", "memory impairment"],
        "pravastatin":   ["myopathy", "hepatotoxicity", "rhabdomyolysis",
                          "dizziness"],
        "fluvastatin":   ["myopathy", "hepatotoxicity", "insomnia",
                          "dyspepsia"],
        "lovastatin":    ["myopathy", "hepatotoxicity", "rhabdomyolysis",
                          "constipation", "abdominal pain"],
        "pitavastatin":  ["myopathy", "hepatotoxicity", "rhabdomyolysis",
                          "arthralgia"],
    }

    # Cached SIDER dataframe — loaded once per process, shared across
    # all CompoundLoader instances to avoid re-reading the gzipped file.
    _sider_df_cache = None

    # SIDER side-effect file downloaded from http://sideeffects.embl.de/download/
    # and placed at Compound_Matching_Engine/sider_data/meddra_all_se.tsv.gz
    _SIDER_FILE = os.path.join(os.path.dirname(__file__), "sider_data", "meddra_all_se.tsv.gz")

    # SIDER/STITCH compound id <-> drug name (from sideeffects.embl.de)
    _SIDER_NAMES_FILE = os.path.join(os.path.dirname(__file__), "sider_data", "drug_names.tsv")

    _sider_name_to_ids_cache = None

    def _name_to_cid(self, drug_name):
        """Tier 2 helper: resolve a drug name to a PubChem CID via REST API.

        Returns the integer CID, or None on any failure.
        """
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
            f"compound/name/{requests.utils.quote(drug_name)}/cids/JSON"
        )
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            cids = resp.json()["IdentifierList"]["CID"]
            return cids[0] if cids else None
        except Exception:
            return None

    def _load_sider_df(self):
        """Lazily load and cache the SIDER meddra side-effect TSV.

        Returns the filtered dataframe (meddra_type == 'PT' only),
        or None if the file does not exist.
        """
        if CompoundLoader._sider_df_cache is not None:
            return CompoundLoader._sider_df_cache

        if not os.path.isfile(self._SIDER_FILE):
            return None

        try:
            df = pd.read_csv(
                self._SIDER_FILE,
                sep="\t",
                header=None,
                names=[
                    "stitch_flat", "stitch_stereo", "umls_cui",
                    "meddra_type", "umls_cui_name", "side_effect_name",
                ],
            )
            df = df[df["meddra_type"] == "PT"]
            CompoundLoader._sider_df_cache = df
            return df
        except Exception:
            return None

    # Offline name->CID map for common statins so Tier 3 can work
    # even when the PubChem API is unreachable.
    _NAME_TO_CID_FALLBACK = {
        "atorvastatin": 60823,
        "rosuvastatin":  461310,
        "simvastatin":   54454,
        "pravastatin":   54687,
        "fluvastatin":   446155,
        "lovastatin":    53232,
        "pitavastatin":  5282452,
    }

    @staticmethod
    def _cid_to_stitch(cid):
        """Convert a PubChem CID to a STITCH flat identifier.

        STITCH flat format: 'CID1' + zero-padded CID to 8 digits.
        Example: 54454 -> CID100054454
        """
        return f"CID1{cid:08d}"

    def _load_sider_name_to_ids(self):
        """Load SIDER drug_names.tsv as a lowercase drug-name -> list[SIDER ID] map."""
        if CompoundLoader._sider_name_to_ids_cache is not None:
            return CompoundLoader._sider_name_to_ids_cache

        if not os.path.isfile(self._SIDER_NAMES_FILE):
            CompoundLoader._sider_name_to_ids_cache = {}
            return CompoundLoader._sider_name_to_ids_cache

        try:
            df = pd.read_csv(
                self._SIDER_NAMES_FILE,
                sep="\t",
                header=None,
                names=["sider_id", "drug_name"],
            )
            name_to_ids: dict[str, list[str]] = {}
            for _, row in df.iterrows():
                sid = str(row["sider_id"]).strip()
                drug = str(row["drug_name"]).strip().lower()
                if not sid or not drug:
                    continue
                name_to_ids.setdefault(drug, []).append(sid)
            CompoundLoader._sider_name_to_ids_cache = name_to_ids
            return name_to_ids
        except Exception:
            CompoundLoader._sider_name_to_ids_cache = {}
            return CompoundLoader._sider_name_to_ids_cache

    def _effects_for_sider_ids(self, sider_ids):
        """Return unique PT side effects for one or more SIDER/STITCH IDs."""
        if not sider_ids:
            return []
        sider_df = self._load_sider_df()
        if sider_df is None:
            return []
        id_set = set(sider_ids)
        hits = sider_df[
            sider_df["stitch_flat"].isin(id_set) | sider_df["stitch_stereo"].isin(id_set)
        ]
        if hits.empty:
            return []
        return hits["side_effect_name"].unique().tolist()

    def _query_sider(self, nn_name):
        """Look up known adverse effects for a drug name.

        Uses a 4-tier fallback strategy:
          Tier 1 — SIDER drug_names.tsv (name -> STITCH IDs) -> SIDER side-effect file
          Tier 2 — PubChem API (name -> CID) -> STITCH ID -> SIDER file
          Tier 3 — Offline CID map -> STITCH ID -> SIDER file (no API)
          Tier 4 — Curated statin hardcoded fallback

        All tiers return list[str] of side-effect names.
        """
        key = nn_name.strip().lower()

        # Tier 1: drug_names.tsv -> SIDER/STITCH IDs -> side-effect file
        name_to_ids = self._load_sider_name_to_ids()
        sider_ids = name_to_ids.get(key, [])
        effects = self._effects_for_sider_ids(sider_ids)
        if effects:
            print(
                f"[SIDER Tier 1] {nn_name} -> {len(effects)} effects from drug_names.tsv"
            )
            return effects

        # Tier 2: PubChem API -> CID -> SIDER file
        cid = self._name_to_cid(key)
        if cid is not None:
            stitch_id = self._cid_to_stitch(cid)
            effects = self._effects_for_sider_ids([stitch_id])
            if effects:
                print(f"[SIDER Tier 2] {nn_name} -> CID {cid} -> "
                      f"{len(effects)} effects from SIDER file")
                return effects

        # Tier 3: offline CID map -> SIDER file (no API)
        fallback_cid = self._NAME_TO_CID_FALLBACK.get(key)
        if fallback_cid is not None and fallback_cid != cid:
            stitch_id = self._cid_to_stitch(fallback_cid)
            effects = self._effects_for_sider_ids([stitch_id])
            if effects:
                print(f"[SIDER Tier 3] {nn_name} -> CID {fallback_cid} -> "
                      f"{len(effects)} effects from SIDER file (offline CID)")
                return effects

        # Tier 4: hardcoded fallback
        # Only return effects for drugs explicitly known.
        # For unknown drugs, return an empty list rather than
        # guessing statin-specific risks that may not apply.
        effects = list(self._SIDER_FALLBACK.get(key, []))
        print(f"[SIDER Tier 4] {nn_name} -> {len(effects)} effects "
              f"from hardcoded fallback")
        return effects

    def load_sider_risks(self, nn_name, tanimoto):
        """Inherit adverse effects from the nearest-neighbour drug.

        Parameters
        ----------
        nn_name : str
            Drug name of the nearest neighbour (from Activity 5).
        tanimoto : float
            Tanimoto similarity score between query and nearest neighbour.

        Returns
        -------
        list[dict]
            Each dict has keys "effect", "tag", and "tanimoto".
            Empty list when tanimoto is below the inheritance threshold.
        """
        if tanimoto < 0.40:
            return []

        if tanimoto >= 0.80:
            tag = "directly inherited"
        else:
            tag = "inherited by similarity"

        effects = self._query_sider(nn_name)

        return [
            {"effect": effect, "tag": tag, "tanimoto": round(tanimoto, 4)}
            for effect in effects
        ]


if __name__ == "__main__":
    loader = CompoundLoader()

    # Activity 4: load reference library
    lib = loader.load_reference_library()
    print(f"\nLoaded {len(lib)} compounds total\n")
    for name, smiles, ic50, fingerprint in lib[:5]:
        print(f"  {name}: IC50={ic50} nM, bits={fingerprint.GetNumOnBits()}")
    if len(lib) > 5:
        print(f"  ... and {len(lib) - 5} more\n")

    # Activity 9: SIDER risk inheritance demo
    print("-----------------------------------------")
    print("Activity 9 — SIDER inherited adverse-effect lookup")
    print("-----------------------------------------")

    test_cases = [
        ("simvastatin",   0.85),
        ("atorvastatin",  0.65),
        ("lovastatin",    0.30),
        ("some_unknown",  0.90),
    ]
    for drug, tani in test_cases:
        print(f"\n--- {drug}, tanimoto={tani} ---")
        risks = loader.load_sider_risks(drug, tani)
        if not risks:
            print("  No risks inherited (below threshold or unknown drug)")
        else:
            for r in risks[:5]:
                print(f"  [{r['tag']}] {r['effect']}")
            if len(risks) > 5:
                print(f"  ... and {len(risks) - 5} more")
