"""Scaffold_extractor.py — Murcko scaffold extraction and comparison (Part 1, Step A8)."""
from __future__ import annotations
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator, rdFMCS
from rdkit import DataStructs


@dataclass
class ScaffoldResult:
    query_scaffold: str = ""
    nn_scaffold: str = ""
    shared_atoms: int = 0
    novel_atoms: int = 0
    shared_pct: float = 0.0
    scaffold_similarity: float = 0.0   # Tanimoto of scaffold Morgan fps


class ScaffoldExtractor:
    """
    Extracts Murcko scaffolds and computes scaffold-level Tanimoto similarity.

    Inputs : mol (RDKit Mol from A2), nn_mol (nearest-neighbour Mol from A5)
    Outputs: ScaffoldResult with shared_atoms, scaffold_similarity, scaffold SMILES
    """

    def __init__(self, radius: int = 2, fp_size: int = 2048) -> None:
        self._gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=fp_size
        )

    # ------------------------------------------------------------------
    def extract_murcko(self, mol: Chem.Mol) -> Chem.Mol | None:
        """Return the Murcko scaffold mol for `mol`, or None on failure."""
        try:
            return MurckoScaffold.GetScaffoldForMol(mol)
        except Exception:
            return None

    def murcko_smiles(self, mol: Chem.Mol) -> str:
        """Return canonical SMILES of the Murcko scaffold for `mol`."""
        sca = self.extract_murcko(mol)
        if sca is None:
            return ""
        return Chem.MolToSmiles(sca, isomericSmiles=False)

    def compare(self, candidate_mol: Chem.Mol, neighbour_mol: Chem.Mol) -> dict:
        """
        MCS comparison of the two Murcko scaffolds.

        Returns
        -------
        dict with:
            shared_atoms : int   — atoms in the maximum common substructure
            novel_atoms  : int   — candidate scaffold atoms not in the MCS
            shared_pct   : float — shared_atoms / candidate scaffold heavy atoms
        """
        q = self.extract_murcko(candidate_mol)
        n = self.extract_murcko(neighbour_mol)

        if q is None or n is None or q.GetNumHeavyAtoms() == 0 or n.GetNumHeavyAtoms() == 0:
            return {"shared_atoms": 0, "novel_atoms": 0, "shared_pct": 0.0}

        try:
            mcs = rdFMCS.FindMCS([q, n], timeout=5)
            shared = mcs.numAtoms if mcs and not mcs.canceled else 0
        except Exception:
            shared = 0

        q_size = q.GetNumHeavyAtoms()
        return {
            "shared_atoms": shared,
            "novel_atoms":  q_size - shared,
            "shared_pct":   shared / q_size if q_size else 0.0,
        }

    def _fingerprint(self, mol: Chem.Mol):
        return self._gen.GetFingerprint(mol)

    # ------------------------------------------------------------------
    def extract(self, query_mol: Chem.Mol, nn_mol: Chem.Mol) -> ScaffoldResult:
        """
        Parameters
        ----------
        query_mol : RDKit Mol of the AI-generated candidate (from SmilesParser)
        nn_mol    : RDKit Mol of the nearest-neighbour ChEMBL compound (from SimilarityScorer)

        Returns
        -------
        ScaffoldResult
        """
        result = ScaffoldResult()

        q_sca_mol = self.extract_murcko(query_mol)
        n_sca_mol = self.extract_murcko(nn_mol)

        result.query_scaffold = (
            Chem.MolToSmiles(q_sca_mol, isomericSmiles=False) if q_sca_mol else ""
        )
        result.nn_scaffold = (
            Chem.MolToSmiles(n_sca_mol, isomericSmiles=False) if n_sca_mol else ""
        )

        if q_sca_mol and n_sca_mol and q_sca_mol.GetNumHeavyAtoms() > 0:
            cmp = self.compare(query_mol, nn_mol)
            result.shared_atoms = cmp["shared_atoms"]
            result.novel_atoms  = cmp["novel_atoms"]
            result.shared_pct   = cmp["shared_pct"]

            q_fp = self._fingerprint(q_sca_mol)
            n_fp = self._fingerprint(n_sca_mol)
            result.scaffold_similarity = DataStructs.TanimotoSimilarity(q_fp, n_fp)

        return result

    # ------------------------------------------------------------------
    def get_scaffold_smiles(self, smiles: str) -> str:
        """Convenience: return scaffold SMILES for a raw SMILES string."""
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return ""
        return self.murcko_smiles(mol)

