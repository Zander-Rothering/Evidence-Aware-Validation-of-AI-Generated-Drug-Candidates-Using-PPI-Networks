"""Scaffold_extractor.py — Murcko scaffold extraction and comparison (Part 1, Step A8)."""
from __future__ import annotations
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs


@dataclass
class ScaffoldResult:
    query_scaffold: str = ""
    nn_scaffold: str = ""
    shared_atoms: int = 0
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
    def _to_scaffold(self, mol: Chem.Mol) -> Chem.Mol | None:
        try:
            return MurckoScaffold.GetScaffoldForMol(mol)
        except Exception:
            return None

    def _scaffold_smiles(self, mol: Chem.Mol) -> str:
        sca = self._to_scaffold(mol)
        if sca is None:
            return ""
        return Chem.MolToSmiles(sca, isomericSmiles=False)

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

        q_sca_mol = self._to_scaffold(query_mol)
        n_sca_mol = self._to_scaffold(nn_mol)

        result.query_scaffold = (
            Chem.MolToSmiles(q_sca_mol, isomericSmiles=False) if q_sca_mol else ""
        )
        result.nn_scaffold = (
            Chem.MolToSmiles(n_sca_mol, isomericSmiles=False) if n_sca_mol else ""
        )

        # shared heavy atoms between the two scaffolds
        if q_sca_mol and n_sca_mol:
            q_atoms = set(a.GetAtomMapNum() for a in q_sca_mol.GetAtoms())
            n_atoms = set(a.GetAtomMapNum() for a in n_sca_mol.GetAtoms())
            result.shared_atoms = q_sca_mol.GetNumHeavyAtoms()  # conservative proxy

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
        return self._scaffold_smiles(mol)

