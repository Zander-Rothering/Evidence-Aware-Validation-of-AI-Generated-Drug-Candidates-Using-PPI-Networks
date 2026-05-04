"""structure_visualiser.py — 2-D structure rendering for query and nearest-neighbour compounds."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D


class StructureVisualiser:
    """
    Renders 2-D structure images for pipeline inspection and reporting.

    Typical uses
    ------------
    - Side-by-side query vs nearest-neighbour display
    - Scaffold overlay highlight
    - Batch grid image of all Part 1 candidates
    """

    def __init__(self, img_size: Tuple[int, int] = (400, 300)) -> None:
        self.img_size = img_size

    # ------------------------------------------------------------------
    def mol_to_svg(self, mol: Chem.Mol, highlight_atoms: Optional[List[int]] = None) -> str:
        """Return an SVG string for a single molecule."""
        drawer = rdMolDraw2D.MolDraw2DSVG(self.img_size[0], self.img_size[1])
        if highlight_atoms:
            drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
        else:
            drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    def smiles_to_svg(self, smiles: str, **kwargs) -> str:
        """Convenience wrapper: convert SMILES directly to SVG."""
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return ""
        return self.mol_to_svg(mol, **kwargs)

    # ------------------------------------------------------------------
    def save_mol(self, mol: Chem.Mol, path: str | Path) -> Path:
        """Save a PNG of a single molecule."""
        path = Path(path)
        img = Draw.MolToImage(mol, size=self.img_size)
        img.save(str(path))
        return path

    def save_smiles(self, smiles: str, path: str | Path) -> Optional[Path]:
        """Save a PNG from a SMILES string; returns None if SMILES is invalid."""
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        return self.save_mol(mol, path)

    # ------------------------------------------------------------------
    def comparison_grid(
        self,
        query_mol: Chem.Mol,
        nn_mol: Chem.Mol,
        path: str | Path,
        legends: Optional[List[str]] = None,
    ) -> Path:
        """
        Save a 2-panel side-by-side grid (query | nearest-neighbour).

        Parameters
        ----------
        query_mol : AI-generated compound mol
        nn_mol    : nearest-neighbour ChEMBL compound mol
        path      : output PNG path
        legends   : optional [query_label, nn_label]
        """
        path = Path(path)
        mols = [query_mol, nn_mol]
        legs = legends or ["Query", "Nearest Neighbour"]
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=2,
            subImgSize=self.img_size,
            legends=legs,
        )
        img.save(str(path))
        return path

    def batch_grid(
        self,
        smiles_list: List[str],
        path: str | Path,
        mols_per_row: int = 4,
        legends: Optional[List[str]] = None,
    ) -> Path:
        """
        Save a grid image for a batch of SMILES (e.g. all Part 1 candidates).

        Invalid SMILES are skipped.
        """
        path = Path(path)
        mols, valid_legends = [], []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is not None:
                mols.append(mol)
                valid_legends.append(legends[i] if legends else smi[:20])
        if not mols:
            return path
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=mols_per_row,
            subImgSize=self.img_size,
            legends=valid_legends,
        )
        img.save(str(path))
        return path
