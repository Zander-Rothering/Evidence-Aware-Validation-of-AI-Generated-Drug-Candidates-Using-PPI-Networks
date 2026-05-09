"""Drug_likeness_filter.py — PAINS/Brenk and Lipinski screening (Part 1, Steps A6/A7)."""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from .Filter_result import FilterResult


class DrugLikenessFilter:
    """
    Applies structural alert filters and Lipinski Rule-of-Five checks.

    Step A6 — PAINS and Brenk catalog screening.
      Any match forces is_clean=False and overrides all other signals → HIGH risk.

    Step A7 — Lipinski / QED descriptors.
      violations >= 2 → HIGH.
      QED < 0.34 on a MEDIUM result → escalated to HIGH by MatchingEngine.

    Inputs : mol object (RDKit Mol from SmilesParser A2)
    Outputs: FilterResult  [also contributes ML features to RF/SVM]
    """

    def __init__(self) -> None:
        # PAINS catalog
        pains_params = FilterCatalogParams()
        pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        self._pains = FilterCatalog(pains_params)

        # Brenk catalog
        brenk_params = FilterCatalogParams()
        brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        self._brenk = FilterCatalog(brenk_params)

    # ------------------------------------------------------------------
    def filter(self, mol: Chem.Mol) -> FilterResult:
        """
        Parameters
        ----------
        mol : valid RDKit Mol from SmilesParser

        Returns
        -------
        FilterResult with all fields populated
        """
        result = FilterResult()

        # --- PAINS (A6) ---
        pains_match = self._pains.GetFirstMatch(mol)
        if pains_match is not None:
            result.pains_flag = True
            result.matches.append(pains_match.GetDescription())

        # --- Brenk (A6) ---
        brenk_match = self._brenk.GetFirstMatch(mol)
        if brenk_match is not None:
            result.brenk_flag = True
            result.matches.append(brenk_match.GetDescription())

        result.is_clean = not (result.pains_flag or result.brenk_flag)

        # --- Lipinski / descriptors (A7) ---
        result.mw        = round(Descriptors.MolWt(mol), 2)
        result.logp      = round(Descriptors.MolLogP(mol), 2)
        result.hbd       = rdMolDescriptors.CalcNumHBD(mol)
        result.hba       = rdMolDescriptors.CalcNumHBA(mol)
        result.qed       = round(QED.qed(mol), 3)
        result.tpsa      = round(rdMolDescriptors.CalcTPSA(mol), 2)
        result.rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

        result.violations = sum([
            result.mw   > 500,
            result.logp > 5,
            result.hbd  > 5,
            result.hba  > 10,
        ])
        result.lipinski_pass = result.violations == 0

        return result

    # ------------------------------------------------------------------
    def is_pains_clean(self, mol: Chem.Mol) -> bool:
        return self._pains.GetFirstMatch(mol) is None

    def is_brenk_clean(self, mol: Chem.Mol) -> bool:
        return self._brenk.GetFirstMatch(mol) is None

