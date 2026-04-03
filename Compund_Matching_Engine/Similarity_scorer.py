"""Activity 5 — Compute Tanimoto Similarity Scores.
   Developed with AI assistance (Claude, Anthropic) for syntax support.
"""

from rdkit.Chem import DataStructs
from rdkit.DataStructs import ExplicitBitVect


class SimilarityResult:
    """Container for the output of a query-vs-library similarity search."""

    def __init__(
        self,
        nn_name:   str,                                  # nearest neighbour name (ChEMBL ID or drug name)
        nn_smiles: str,                                  # nearest neighbour SMILES string
        nn_ic50:   float,                                # nearest neighbour IC50 in nM
        nn_score:  float,                                # Tanimoto similarity score (0.0 – 1.0)
        band:      str,                                  # threshold band: "HIGH", "MEDIUM", or "LOW"
        ranked:    list[tuple[str, str, float, float]],  # all (name, smiles, ic50, score) sorted descending
    ):
        self.nn_name   = nn_name
        self.nn_smiles = nn_smiles
        self.nn_ic50   = nn_ic50
        self.nn_score  = nn_score
        self.band      = band
        self.ranked    = ranked

    def __repr__(self) -> str:
        return (
            f"SimilarityResult("
            f"nn={self.nn_name}, "
            f"score={self.nn_score:.4f}, "
            f"band={self.band}, "
            f"ic50={self.nn_ic50} nM, "
            f"library_size={len(self.ranked)})"
        )


class SimilarityScorer:
    """Rank a query fingerprint against a reference library by Tanimoto similarity."""

    def __init__(self, library: list[tuple[str, str, float, ExplicitBitVect]]) -> None:
        """Accept the library returned by CompoundLoader.load_reference_library()."""
        self.library = library

    def score(self, query_fp: ExplicitBitVect) -> SimilarityResult:
        """Compare query against every reference compound and return ranked results.

        Returns a SimilarityResult whose .nn_name and .nn_score can feed
        directly into CompoundLoader.load_sider_risks().
        """
        if not self.library:
            raise ValueError("reference library is empty")

        scores: list[tuple[str, str, float, float]] = []
        for name, smiles, ic50, ref_fp in self.library:
            # calculate tanimoto similarity using RDkit's DataStructs module
            tanimoto = DataStructs.TanimotoSimilarity(query_fp, ref_fp)
            scores.append((name, smiles, ic50, tanimoto))
        # sort by tanimoto score descending
        scores.sort(key=lambda x: x[3], reverse=True)

        nn_name, nn_smiles, nn_ic50, nn_score = scores[0]
        band = self._interpret_band(nn_score) # map tanimoto score to a threshold band label

        return SimilarityResult(
            nn_name=nn_name,
            nn_smiles=nn_smiles,
            nn_ic50=nn_ic50,
            nn_score=nn_score,
            band=band,
            ranked=scores,
        )

    @staticmethod
    def _interpret_band(score: float) -> str:
        """Map a Tanimoto score to a threshold band label."""
        if score >= 0.80:
            return "HIGH"
        if score >= 0.40:
            return "MEDIUM"
        return "LOW"


if __name__ == "__main__":
    from rdkit import Chem
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    from compound_loader import CompoundLoader

    loader = CompoundLoader()
    library = loader.load_reference_library()

    fp_gen = GetMorganGenerator(radius=2, fpSize=2048)

    # Use atorvastatin as the test query
    query_smiles = (
        "CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)"
        "C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4"
    )
    query_mol = Chem.MolFromSmiles(query_smiles)
    query_fp = fp_gen.GetFingerprint(query_mol)

    scorer = SimilarityScorer(library)
    result = scorer.score(query_fp)

    print(f"\n{result}\n")
    print(f"Nearest neighbour: {result.nn_name}")
    print(f"  SMILES : {result.nn_smiles}")
    print(f"  IC50   : {result.nn_ic50} nM")
    print(f"  Score  : {result.nn_score:.4f}")
    print(f"  Band   : {result.band}\n")

    print("Top 5 ranked compounds:")
    for rank, (name, smiles, ic50, score) in enumerate(result.ranked[:5], 1):
        band = SimilarityScorer._interpret_band(score)
        print(f"  {rank}. {name}: score={score:.4f} ({band}), IC50={ic50} nM")
    if len(result.ranked) > 5:
        print(f"  ... and {len(result.ranked) - 5} more")
