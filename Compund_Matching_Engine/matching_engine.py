from rdkit import Chem

try:
    from .compound_loader import CompoundLoader
    from .fingerprint_encoder import FingerprintEncoder
    from .Similarity_scorer import SimilarityScorer
    from .Drug_likeness_filter import DrugLikenessFilter
    from .Scaffold_extractor import ScaffoldExtractor
    from .Match_result import MatchResult
except ImportError:
    from compound_loader import CompoundLoader
    from fingerprint_encoder import FingerprintEncoder
    from Similarity_scorer import SimilarityScorer
    from Drug_likeness_filter import DrugLikenessFilter
    from Scaffold_extractor import ScaffoldExtractor
    from Match_result import MatchResult


def parse_smiles(smiles: str):
    # Step 1: Ensure that entry is not empty
    if not smiles or not smiles.strip():
        return None, "ERROR: empty input"

    # Step 2: pass to RDKit's parser
    mol = Chem.MolFromSmiles(smiles, sanitize=False)

    # Step 3: if parser returns nothing, stop
    if mol is None:
        return None, "ERROR: invalid SMILES string"

    # Step 4 and then 5: run sanitisation
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        return None, f"ERROR: sanitisation failed — {str(e)}"

    # Step 6: return molecule and OK status
    return mol, "OK"


# Minimal orchestrator for Part 1 (wraps into MatchResult)
# Runs the already-implemented Part 1 components in sequence and fills the values
# needed by downstream stages
# Goal: make the NLPAgent no longer require manual inputs for nn_name/search_terms/sider_risks
# will develop further to meet the requirements for other parts as well later.
class MatchingEngine:
    """Run Part 1 compound matching and return a populated MatchResult."""
    def __init__(self, target: str = "HMGCR", target_chembl_id: str = "CHEMBL402") -> None:
        self.target = target

        # Load corrected HMGCR reference compounds from ChEMBL
        self.loader = CompoundLoader(target_chembl_id=target_chembl_id)
        self.named_statin_by_smiles = {}

        # Add named statins so downstream PubMed/SIDER steps can use drug names
        self.library = self.with_named_statins(self.loader.load_reference_library())

        # Convert generated molecules into fingerprints, then rank nearest neighbors
        self.encoder = FingerprintEncoder()
        self.similarity_scorer = SimilarityScorer(self.library)
        self.drug_likeness_filter = DrugLikenessFilter()
        self.scaffold_extractor = ScaffoldExtractor()

    def run(self, smiles: str) -> MatchResult:
        """Convert one generated SMILES into the Part 1 output container."""
        mol, status = parse_smiles(smiles)
        if mol is None:
            return MatchResult(query_smiles=smiles, target=self.target, search_terms=self.target)

        # A3 + A5: encode the generated molecule and find the closest reference drug
        filter_result = self.drug_likeness_filter.filter(mol)
        query_fp = self.encoder.encode(mol)
        similarity = self.similarity_scorer.score(query_fp)
        nn_mol = Chem.MolFromSmiles(similarity.nn_smiles)
        scaffold_result = self.scaffold_extractor.extract(mol, nn_mol) if nn_mol else None
        literature_name = self.resolve_literature_name(similarity.nn_name, similarity.nn_smiles)
        rule_tier = self.assign_rule_tier(filter_result, similarity)

        # A9: use the readable literature name for SIDER when the nearest ChEMBL ID
        # is actually a known statin such as atorvastatin.
        sider_records = self.loader.load_sider_risks(literature_name, similarity.nn_score)

        # Pack the minimum Part 1 outputs needed by NLPAgent
        return MatchResult(
            query_smiles=smiles,
            nn_name=similarity.nn_name,
            nn_smiles=similarity.nn_smiles,
            nn_ic50=similarity.nn_ic50,
            tanimoto=similarity.nn_score,
            shared_atoms=scaffold_result.shared_atoms if scaffold_result else 0,
            novel_atoms=scaffold_result.novel_atoms if scaffold_result else 0,
            shared_pct=scaffold_result.shared_pct if scaffold_result else 0.0,
            scaffold_similarity=scaffold_result.scaffold_similarity if scaffold_result else 0.0,
            query_scaffold=scaffold_result.query_scaffold if scaffold_result else "",
            nn_scaffold=scaffold_result.nn_scaffold if scaffold_result else "",
            filter_result=filter_result,
            sider_risks=[record["effect"] for record in sider_records],
            search_terms=self.build_search_terms(literature_name),
            risk_tier=rule_tier, # this line will be updated when other rules come into play
            target=self.target,
        )

    def with_named_statins(self, library: list[tuple]) -> list[tuple]:
        """Add named statins so NLP/SIDER handoff can use drug names, not only CHEMBL IDs."""
        combined = list(library)
        seen_smiles = {row[1] for row in combined}

        # ChEMBL rows often have CHEMBL IDs; fallback statins provide readable drug names
        for row in self.loader._load_fallback_statins():
            # Store a SMILES lookup so CHEMBL ID can be recognized as drug name.
            self.named_statin_by_smiles[self.canonical_smiles(row[1])] = row[0]
            if row[1] not in seen_smiles:
                combined.append(row)
                seen_smiles.add(row[1])
        return combined

    def resolve_literature_name(self, nn_name: str, nn_smiles: str) -> str:
        """Return a readable drug name when a ChEMBL nearest neighbor matches a named statin."""
        return self.named_statin_by_smiles.get(self.canonical_smiles(nn_smiles), nn_name.strip())

    def build_search_terms(self, literature_name: str) -> str:
        """Build PubMed-friendly search terms for the NLPAgent."""
        terms = [literature_name.strip(), self.target]
        return " ".join(term for term in terms if term)

    def assign_rule_tier(self, filter_result, similarity) -> str:
        """Assign A10a rule-based risk tier from filter and similarity outputs."""
        if filter_result.pains_flag or filter_result.brenk_flag:
            return "HIGH"

        if filter_result.violations >= 2:
            return "HIGH"

        if filter_result.qed < 0.34:
            return "HIGH"

        if similarity.nn_score >= 0.80 and similarity.nn_ic50 is not None and similarity.nn_ic50 <= 100:
            return "LOW"

        return "MEDIUM"

    @staticmethod
    def canonical_smiles(smiles: str) -> str:
        """Normalize SMILES so exact statin matches can be recognized."""
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol is not None else smiles.strip()

# Test with Atorvastatin
if __name__ == "__main__":
    smiles = "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O"
    mol, status = parse_smiles(smiles)
    print(status)
    # Test 2 AI-generated molecule (from MolGPT paper, HMGCR-like)
    ai_generated = "O=C(O)CC(O)CC(O)CCn1c(C(C)C)c(-c2ccc(F)cc2)c(C(=O)Nc2ccccc2)c1-c1ccccc1"
    mol2, status2 = parse_smiles(ai_generated)
    print(f"AI-generated molecule: {status2}")

    # Test 3 deliberately broken SMILES to confirm error handling works
    broken = "CC(C)c1ccc(INVALID!!!)cc1"
    mol3, status3 = parse_smiles(broken)
    print(f"Broken SMILES: {status3}")
