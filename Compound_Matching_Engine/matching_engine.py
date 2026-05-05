from rdkit import Chem, DataStructs

try:
    from .compound_loader import CompoundLoader
    from .fingerprint_encoder import FingerprintEncoder
    from .Similarity_scorer import SimilarityScorer
    from .Drug_likeness_filter import DrugLikenessFilter
    from .Scaffold_extractor import ScaffoldExtractor
    from .Match_result import MatchResult
    from .risk_classifier import RiskClassifier
except ImportError:
    from compound_loader import CompoundLoader
    from fingerprint_encoder import FingerprintEncoder
    from Similarity_scorer import SimilarityScorer
    from Drug_likeness_filter import DrugLikenessFilter
    from Scaffold_extractor import ScaffoldExtractor
    from Match_result import MatchResult
    from risk_classifier import RiskClassifier


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
        self.fallback_statins = self.loader._load_fallback_statins()

        # Add named statins so downstream PubMed/SIDER steps can use drug names
        self.library = self.with_named_statins(self.loader.load_reference_library())

        # Convert generated molecules into fingerprints, then rank nearest neighbors
        self.encoder = FingerprintEncoder()
        self.similarity_scorer = SimilarityScorer(self.library)
        self.drug_likeness_filter = DrugLikenessFilter()
        self.scaffold_extractor = ScaffoldExtractor()

        # A10b ANN classifier is optional: MatchingEngine should still run
        # even if the saved model artifact is missing or incompatible.
        try:
            self.risk_classifier = RiskClassifier.load()
        except Exception as e:
            print(f"[A10c] RiskClassifier unavailable: {e}")
            self.risk_classifier = None

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
        # Step 1: literature_name only converts CHEMBL IDs when the nearest-neighbor
        # SMILES exactly matches a known fallback statin.
        # Step 2: sider_lookup_name is SIDER/NLP-only. If the name is still a CHEMBL ID,
        # use the nearest marketed fallback statin as a readable safety/literature seed.
        # The true A5 nn_name remains similarity.nn_name in MatchResult.
        sider_lookup_name = self.resolve_sider_lookup_name(literature_name, query_fp)
        rule_tier = self.assign_rule_tier(filter_result, similarity)
        similarity_risk_score = self.compute_similarity_risk_score(similarity)

        # A10b: optional ANN prediction stored as supporting evidence.
        ml_tier = ""
        ml_proba = {}
        if self.risk_classifier is not None:
            try:
                ml_prediction = self.risk_classifier.predict(smiles)
                ml_tier = ml_prediction.tier
                ml_proba = ml_prediction.proba
            except Exception as e:
                print(f"[A10c] RiskClassifier prediction failed: {e}")

        # A9: inherit adverse-effect seeds from the SIDER/NLP lookup name.
        # This may be an exact statin name or a marketed-statin fallback when the
        # true nearest neighbor is an unresolved CHEMBL ID.
        sider_records = self.loader.load_sider_risks(sider_lookup_name, similarity.nn_score)

        # A10c: combine A10a rule tier with A10b ANN evidence.
        final_tier, confidence_flag = self.combine_rule_and_ml(rule_tier, ml_tier, ml_proba)

        # Pack the minimum Part 1 outputs needed by NLPAgent
        return MatchResult(
            query_smiles=smiles,
            nn_name=similarity.nn_name,
            nn_smiles=similarity.nn_smiles,
            nn_ic50=similarity.nn_ic50,
            tanimoto=similarity.nn_score,
            similarity_risk_score=similarity_risk_score,
            shared_atoms=scaffold_result.shared_atoms if scaffold_result else 0,
            novel_atoms=scaffold_result.novel_atoms if scaffold_result else 0,
            shared_pct=scaffold_result.shared_pct if scaffold_result else 0.0,
            scaffold_similarity=scaffold_result.scaffold_similarity if scaffold_result else 0.0,
            query_scaffold=scaffold_result.query_scaffold if scaffold_result else "",
            nn_scaffold=scaffold_result.nn_scaffold if scaffold_result else "",
            filter_result=filter_result,
            sider_risks=[record["effect"] for record in sider_records],
            sider_risk_records=sider_records,
            search_terms=self.build_search_terms(sider_lookup_name),
            risk_tier=final_tier,
            ml_tier=ml_tier,
            ml_proba=ml_proba,
            confidence_flag=confidence_flag,
            target=self.target,
        )

    def with_named_statins(self, library: list[tuple]) -> list[tuple]:
        """Add named statins so NLP/SIDER handoff can use drug names, not only CHEMBL IDs."""
        combined = list(library)
        seen_smiles = {row[1] for row in combined}

        # ChEMBL rows often have CHEMBL IDs; fallback statins provide readable drug names
        for row in self.fallback_statins:
            # Store a SMILES lookup so CHEMBL ID can be recognized as drug name.
            self.named_statin_by_smiles[self.canonical_smiles(row[1])] = row[0]
            if row[1] not in seen_smiles:
                combined.append(row)
                seen_smiles.add(row[1])
        return combined

    def resolve_literature_name(self, nn_name: str, nn_smiles: str) -> str:
        """Return a readable drug name when a ChEMBL nearest neighbor matches a named statin."""
        return self.named_statin_by_smiles.get(self.canonical_smiles(nn_smiles), nn_name.strip())

    def resolve_sider_lookup_name(self, literature_name: str, query_fp) -> str:
        """
        Return a SIDER/NLP lookup seed without changing the true A5 nearest neighbor.

        If the input name is already readable, use it.
        If it is still a CHEMBL ID, use the closest fallback marketed statin by
        Tanimoto to the query fingerprint, but only when similarity >= 0.40.
        """
        name = literature_name.strip()
        if not name.lower().startswith("chembl"):
            return name

        best_name = name
        best_score = -1.0
        for fallback_name, _smiles, _ic50, fallback_fp in self.fallback_statins:
            if fallback_fp is None:
                continue
            score = DataStructs.TanimotoSimilarity(query_fp, fallback_fp)
            if score > best_score:
                best_score = score
                best_name = fallback_name

        if best_score >= 0.40:
            return best_name
        return name

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

    def compute_similarity_risk_score(self, similarity) -> float:
        """Convert A5 Tanimoto similarity into a 0-1 novelty/uncertainty risk signal."""
        risk = 1.0 - float(similarity.nn_score)
        return round(max(0.0, min(1.0, risk)), 4)

    def combine_rule_and_ml(self, rule_tier: str, ml_tier: str, ml_proba: dict) -> tuple[str, str]:
        """A10c: combine rule and ANN verdicts with conservative escalation."""
        if not ml_tier:
            # No ANN prediction available -> fall back to A10a rule alone.
            final_tier = rule_tier
            flag = "RULE_ONLY"
            return final_tier, flag

        rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        ml_confidence = float(ml_proba.get(ml_tier, 0.0)) if ml_proba else 0.0

        if rule_tier == ml_tier:
            # Rule and ANN agree; flag splits on ANN self-confidence.
            final_tier = rule_tier
            if ml_confidence >= 0.70:
                flag = "AGREE_HIGH_CONF"
            else:
                flag = "AGREE_LOW_CONF"
            return final_tier, flag

        if rank.get(ml_tier, 0) > rank.get(rule_tier, 0):
            # ANN says it is riskier than the rule said.
            if ml_confidence >= 0.70:
                # Confident ANN escalation: trust ANN, raise final tier.
                final_tier = ml_tier
                flag = "REVIEW_ML_ESCALATED"
            else:
                # Low-confidence escalation: keep rule tier, flag for review.
                final_tier = rule_tier
                flag = "REVIEW_ML_HIGHER_LOW_CONF"
            return final_tier, flag

        # ANN predicts a less-risky tier than the rule. Conservatively keep
        # the rule's verdict and flag the disagreement for review.
        final_tier = rule_tier
        flag = "REVIEW_RULE_MORE_CONSERVATIVE"
        return final_tier, flag

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
