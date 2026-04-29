"""
B5 - NLP Agent orchestrator.

Wires B1 -> B2 -> B3 -> B4 together.
Takes a MatchResult from Part 1, returns a LiteratureResult.

Developed with AI assistance for syntax support.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from Compund_Matching_Engine import MatchingEngine, MatchResult

from pubmed_searcher import PubMedSearcher
from ner_extractor import NERExtractor
from signal_classifier import SignalClassifier
from literature_risk_scorer import LiteratureRiskScorer
from literature_result import LiteratureResult


class NLPAgent:
    def __init__(self):
        self.searcher = PubMedSearcher()
        self.ner = NERExtractor()
        self.classifier = SignalClassifier()
        self.scorer = LiteratureRiskScorer()

    def get_literature_search_inputs(self, match_result: MatchResult) -> tuple[str, list[str], str]:
        """ Part 1 -> Part 2A input bridge: 
        Get literature-search inputs from Part 1, with safe MVP fallbacks.
        """
        search_terms = getattr(match_result, "search_terms", None) or match_result.nn_name
        target = getattr(match_result, "target", None) or "HMGCR"
        sider_risks = getattr(match_result, "sider_risks", None)

        if isinstance(sider_risks, str):
            sider_risks = [sider_risks]
        elif sider_risks is None:
            sider_risks = []
        else:
            sider_risks = list(sider_risks)

        if not search_terms:
            raise ValueError("NLPAgent needs `search_terms` or `nn_name` from MatchResult.")

        return str(search_terms), sider_risks, str(target)

    
    def run(self, match_result: MatchResult) -> LiteratureResult:
        search_terms, sider_risks, target = self.get_literature_search_inputs(match_result)
        search_out = self.searcher.search(search_terms, sider_risks=sider_risks, target=target)
        abstracts = search_out["abstracts"]
        evidence = search_out.get("evidence_source", "unknown")
        
        entities = self.ner.predict(abstracts, target_context=target)
        signals = self.classifier.classify(entities)
        score = self.scorer.score(signals, evidence)
        
        # Package B1-B4 outputs into the final `LiteratureResult`
        return LiteratureResult.from_pipeline_outputs(
            score_output=score,
            evidence_source=evidence,
            signals=signals,
            search_terms=search_terms,
            target=target,
            pmids=search_out.get("pmids", []),
        )
    
if __name__ == "__main__":
    """
    From: `EDA_MolGPT.ipynb`
    "**Top hit T=0.58 vs atorvastatin** for `CC(=O)c1c(F)c(-c2ccc(F)cc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O`
    a clean novel pyrrole-core statin with a trifluoroaryl decoration and the canonical dihydroxy-acid tail."
    -> this smiles will be used as `query_smiles` for demo.
    """
    query_smiles = "CC(=O)c1c(F)c(-c2ccc(F)cc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O"

    # Part 1 `matching_engine.py` now fills `nn_name`, `tanimoto`, `sider_risks`, `search_terms`, and `target`.
    # This replaces the previous demo-only hardcoded MatchResult fields.
    match = MatchingEngine().run(query_smiles)
    
    agent = NLPAgent()
    result = agent.run(match)
    
    print("NLP Agent result")
    print(f"search_terms:          {result.search_terms}")
    print(f"target:                {result.target}")
    print(f"literature_risk_score: {result.literature_risk_score}")
    print(f"evidence_confidence:   {result.evidence_confidence}")
    print(f"evidence_level:        {result.evidence_level}")
    print(f"top_signals ({len(result.top_signals)}):")
    for res in result.top_signals:
        print(f"  {res['signal_type']:15s} | {res['entity']:25s} | {res['weight']:.4f}")
