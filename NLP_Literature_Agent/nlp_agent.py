"""
B5 - NLP Agent orchestrator.

Wires B1 -> B2 -> B3 -> B4 together.
Takes a MatchResult from Part 1, returns a LiteratureResult.

"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from Compund_Matching_Engine import MatchResult

from pubmed_searcher import PubMedSearcher
from ner_extractor import NERExtractor
from signal_classifier import SignalClassifier
from literature_risk_scorer import LiteratureRiskScorer



class NLPAgent:
    def __init__(self):
        self.searcher = PubMedSearcher()
        self.ner = NERExtractor()
        self.classifier = SignalClassifier()
        self.scorer = LiteratureRiskScorer()
    
    def run(self, match_result: MatchResult) -> dict:
        compound = match_result.nn_name
        target = "HMGCR"
        search_out = self.searcher.search(compound)
        abstracts = search_out["abstracts"]
        evidence = search_out.get("evidence_source", "unknown")
        
        entities = self.ner.predict(abstracts, target_context=target)
        signals = self.classifier.classify(entities)
        score = self.scorer.score(signals, evidence)
        
        return {
            "score_output": score,       
            "evidence_source": evidence,
            "signals": signals,
        }
    


if __name__ == "__main__":
    """
    From: `EDA_MolGPT.ipynb`
    "**Top hit T=0.58 vs atorvastatin** for `CC(=O)c1c(F)c(-c2ccc(F)cc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O`
    a clean novel pyrrole-core statin with a trifluoroaryl decoration and the canonical dihydroxy-acid tail."
    -> this smiles will be used as `query_smiles` for demo.
    """
    match = MatchResult(
        query_smiles="CC(=O)c1c(F)c(-c2ccc(F)cc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
        nn_name="atorvastatin",
        nn_smiles="CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4", #atorvastatin: https://pubchem.ncbi.nlm.nih.gov/#query=C33H35FN2O5
        tanimoto="",
        is_novel=True,
    )
    
    agent = NLPAgent()
    result = agent.run(match)
    
    print("NLP Agent result")
    print(f"literature_risk_score: {result['score_output']['literature_risk_score']}")
    print(f"evidence_confidence:   {result['score_output']['evidence_confidence']}")
    print(f"evidence_source:       {result['evidence_source']}")
    print(f"signals ({len(result['signals'])}):")
    for res in result["signals"]:
        print(f"  {res['signal_type']:15s} | {res['entity']:25s} | {res['weight']:.4f}")