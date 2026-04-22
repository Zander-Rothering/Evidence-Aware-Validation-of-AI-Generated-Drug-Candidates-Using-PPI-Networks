"""
B5 - NLP Agent orchestrator.

Wires B1 -> B2 -> B3 -> B4 together.
Takes a MatchResult from Part 1, returns a LiteratureResult.

TODO: caching, retry, proper LiteratureResult dataclass
TODO: batch processing 
"""

from pubmed_searcher import PubMedSearcher
from ner_extractor import NERExtractor
from signal_classifier import SignalClassifier
from literature_risk_scorer import LiteratureRiskScorer

from Compund_Matching_Engine import MatchResult


class NLPAgent:
    def __init__(self):
        self.searcher = PubMedSearcher()
        self.ner = NERExtractor()
        self.classifier = SignalClassifier()
        self.scorer = LiteratureRiskScorer()
    
    def run(self, match_result: MatchResult) -> dict:
        pass