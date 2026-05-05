"""
nlp_literature_agent — Part 2: NLP Literature Agent
====================================================

Public API
----------
    from NLP_Literature_Agent import NLPAgent, LiteratureResult

    agent = NLPAgent()
    result: LiteratureResult = agent.run(match_result)

Class index
-----------
    NLPAgent              nlp_agent.py               Part 2 orchestrator
    PubMedSearcher        pubmed_searcher.py         PubMed search
    NERExtractor          ner_extractor.py           Named Entity Recognition
    SignalClassifier      signal_classifier.py       Signal classification
    LiteratureRiskScorer  literature_risk_scorer.py  LiteratureRisk score computation
    LiteratureResult      literature_result.py       Part 2 output container -> Part 3
"""

from .nlp_agent import NLPAgent
from .literature_result import LiteratureResult
from .literature_risk_scorer import LiteratureRiskScorer
from .pubmed_searcher import PubMedSearcher
from .ner_extractor import NERExtractor
from .signal_classifier import SignalClassifier

__all__ = [
    "NLPAgent",
    "LiteratureResult",
    "LiteratureRiskScorer",
    "PubMedSearcher",
    "NERExtractor",
    "SignalClassifier",
]