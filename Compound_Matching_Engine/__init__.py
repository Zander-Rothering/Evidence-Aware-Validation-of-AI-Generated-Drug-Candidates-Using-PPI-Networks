"""
compound_matching — Part 1: Compound Matching Engine
=====================================================

Public API
----------
    from Compund_Matching_Engine import MatchingEngine, MatchResult

    engine = MatchingEngine()                  # loads ChEMBL references at startup
    result: MatchResult = engine.evaluate(smiles)   # full A2-A10b pipeline

Class index
-----------
    MatchingEngine      matching_engine.py      Part 1 orchestrator (A2-A10b)
    CompoundLoader      compound_loader.py      ChEMBL / fallback reference set (A4)
    FingerprintEncoder  fingerprint_encoder.py  2048-bit Morgan encoder (A3)
    SimilarityScorer    Similarity_scorer.py    Tanimoto NN search + ML scorer (A5)
    DrugLikenessFilter  Drug_likeness_filter.py PAINS / Brenk / Lipinski (A6, A7)
    ScaffoldExtractor   Scaffold_extractor.py   Murcko scaffold comparison (A8)
    NoveltyChecker      novelty_checker.py      Training-set novelty flag (A9)
    RulesEngine         Rules_engine.py         Deterministic 6-rule tier (A10a)
    RiskClassifier      Risk_classifier.py      RF + RBF-SVM ensemble (A10b)
    StructureVisualiser structure_visualiser.py 2-D structure rendering
    MatchResult         Match_result.py         Part 1 output container -> Part 3
    FilterResult        Filter_result.py        Drug-likeness sub-result
"""

from .matching_engine import parse_smiles, MatchingEngine
from .compound_loader import CompoundLoader
from .fingerprint_encoder import FingerprintEncoder
from .Similarity_scorer import SimilarityScorer, SimilarityResult
from .Drug_likeness_filter import DrugLikenessFilter
from .Scaffold_extractor import ScaffoldExtractor, ScaffoldResult
from .novelty_checker import NoveltyChecker
from .Rules_engine import RulesEngine, RuleVerdict
from .risk_classifier import RiskClassifier
from .structure_visualizer import StructureVisualiser
from .Match_result import MatchResult
from .Filter_result import FilterResult

__all__ = [
    "parse_smiles",
    "MatchingEngine",
    "CompoundLoader",
    "FingerprintEncoder",
    "SimilarityScorer",
    "SimilarityResult",
    "DrugLikenessFilter",
    "ScaffoldExtractor",
    "ScaffoldResult",
    "NoveltyChecker",
    "RulesEngine",
    "RuleVerdict",
    "RiskClassifier",
    "StructureVisualiser",
    "MatchResult",
    "FilterResult",
]
