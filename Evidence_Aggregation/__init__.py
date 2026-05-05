"""
evidence_aggregation — Part 3: Evidence Aggregation
===================================================

Public API
----------
    from Evidence_Aggregation import ValidationPipeline, EvidenceAggregator, RiskScore, RiskWeightConfig
    from Evidence_Aggregation import save_csv_results, save_json_results

    pipeline = ValidationPipeline()
    result = pipeline.run(smiles)

Class index
-----------
    ValidationPipeline  validation_pipeline.py       Part 3 orchestrator (D3)
    EvidenceAggregator  evidence_aggregator.py       Weighted evidence combiner
    RiskScore           risk_score.py                Final output container
    RiskWeightConfig    risk_weight_config.py        Aggregation weight config
    save_csv_results    final_report_generator.py    Batch screening summary export
    save_json_results   final_report_generator.py    Full evidence package export
"""

from .validation_pipeline import ValidationPipeline
from .evidence_aggregator import EvidenceAggregator
from .final_report_generator import save_csv_results, save_json_results
from .risk_score import RiskScore
from .risk_weight_config import RiskWeightConfig

__all__ = [
    "ValidationPipeline",
    "EvidenceAggregator",
    "RiskScore",
    "RiskWeightConfig",
    "save_csv_results",
    "save_json_results",
]
