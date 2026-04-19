"""Literature Risk Scorer — Part 2A, Activity B4.

Converts signal list (B3) into a numeric risk score 0-1.
Score is down-weighted if evidence came from class-level
fallback rather than direct compound literature.

"""


# Evidence level weights — direct compound hit scores higher than fallback
EVIDENCE_WEIGHTS = {
    "level_0": 1.00,  # compound + target (most direct)
    "level_1": 0.80,  
    "level_2": 0.60,  
    "level_3": 0.40, 
}


class LiteratureRiskScorer:
    """Convert B3 signals into a single literature_risk_score 0-1."""
    
    def __init__(self) -> None:
        pass

