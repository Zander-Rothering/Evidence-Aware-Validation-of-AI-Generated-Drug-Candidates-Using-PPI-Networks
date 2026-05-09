"""Rules_engine.py — deterministic 6-rule risk-tier classifier (Part 1, Step A10a).

Each rule emits a RuleVerdict (triggered or not) into an evidence trail.
The final tier is the most-severe impact among triggered rules:
    HIGH  > MEDIUM > LOW.
"""
from __future__ import annotations
from dataclasses import dataclass


_TIER_RANK = {"LOW": 0, "NEUTRAL": 0, "MEDIUM": 1, "HIGH": 2}
_RANK_TIER = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}


@dataclass
class RuleVerdict:
    """One rule's outcome — written to the evidence trail whether triggered or not."""
    rule_name: str
    triggered: bool
    triggering_value: float | int | bool
    tier_impact: str   # "HIGH" | "MEDIUM" | "LOW" | "NEUTRAL"
    description: str


class RulesEngine:
    """
    Six-rule deterministic risk classifier.

    Rule order (fixed):
      R1 tanimoto >= 0.95 and not is_novel  -> HIGH  (exact training reproduction)
      R2 tanimoto < 0.30                    -> HIGH  (structural divergence)
      R3 lipinski_violations >= 2           -> HIGH  (bioavailability concern)
      R4 pains_flag or brenk_flag           -> HIGH  (reactive / problematic scaffold)
      R5 sider_count > sider_median         -> MEDIUM(elevated adverse-effect burden)
      R6 qed < 0.34                         -> MEDIUM(low composite drug-likeness)

    Final tier = highest tier_impact among triggered rules (defaults to LOW).
    """

    def __init__(self, sider_median: float = 5.0) -> None:
        # TODO: replace placeholder with the cohort SIDER median computed
        # from the loaded reference set; pass in from MatchingEngine.
        self.sider_median = sider_median

    def evaluate(
        self,
        *,
        tanimoto: float,
        is_novel: bool,
        lipinski_violations: int,
        pains_flag: bool,
        brenk_flag: bool,
        sider_count: int,
        qed: float,
    ) -> tuple[str, list[RuleVerdict]]:
        """Apply all 6 rules and return (final_tier, evidence_trail).

        Evidence trail contains all 6 rules in fixed order, regardless of
        whether they triggered. Final tier is the worst tier among triggered
        rules; LOW if none trigger.
        """
        trail: list[RuleVerdict] = []

        # R1 -- exact training reproduction
        r1_trig = (tanimoto >= 0.95) and (is_novel is False)
        trail.append(RuleVerdict(
            rule_name="R1_exact_reproduction",
            triggered=r1_trig,
            triggering_value=round(float(tanimoto), 4),
            tier_impact="HIGH",
            description="Exact training reproduction (Tanimoto >= 0.95 and not novel)",
        ))

        # R2 -- structural divergence
        r2_trig = tanimoto < 0.30
        trail.append(RuleVerdict(
            rule_name="R2_structural_divergence",
            triggered=r2_trig,
            triggering_value=round(float(tanimoto), 4),
            tier_impact="HIGH",
            description="Diverges from known active class (Tanimoto < 0.30)",
        ))

        # R3 -- Lipinski violations
        r3_trig = lipinski_violations >= 2
        trail.append(RuleVerdict(
            rule_name="R3_lipinski_violations",
            triggered=r3_trig,
            triggering_value=int(lipinski_violations),
            tier_impact="HIGH",
            description=">=2 Lipinski rule violations",
        ))

        # R4 -- reactive / problematic scaffold
        r4_trig = bool(pains_flag) or bool(brenk_flag)
        trail.append(RuleVerdict(
            rule_name="R4_reactive_scaffold",
            triggered=r4_trig,
            triggering_value=bool(pains_flag or brenk_flag),
            tier_impact="HIGH",
            description="PAINS or Brenk structural alert",
        ))

        # R5 -- adverse-effect burden above cohort median
        r5_trig = sider_count > self.sider_median
        trail.append(RuleVerdict(
            rule_name="R5_high_sider_burden",
            triggered=r5_trig,
            triggering_value=int(sider_count),
            tier_impact="MEDIUM",
            description=f"Adverse-effect count above cohort median ({self.sider_median})",
        ))

        # R6 -- low QED
        r6_trig = qed < 0.34
        trail.append(RuleVerdict(
            rule_name="R6_low_qed",
            triggered=r6_trig,
            triggering_value=round(float(qed), 4),
            tier_impact="MEDIUM",
            description="Composite drug-likeness below threshold (QED < 0.34)",
        ))

        worst_rank = 0
        for v in trail:
            if v.triggered:
                worst_rank = max(worst_rank, _TIER_RANK.get(v.tier_impact, 0))
        final_tier = _RANK_TIER[worst_rank]
        return final_tier, trail
