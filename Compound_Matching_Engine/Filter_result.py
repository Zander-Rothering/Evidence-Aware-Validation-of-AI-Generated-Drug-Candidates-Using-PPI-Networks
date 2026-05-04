"""Filter_result.py — container for drug-likeness filter outputs (Part 1, Steps A6/A7)."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class FilterResult:
    """Holds PAINS/Brenk flags and Lipinski descriptor values for one compound."""

    # PAINS / Brenk flags (A6)
    pains_flag: bool = False
    brenk_flag: bool = False
    matches: List[str] = field(default_factory=list)   # names of matched alert patterns
    is_clean: bool = True                               # True iff both flags are False

    # Lipinski descriptors (A7)
    mw: float = 0.0
    logp: float = 0.0
    hbd: int = 0
    hba: int = 0
    violations: int = 0   # Lipinski violations (0–4); >=2 → HIGH risk
    qed: float = 0.0      # <0.34 on a MEDIUM result → escalates to HIGH
    tpsa: float = 0.0
    rot_bonds: int = 0

    # Aggregated verdict contributed to ML feature vector
    lipinski_pass: bool = True   # violations == 0

    def to_ml_features(self) -> dict:
        """Return flat dict for Random Forest feature assembly."""
        return {
            "pains_flag":    int(self.pains_flag),
            "brenk_flag":    int(self.brenk_flag),
            "mw":            self.mw,
            "logp":          self.logp,
            "hbd":           self.hbd,
            "hba":           self.hba,
            "violations":    self.violations,
            "qed":           self.qed,
            "tpsa":          self.tpsa,
            "rot_bonds":     self.rot_bonds,
            "lipinski_pass": int(self.lipinski_pass),
        }

    def __repr__(self) -> str:
        return (
            f"FilterResult(pains={self.pains_flag}, brenk={self.brenk_flag}, "
            f"viol={self.violations}, qed={self.qed:.3f}, clean={self.is_clean})"
        )
