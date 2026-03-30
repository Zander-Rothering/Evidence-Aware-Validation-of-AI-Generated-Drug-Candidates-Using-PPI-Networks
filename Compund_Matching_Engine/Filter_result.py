
#Run PAINS and Brenk Substructure Filters



class FilterResult:

    def __init__(
        self,

        # ── Activity 6 — PAINS and Brenk ─────────────────────
        pains_flag:          bool,   # True = PAINS pattern found
        brenk_flag:          bool,   # True = Brenk pattern found
        pains_matches:       list,   # names of matched PAINS patterns
        brenk_matches:       list,   # names of matched Brenk patterns
        is_clean:            bool,   # True only if both flags False

        # ── Activity 7 — Lipinski descriptors ────────────────
        mol_weight:          float,  # molecular weight in Da
        logp:                float,  # lipophilicity
        hbd:                 int,    # hydrogen bond donors
        hba:                 int,    # hydrogen bond acceptors
        lipinski_violations: int,    # count of failed RO5 rules

        # ── Activity 7 — QED and supporting descriptors ──────
        qed:                 float,  # drug-likeness score 0 to 1
        tpsa:                float,  # topological polar surface area
        rotatable_bonds:     int,    # molecular flexibility measure

    ):
        # ── Activity 6 fields ─────────────────────────────────
        self.pains_flag          = pains_flag
        self.brenk_flag          = brenk_flag
        self.pains_matches       = pains_matches
        self.brenk_matches       = brenk_matches
        self.is_clean            = is_clean

        # ── Activity 7 fields ─────────────────────────────────
        self.mol_weight          = mol_weight
        self.logp                = logp
        self.hbd                 = hbd
        self.hba                 = hba
        self.lipinski_violations = lipinski_violations
        self.qed                 = qed
        self.tpsa                = tpsa
        self.rotatable_bonds     = rotatable_bonds

    def __repr__(self):
        return (
            f"FilterResult("
            f"is_clean={self.is_clean}, "
            f"pains={self.pains_flag}, "
            f"brenk={self.brenk_flag}, "
            f"lipinski_violations={self.lipinski_violations}, "
            f"qed={self.qed:.2f}, "
            f"mw={self.mol_weight:.1f}, "
            f"logp={self.logp:.2f})"
        )
