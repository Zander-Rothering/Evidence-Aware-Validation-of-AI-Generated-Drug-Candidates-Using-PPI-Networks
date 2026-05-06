"""D3 validation pipeline orchestrator.

Orchestrator that connects:
- Part 1: Compound matching
- Part 2A: NLP literature analysis
- Part 2B bridge: network risk input normalization/fallback
- Part 3: Evidence aggregation
"""

import csv
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Compound_Matching_Engine import MatchingEngine
from NLP_Literature_Agent import NLPAgent

try:
    from Compound_Matching_Engine.novelty_checker import NoveltyChecker
    from Evidence_Aggregation.evidence_aggregator import EvidenceAggregator
    from Evidence_Aggregation.final_report_generator import save_annotated_csv, save_annotated_json
    from Evidence_Aggregation.risk_score import RiskScore
    from Evidence_Aggregation.risk_weight_config import RiskWeightConfig
except ImportError:
    from Compound_Matching_Engine.novelty_checker import NoveltyChecker
    from evidence_aggregator import EvidenceAggregator
    from final_report_generator import save_annotated_csv, save_annotated_json
    from risk_score import RiskScore
    from risk_weight_config import RiskWeightConfig


DEFAULT_INPUT_CSVS = [
    PROJECT_ROOT / "generation_run" / "v2_validated_novel.csv",
    PROJECT_ROOT / "generation_run" / "v2_T08_validated_novel.csv",
]
# Set to None to run all unique SMILES from both Girish novel-only CSV files.
DEFAULT_BATCH_LIMIT = None # change this number to number of SMILES to run; `None` means all SMILES


class ValidationPipeline:
    """Orchestrator: SMILES -> MatchResult -> LiteratureResult -> NetworkResult -> RiskScore."""

    def __init__(self, config: RiskWeightConfig = None):
        self.matching_engine = MatchingEngine()
        self.nlp_agent = NLPAgent()
        self.aggregator = EvidenceAggregator(config or RiskWeightConfig())

    def run(self, smiles: str) -> RiskScore:
        # Part 1
        match_result = self.matching_engine.run(smiles)

        # Part 2A (PDF says parallel with Part 2B, but sequential is fine for MVP)
        literature_result = self.nlp_agent.run(match_result)

        # Part 2B priority order:
        # 1) Use NetworkResult if available (future full integration path).
        # 2) If this returns None, EvidenceAggregator can use explicit network_risk input.
        # 3) If both are missing, EvidenceAggregator reads gnn_files/network_risk.txt fallback.
        network_result = self._try_load_network_result()

        # Part 3
        return self.aggregator.aggregate(match_result=match_result, literature_result=literature_result, network_result=network_result)

    def run_batch(self, smiles_list: list[str]) -> list[RiskScore]:
        """Run the validation pipeline for multiple generated SMILES."""
        results = []
        for smiles in self.with_progress(smiles_list, "Validation pipeline"):
            results.append(self.run(smiles))
        return results

    @staticmethod
    def with_progress(items: list[str], label: str):
        """Yield SMILES with tqdm progress when available."""
        if tqdm is None:
            return items
        return tqdm(items, desc=label, unit="mol")

    @staticmethod
    def annotate_pubchem_novelty(smiles_list: list[str]) -> dict[str, dict]:
        """Add final-output PubChem novelty annotations for each SMILES."""
        checker = NoveltyChecker(check_novelty=True)
        novelty_by_smiles = {}
        for smiles in ValidationPipeline.with_progress(smiles_list, "PubChem novelty"):
            result = checker.is_novel(smiles)
            novelty_by_smiles[smiles] = {
                "is_pubchem_novel": result["is_novel"],
            }
        return novelty_by_smiles

    @staticmethod
    def load_smiles_from_csv(csv_paths: list[Path] = None, limit: int | None = DEFAULT_BATCH_LIMIT) -> list[str]:
        """Load de-duplicated generated SMILES from one or more validation CSV files."""
        smiles_list = []
        seen = set()

        for csv_path in (csv_paths or DEFAULT_INPUT_CSVS):
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    smiles = (row.get("smiles") or "").strip()
                    if not smiles or smiles in seen:
                        continue
                    smiles_list.append(smiles)
                    seen.add(smiles)
                    if limit is not None and len(smiles_list) >= limit:
                        return smiles_list

        return smiles_list

    def _try_load_network_result(self):
        """Try to load future NetworkResult; return None if not implemented yet."""
        try:
            from GNN_PPI_Network.network_result import NetworkResult
            # Primary path: use Part 2B structured output when the loader exists.
            return NetworkResult.load_latest()
        except (ImportError, AttributeError):
            # Returning None is intentional:
            # aggregate(..., network_result=None) tells EvidenceAggregator to continue with its
            # internal fallback chain (explicit network_risk -> network_risk.txt).
            return None

if __name__ == "__main__":
    smiles_batch = ValidationPipeline.load_smiles_from_csv()
    print(f"Loaded {len(smiles_batch)} unique SMILES from Girish novel-only CSV files")
    pipeline = ValidationPipeline()
    # Girish novel-only CSVs remove exact ChEMBL duplicates
    # NoveltyChecker adds layer of sanity check and one PubChem novelty-check field
    novelty_by_smiles = ValidationPipeline.annotate_pubchem_novelty(smiles_batch)
    print("Running validation pipeline...")
    results = pipeline.run_batch(smiles_batch)

    print(f"ValidationPipeline batch result ({len(results)} molecules)")
    for result in results:
        print(result)

    annotated_rows = []
    for result in results:
        row = result.to_dict()
        row["is_pubchem_novel"] = novelty_by_smiles.get(
            result.query_smiles, {"is_pubchem_novel": ""}
        )["is_pubchem_novel"]
        annotated_rows.append(row)

    print("Saving validation outputs...")
    save_annotated_csv(annotated_rows, "validation_results.csv")
    save_annotated_json(annotated_rows, "validation_results.json")
