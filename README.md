# Evidence-Aware-Validation-of-AI-Generated-Drug-Candidates-Using-PPI-Networks

This project integrates a compound matching engine, a natural language processing (NLP) agent, and a graph neural network (GNN) built on a protein–protein interaction (PPI) network to assess the risk profile of potential drug candidates. The compound matching engine uses molecular similarity metrics to compare a candidate against known compounds and generate a structure based risk score. The NLP agent analyzes scientific literature to extract relevant biomedical signals and produce a literature informed risk score. The GNN models relationships within the PPI network to capture biological interactions associated with the candidate, yielding a biological context risk score. Together, these components provide a unified risk assessment that combines structural, literature-derived, and biological insights into a single, comprehensive evaluation.

## Environment Setup

This project is setup to use a Conda environment to manage dependencies across operating systems. A Makefile is provided for simplified environmental creation, updates, and removal. Prior to running Makefile commands ensure that Conda is installed. Please note that dependencies may take a substantial amount of time to install.

### Environment Configurations

Two environmental configurations are available for installation based on the operating system you are running:

`environmental_pc.yml` for Windows/Linux (CUDA-enabeled systems)

`environmental_mac.yml` for macOS (Intel or Apple Silicon with MPS support)

### Environment Creation

The following Makefile commands can be used for creation of the project environment:

`make pc_env` for Windows/Linux

`make mac_env` for macOS

### Environment Updates

If updates to dependencies are required, please add them to the appropriate environment `yml` and run the following Makefile commands:

`make update_pc` for Windows/Linux

`make update_mac` for macOS

### Environment Removal

To remove the environment and all dependencies run:

`make remove`

### Environment Troubleshooting

It is recommended to install all dependencies using the Makefile command to avoid installation issues. If issues arise when installing dependencies all may be installed via `pip` with the exception of `RDKit` which should be installed via `conda-forge`.

## Repository Structure

`Compound_Matching_Engine/` SMILES-based compound similarity and risk scoring pipeline

`NLP_Literature_Agent/` Literature-based compound risk scoring pipeline

`GNN_PPI_Network/` PPI network construction and GNN risk scoring pipeline

`Evidence_Aggregation/` Evidence compiler to determine final risk score for pipeline

`MolGPT_Generation/` Upstream MolGPT v2 fine-tuning, sampling, validation, and rendering pipeline that produces the candidate SMILES consumed by the validation pipeline

## Pipeline Architecture

### MolGPT Generation Pipeline

The MolGPT Generation Pipeline is the upstream workflow that produces the AI-generated candidate SMILES which the four-component validation pipeline then evaluates. It fine-tunes a transformer-decoder generative model on filtered HMGCR inhibitor SMILES from ChEMBL402, samples thousands of candidates from the fine-tuned model, applies pharmacophore and drug-likeness filters, and renders the top-ranked novel candidates as a structured summary. The scripts live in `MolGPT_Generation/` and are invoked directly with `python -m MolGPT_Generation.<script>` rather than through the Makefile, since most steps are one-time or rerun-rarely operations rather than recurring pipeline runs.

The fetch step uses `fetch_chembl402.py` to pull all activity records for target CHEMBL402 from the ChEMBL REST API and writes the result to `MolGPT_Generation/data/chembl402_activities.tsv`. The corpus diagnostic step uses `analyze_corpus.py` to apply the project's filter chain (activity, pharmacophore, property window, dedup) to the raw activity records and report compound survival counts at each stage; this script is read-only and produces no downstream artifact. The training data preparation step uses `augment_smiles.py` to generate randomised non-canonical SMILES variants of the cleaned 51-compound reference set in `datasets/statin_filtered.csv` and writes the augmented corpus to `datasets/statin_augmented.csv` with a strict molecule-level train/val split.

The actual MolGPT v2 fine-tuning is performed inside the `molgpt/` submodule using `python train/train.py` and is not orchestrated by `MolGPT_Generation/`; the resulting checkpoint lands at `cond_gpt/weights/statin_model_v2.pt`. The generation step uses `generate_statins_v2.py` to load the trained checkpoint, sample 5000 candidate SMILES at temperature 1.0 from the v2 weights, and write `MolGPT_Generation/generation_run/v2_candidates.csv` and `v2_valid.smi`. The validation step uses `validate_statins_v2.py` to apply the HMG warhead pharmacophore SMARTS, Lipinski Rule-of-Five, PAINS catalog, and Tanimoto similarity scoring against both the 51 ChEMBL training compounds and the 7 marketed statins, writing `v2_validated.csv` to the same `generation_run/` directory.

The rendering step uses `render_v2_top16.py` to draw the top-N novel candidates as a 4x4 PNG grid annotated with warhead, Lipinski, and PAINS flags. The summary step uses `build_v2_summary.py` to filter the validated set to novel-only candidates, build a two-panel summary figure pairing the training loss curve with the Tanimoto similarity distribution, and write a markdown summary suitable for the project report. The `*_validated_novel.csv` files produced at this stage are the input that the validation pipeline reads as its candidate batch.

### Compound Matching Engine Architecture

The Compound Matching Engine is the structural evidence stream of the validation pipeline. It transforms each generated SMILES into a `MatchResult` dataclass through a chained sequence of typed stages (A2 through A10c) that handle parsing, similarity scoring, drug-likeness filtering, and risk classification. The A2 `SmilesParser` validates each input SMILES via RDKit and rejects fragments and small molecules. The A3 `FingerprintEncoder` produces a 2048-bit Morgan circular fingerprint at radius 2, exposing both an RDKit `ExplicitBitVect` for similarity computation and a NumPy array for the ANN classifier from a single generator instance. The A4 `CompoundLoader` queries ChEMBL target CHEMBL402 (HMG-CoA Reductase) for up to 200 reference inhibitors at startup and falls back to seven hardcoded marketed statins when the API is unreachable.

The A5 `SimilarityScorer` ranks the query against the cached reference set using vectorised Tanimoto similarity and returns the nearest neighbour ChEMBL identifier with its raw score. The A6 and A7 `DrugLikenessFilter` computes seven physicochemical descriptors (MW, LogP, HBD, HBA, QED, TPSA, rotatable bonds) and applies the four rule Lipinski check along with the PAINS and Brenk structural alert catalogs. The A8 `ScaffoldExtractor` extracts the Murcko scaffold of the query and its nearest neighbour and returns a scaffold level Tanimoto coefficient that captures core ring system overlap independent of sidechain decoration. The A9 SIDER lookup maps the nearest neighbour ChEMBL identifier to a precompiled adverse effect table for downstream NLP cueing.

The A10a `RulesEngine` is a deterministic six rule decision engine that produces an interpretable risk tier based on Tanimoto thresholds, Lipinski violations, structural alert flags, SIDER adverse effect counts, and QED. The A10b `RiskClassifier` is a small PyTorch ANN trained on weak rule derived structural labels that returns a supporting ML tier and class probability. The A10c reconciliation step compares the rule and ANN verdicts and writes a confidence flag indicating whether they agreed. The final `MatchResult` carries the canonical query SMILES, nearest neighbour metadata, the full feature dictionary, both risk tiers, the model probability, and a structured evidence trail downstream to the NLP Literature Agent and the Evidence Aggregator.

### NLP Agent Architecture

The NLP Literature Agent converts each `MatchResult` into a literature-based risk score using nearest-neighbor compound similarity, HMGCR target context, and SIDER-derived safety signals. Nearest-neighbor ChEMBL identifiers are converted into readable drug names, with exact statin SMILES matched directly and unresolved compounds assigned to the closest marketed statin only when Tanimoto similarity is at least 0.40, preventing weakly related molecules from inheriting overly specific literature evidence. PubMed abstracts are retrieved through the NCBI Entrez using fallback queries, ranging from compound specific HMGCR searches to broader statin class safety literature. Retrieved abstracts are processed using the pretrained biomedical NER model `d4data/biomedical-ner-all` to extract biomedical entities and safety-related signals, which are aggregated by the `LiteratureRiskScorer` into a normalized 0–1 literature risk score for the final Evidence Aggregator.

### PPI GNN Architecture

The PPI GNN was developed to model biological risk across interacting proteins, using the statin target protein HMGCR as a proof of concept seed for network construction. The PPI network is built using interaction data from STRING and BioGRID, where nodes represent proteins and edges represent protein–protein interactions. Protein features are retrieved from UniProt and embedded using pretrained ESM-2 sequence embeddings and Node2Vec Gene Ontology embeddings. These features, along with edge attribute interaction scores and experimental system types, were combined into a PyTorch Geometric Data object for training. Note that the current architecture does not utilize edge attributes during GNN training.

The GNN was implemented using the PyTorch Geometric message-passing framework and consists of three graph convolution layers with ReLU activations and dropout regularization. Node labels are generated from DisGeNET disease-association scores and used for binary node classification. The model is trained using Adam optimization and cross-entropy loss, producing node-level risk probabilities that are aggregated into an overall biological network risk score.

### Risk Evidence Argregation Architecture

The Evidence Aggregator serves as the final decision layer of the pipeline, combining all evidence streams into a single unified risk score. Individual risk scores from each component are weighted and integrated using a normalized weighted sum. The resulting score is then assigned to one of three final risk tiers, HIGH/MEDIUM/LOW, based on both the overall score and agreement between the individual evidence streams. Final predictions, along with all intermediate evidence are exported to a CSV file.

## Running Pipeline

### Running Compound Matching Engine

To run the compound similarity and matching pipeline use:

`make run_compound_matching`

This command runs `Compound_Matching_Engine/matching_engine.py`, which executes the Compound Matching Engine end-to-end on a demo SMILES and prints the resulting `MatchResult`: nearest-neighbor reference compound, Tanimoto similarity, novelty flag, rule-based and ML-based risk tiers, SIDER seeds, PubMed search terms, and the final similarity-based risk score.

### Running NLP Literature Agent

To run the NLP agent for risk score extraction from literature use:

`make run_nlp_agent`

This command runs `NLP_Literature_Agent/nlp_agent.py`, which executes the full NLP Literature Agent end-to-end on a demo SMILES. It first uses the Compound Matching Engine (Part 1) to derive nearest-neighbor search terms, then runs the B1–B4 NLP pipeline (PubMed search -> biomedical NER -> signal classification -> literature risk scoring) and prints the resulting `LiteratureResult`: search terms, target, literature risk score, evidence confidence and level, and the top extracted safety/efficacy signals.

### Running PPI Network GNN

The GNN portion of this pipeline can be executed from the root directory using the provided Makefile.

To run the full GNN pipeline to generate a risk score use:

`make run_gnn`

This command executes the main script located in GNN_PPI_Network/network_result.py and handles the end-to-end workflow. On compeletion, it writes `network_risk.txt` to the `GNN_PPI_Network/gnn_files` directory.

For individual control of each component of the GNN pipeline, you can also run each python script individually from within the `GNN_PPI_Network` subdirectory.

### Running Risk Aggregator

The full end-to-end pipeline for computing an aggregated risk score for a drug candidate can be executed from the root directory using the provided Makefile.

To run the complete pipeline and generate a final risk score for a drug candidate against the target protein use:

`make run_pipeline`

This command runs `Evidence_Aggregation/validation_pipeline.py`, the final validation orchestrator. It executes the full validation workflow, combining outputs from the Compound Matching Engine (Part 1), the NLP Literature Agent (Part 2A), and the PPI Network GNN (Part 2B). These independent evidence streams are then integrated in Part 3 to produce a final aggregated and explainable risk score for the candidate compound. On completion, it writes `validation_results.csv` and `validation_results.json` (one row/record per candidate) to the `Evidence_Aggregation/` directory.

## Pipeline Results

### Compound Matching Results

On the demo SMILES `CC(=O)c1c(F)c(-c2ccc(F)cc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O` (a novel pyrrole core statin from the MolGPT generation run), the Compound Matching Engine populates a `MatchResult` dataclass that becomes the structural input to every downstream stage. The engine encodes the candidate to a Morgan fingerprint, ranks it against the corrected HMGCR reference library loaded from ChEMBL target CHEMBL402, and records the nearest neighbor name, SMILES, IC50, and the maximum Tanimoto similarity. From there it computes a scaffold overlap, a novelty flag against the reference set, an A10a rule tier from the drug likeness filter, an A10b ANN risk tier with class probabilities, and an A10c reconciliation flag that records whether the rule and ANN verdicts agreed.

On the demo compound the engine returns a Tanimoto of 0.5758 against its nearest HMGCR reference, a derived `similarity_risk_score` of 0.4242 (defined as one minus Tanimoto), `target = HMGCR`, an `evidence_level` query seed of `level_0` for downstream PubMed retrieval, and a `confidence_flag` of `REVIEW_RULE_MORE_CONSERVATIVE`, indicating the rules engine assigned a higher tier than the ANN at low ANN confidence. The full `MatchResult` is held in memory and forwarded directly to the NLP Literature Agent and the Evidence Aggregator; the per candidate row is also persisted as the leftmost columns of `Evidence_Aggregation/validation_results.csv`.

### NLP Agent Results

The NLP Literature Agent consumes the `MatchResult` and returns a `LiteratureResult` containing `literature_risk_score`, `evidence_confidence`, `evidence_level`, `top_signals`, `search_terms`, `target`, and the supporting `pmids`. The score is built by the `LiteratureRiskScorer` as the mean weight of NER derived `SAFETY_FLAG` signals minus 0.35 times the mean weight of `EFFICACY` signals, then multiplied by the evidence tier weight tied to which fallback PubMed query succeeded (1.00 for compound plus target at level_0 down to 0.40 for target only at level_3).

On the demo SMILES the agent reaches PubMed at `level_0`, meaning the most specific compound plus target query returned usable abstracts, giving an `evidence_confidence` of 1.00. The biomedical NER model `d4data/biomedical-ner-all` extracts `DRUG`, `PROTEIN`, `DISEASE`, and `ADVERSE_EFFECT` spans from the abstracts, the `SignalClassifier` converts them into weighted signals, and the resulting `literature_risk_score` is 0.6624. The top signals list is sorted by weight and truncated to the top ten by default, while the full `LiteratureResult` is forwarded to the Evidence Aggregator. The compound level numeric outputs (`literature_risk_score` and `evidence_level`) are persisted alongside the other streams in `validation_results.csv` and `validation_results.json`.

### GNN Results

The PPI GNN was trained on a 479 node, 9481 edge protein interaction network seeded on HMGCR and built from STRING and BioGRID. Node features have 384 dimensions, combining ESM-2 sequence embeddings with Node2Vec Gene Ontology embeddings retrieved through UniProt, and binary node labels derived from DisGeNET disease association scores assign 62 of the 479 proteins as positive. The model is the three layer message passing GNN defined in `gnn_model.py` (hidden dimension 128, dropout 0.5), trained with Adam at learning rate 0.001 and weight decay 0.3 under cross entropy loss.

After training, the `RiskAggregator` runs `softmax` on the per node logits and averages the class one probabilities across all 479 proteins to produce a single network risk score. On the demo run the saved aggregate is 0.4918, written verbatim to `GNN_PPI_Network/gnn_files/network_risk.txt` and consumed by the Evidence Aggregator through its Part 2B fallback path. Training metrics are persisted as a per epoch `(train_loss, val_loss, val_acc)` array in `gnn_files/best_training_metrics.npy` and `gnn_files/training_metrics.npy`, with the saved best run reaching a peak validation accuracy of 0.868 and a final logged epoch at `train_loss = 0.447`, `val_loss = 0.453`, `val_acc = 0.868`. Test loss and test accuracy are printed at the end of training but are not persisted to disk.

- Network: 479 nodes, 9481 edges, 384 dimensional node features
- Labels: 62 / 479 positive (DisGeNET disease association)
- Best validation accuracy: 0.868
- Aggregated network risk score on demo: 0.4918

### Evidence Aggregation Results

The Evidence Aggregator combines the three streams using a fixed weight configuration of 0.4 for compound matching, 0.3 for literature, and 0.3 for network, and reads the Part 2B network value either from a future `NetworkResult`, an explicit float argument, or the on disk `gnn_files/network_risk.txt` fallback. The combined score is the weighted sum of `similarity_risk_score`, `literature_risk_score`, and `network_risk_score`, while the final tier is derived from both the combined score (≥ 0.70 → HIGH, ≥ 0.40 → MEDIUM, otherwise LOW) and per stream agreement (any stream ≥ 0.70 lifts the floor to MEDIUM, all three at ≥ 0.70 forces HIGH).

On the demo SMILES the three streams combine as `0.4 × 0.4242 + 0.3 × 0.6624 + 0.3 × 0.4918 = 0.516`, with the literature stream alone above the 0.70 threshold, so the final `risk_tier` is `MEDIUM`. The complete `RiskScore` dataclass (`risk_tier`, `query_smiles`, `similarity_score`, `literature_risk_score`, `network_risk_score`, `combined_score`, `flagged_proteins`, `top_signals`, `ml_proba`, `target`, `search_terms`, `confidence_flag`, `evidence_level`, `pmids`) is held per candidate during a batch run. The compact summary written to `Evidence_Aggregation/validation_results.csv` and `Evidence_Aggregation/validation_results.json` keeps `query_smiles`, `is_pubchem_novel`, `risk_tier`, `combined_score`, `similarity_score`, `literature_risk_score`, `network_risk_score`, `target`, `confidence_flag`, and `evidence_level`. Across the 495 unique novel SMILES processed in the saved batch, 490 candidates land in MEDIUM and 5 in LOW, with no candidate reaching HIGH.