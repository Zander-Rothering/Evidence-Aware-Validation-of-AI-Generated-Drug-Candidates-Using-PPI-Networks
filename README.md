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

## Pipeline Architecture

### Compound Matching Engine Architecture

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

### NLP Agent Results

### GNN Results

### Evidence Aggregation Results