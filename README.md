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

This command executes the main script located in GNN_PPI_Network/network_result.py and handles the end-to-end workflow.

For individual control of each component of the GNN pipeline, you can also run each python script individually from within the `GNN_PPI_Network` subdirectory.

### Running Risk Aggregator

The full end-to-end pipeline for computing an aggregated risk score for a drug candidate can be executed from the root directory using the provided Makefile.

To run the complete pipeline and generate a final risk score for a drug candidate against the target protein use:

`make run_pipeline`

This command runs `Evidence_Aggregation/validation_pipeline.py`, the final validation orchestrator. It executes the full validation workflow, combining outputs from the Compound Matching Engine (Part 1), the NLP Literature Agent (Part 2A), and the PPI Network GNN (Part 2B). These independent evidence streams are then integrated in Part 3 to produce a final aggregated and explainable risk score for the candidate compound. On completion, it writes `validation_results.csv` and `validation_results.json` (one row/record per candidate) to the `Evidence_Aggregation/` directory.

## Pipeline Architecture