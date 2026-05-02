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
