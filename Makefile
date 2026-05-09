ENV_NAME=project_env

.PHONY: pc_env update_pc mac_env update_mac remove run_gnn run_compound_matching run_nlp_agent run_pipeline

pc_env:
	conda env create -f environment_pc.yml

update_pc:
	conda env update -n $(ENV_NAME) -f environment_pc.yml --prune

mac_env:
	conda env create -f environment_mac.yml

update_mac:
	conda env update -n $(ENV_NAME) -f environment_mac.yml --prune

remove:
	conda remove -n $(ENV_NAME) --all -y

run_compound_matching:
	cd Compound_Matching_Engine && python matching_engine.py

run_gnn:
	cd GNN_PPI_Network && python network_result.py

run_nlp_agent:
	cd NLP_Literature_Agent && python literature_result.py

run_pipeline:
	cd Evidence_Aggregation && python validation_pipeline.py