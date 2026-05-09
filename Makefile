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
	python -m Compound_Matching_Engine.matching_engine

run_nlp_agent:
	cd NLP_Literature_Agent && python -m nlp_agent

run_gnn:
	cd GNN_PPI_Network && python -m network_result

run_pipeline:
	cd Evidence_Aggregation && python -m validation_pipeline