ENV_NAME=project_env

.PHONY: pc_env update_pc mac_env update_mac remove

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

run_gnn:
	cd GNN_PPI_Network && python network_result.py

run_compound_matching:

run_nlp_agent: