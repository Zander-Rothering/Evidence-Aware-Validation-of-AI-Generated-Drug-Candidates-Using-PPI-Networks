ENV_NAME=project_env

.PHONY: create_pc update_pc create_mac update_mac remove

create_pc:
	conda env create -f environment_pc.yml

update_pc:
	conda env update -n $(ENV_NAME) -f environment_pc.yml --prune

create_mac:
	conda env create -f environment_mac.yml

update_mac:
	conda env update -n $(ENV_NAME) -f environment_mac.yml --prune

remove:
	conda remove -n $(ENV_NAME) --all -y