import requests
import pandas as pd

class feature_extracter:
    def __init__(self, proteins_file= "proteins.txt"):
        with open(proteins_file, "r") as f:
            proteins = [line.strip() for line in f if line.strip()] 
        
        self.proteins = proteins

    def get_uniprot(self, save_path= "uniprot_features.csv"):
        url = "https://rest.uniprot.org/uniprotkb/search"

        uniprot_features = []

        for p in self.proteins:
            query = f'gene_exact:"{p}" AND organism_id:9606'
            payload = {
                'query': query,
                'fields': 'accession,gene_names,ft_domain,cc_function,cc_subcellular_location,go,xref_kegg,xref_reactome',
                'format': 'json'
            }

            uniprot_response = requests.get(url, params=payload)
            uniprot_response.raise_for_status()

            uniprot_data = uniprot_response.json()
            uniprot_results = uniprot_data.get("results", []) 
            
            uniprot_df = pd.json_normalize(uniprot_results)
            uniprot_features.append(uniprot_df)

            uni_proteins_df = pd.concat(uniprot_features, ignore_index=True)

            uni_proteins_df.to_csv(save_path, index=False)

        return uni_proteins_df

    def get_disgenet(self, api_key='37587e7d-5f2c-4434-a3c6-54fad187142b', save_path="disgenet_features.csv"):
        
        base_url = "https://disgenet.com"
        headers = {"api_key": api_key, "Accept": "application/json"}

        disgenet_features = []

        for p in self.proteins:
            disgenet_response = requests.get(f"{base_url}/{p}", headers=headers)
                
            disgenet_data = disgenet_response.json()
            disgenet_results = disgenet_data.get("results", []) 
            disgenet_df = pd.json_normalize(disgenet_results)
            disgenet_features.append(disgenet_df )

            disgenet_proteins_df = pd.concat(disgenet_features, ignore_index=True)
            disgenet_proteins_df.to_csv(save_path, index=False)
        
        return disgenet_proteins_df

    def get_biological_features(self, protein_list, disgenet_api_key, save_path='protein_features.csv'):
        uniprot_df = self.get_uniprot(protein_list)
        
        entrez_ids = uniprot_df['gene_id'].dropna().unique()

        disgenet_df = self.get_disgenet(entrez_ids, disgenet_api_key)

        uniprot_df['gene_id'] = uniprot_df['gene_id'].astype(str)
        disgenet_df['gene_id'] = disgenet_df['gene_id'].astype(str)
            
        protein_features = pd.merge(uniprot_df, disgenet_df, on='gene_id', how='left')

        protein_features.to_csv(save_path, index=False)

        return protein_features

class feature_encoder:
    def __init__(self, features_file='protein_features.csv'):
        features_df = pd.read_csv(features_file)

        self.features_df = features_df