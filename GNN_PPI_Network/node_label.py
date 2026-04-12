import pandas as pd
import requests
class node_labeler():
    def __init__(self, protein_gene_file= "proteins.txt"):
        """
        Initializes list of proteins by their gene name
        """
        # Converts text file of proteins gene names to list
        with open(protein_gene_file, "r") as f:
            protein_genes = [line.strip() for line in f if line.strip()] 
        
        self.protein_genes = protein_genes

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

    def clean_disgenet():
        pass

    def node_labels(protein_features, disease_score_threshold=0.3):
        """
        Creates label for GNN training.
        1 = Strong disease association (High Risk)
        0 = Weak or no association
        """

        # Fill NaNs with 0 (No associations or no data)
        protein_features['score'] = pd.to_numeric(protein_features['score'], errors='coerce').fillna(0)

        # Create labels for proteins based on disease score
        protein_features['label'] = (protein_features['score'] >= disease_score_threshold).astype(int)

        # Create single label for protein based on max labels score
        protein_label = protein_features.groupby('accession')['label'].max()

        return protein_label