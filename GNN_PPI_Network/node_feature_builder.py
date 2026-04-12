import requests
import pandas as pd
import ast
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import torch

class feature_extracter:
    def __init__(self, protein_gene_file= "proteins.txt"):
        """
        Initializes list of proteins by their gene name
        """
        # Converts text file of proteins gene names to list
        with open(protein_gene_file, "r") as f:
            protein_genes = [line.strip() for line in f if line.strip()] 
        
        self.protein_genes = protein_genes

    def get_uniprot(self, save_path= "uniprot_features.csv"):
        """
        Uses Uniprot REST API to extract data for each protein by its gene name

        Parameters:
            save_path : string
                Save path for csv to store Uniprot data for future access
        
        Returns:
            uni_features_df : Pandas DataFrame
                DataFrame containing Uniprot features data for each protein by its gene name
        """
        # Uniprot REST API url
        url = "https://rest.uniprot.org/uniprotkb/search"

        uniprot_features = []

        # Loop through each proteins gene name
        for g in self.protein_genes:
            # Query Uniprot will use to search
            # Organism ID corresponds to Homo Sapiens; Will only pull reviewed data
            query = f'gene_exact:"{g}" AND organism_id:9606 AND reviewed:true'
            
            # Specify payload Uniprot will use to search
            # Only extracts features that can be encoded to use for GNN training
            # Returns json format
            payload = {
                'query': query,
                'fields': 'accession,ft_domain,cc_subcellular_location,go',
                'format': 'json'
            }

            # Uniprot response for each protein
            uniprot_response = requests.get(url, params=payload)
            # Raise error if issue with search
            uniprot_response.raise_for_status()

            # Uniprot response data
            uniprot_data = uniprot_response.json()
            # Extract results from Uniprot response data
            uniprot_results = uniprot_data.get("results", []) 

            # GO IDs under uniProtKBCrossReferences
            uniprot_df = pd.json_normalize(uniprot_results)
            uniprot_features.append(uniprot_df)

            # Concatanate features to DataFrame
            uni_features_df = pd.concat(uniprot_features, ignore_index=True)

            # Save Uniprot features DataFrame as csv for later use
            uni_features_df.to_csv(save_path, index=False)

        return uni_features_df
    
    def clean_uniprot_features(self, features_dataframe=None, input_path: str=None):
        """
        Cleans extracted uniprot features data to get GO IDs only for encoding

        Parameters:
            features_dataframe : pd DataFrame
                DataFrame of uniprot features data as nested dictionaries
            input_path : string
                File path to uniprot features csv file
        
        Returns:
            clnd_uniprot_ftrs : numpy array
                Lists of GO IDs for each protein to use for encoding
        """
        # Check if file path was given if not use passed DataFrame
        if input_path is not None:
            df = pd.read_csv(input_path)
        else:
            df = features_dataframe

        # Extract GO IDs data column and convert to numpy array
        go_data = df['uniProtKBCrossReferences'].to_numpy()
        
        # Pre-allocate array to store GO IDs lists
        cleaned_uniprot_features = np.empty(len(go_data), dtype=object)
        
        # Loop through go_data
        for i in range(len(go_data)):
            # Get current row
            row = go_data[i]

            # Check if row is NaN or empty
            if np.isnan(row) or row == "" or row == "[]":
                cleaned_uniprot_features[i] = np.array([])
                continue

            # Converts row strings into list of dictionaries to parse GO IDs from       
            row_items = ast.literal_eval(row)
            # Parse GO IDs from row
            go_ids = np.array([item['id'] for item in row_items if item.get('database') == 'GO'])
                
            # Add go_ids array to pre-allocated features array
            cleaned_uniprot_features[i] = go_ids

        # Returns a numpy array of lists of GO strings for each protein
        return cleaned_uniprot_features
    
    def node_features_encoder(self):
        """
        Encodes cleaned uniprot features and turns them into a torch tensor for GNN training.

        Returns:
            x_tensor : torch tensor
                Torch tensor of encoded uniprot features for GNN training
        """
        # Get clean uniprot features
        cleaned_features = self.clean_uniprot_features()
        
        # Encode GO IDs into binary matrix for each protein
        mlb = MultiLabelBinarizer()
        x_matrix = mlb.fit_transform(cleaned_features)

        # Transform matrix of encoded features to a torch tensor for GNN
        x_tensor = torch.from_numpy(x_matrix).float()

        return x_tensor

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
    """
    def get_biological_features(self, protein_list, disgenet_api_key, save_path='protein_features.csv'):
        uniprot_df = self.get_uniprot(protein_list)
        
        entrez_ids = uniprot_df['gene_id'].dropna().unique()

        disgenet_df = self.get_disgenet(entrez_ids, disgenet_api_key)

        uniprot_df['gene_id'] = uniprot_df['gene_id'].astype(str)
        disgenet_df['gene_id'] = disgenet_df['gene_id'].astype(str)
            
        protein_features = pd.merge(uniprot_df, disgenet_df, on='gene_id', how='left')

        protein_features.to_csv(save_path, index=False)

        return protein_features
    """