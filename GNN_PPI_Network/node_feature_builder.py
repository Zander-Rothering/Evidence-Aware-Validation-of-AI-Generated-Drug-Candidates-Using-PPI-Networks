import requests
import pandas as pd
import ast
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from concurrent.futures import ThreadPoolExecutor

class feature_extracter:
    def __init__(self, protein_gene_file= "proteins.txt", url = "https://rest.uniprot.org/uniprotkb/search"):
        """
        Initializes class with proteins by their gene name as a list from a text file
        """
        # Converts text file of proteins gene names to list
        with open(protein_gene_file, "r") as f:
            protein_genes = [line.strip() for line in f if line.strip()] 
        
        self.protein_genes = protein_genes
        
        # Uniprot REST API url
        self.url = url

    def get_protein_gene(self, gene):
        """
        Fetch UniProt data for a single gene.

        gene : string
            Gene name of protein to extract data
        """

        query = f'gene_exact:"{gene}" AND organism_id:9606 AND reviewed:true'
            
        # Specify payload Uniprot will use to search
        # Only extracts features that can be encoded to use for GNN training
        # Returns json format
        payload = {
            'query': query,
            'fields': 'accession,ft_domain,cc_subcellular_location,go',
            'format': 'json'
        }

        # Uniprot response for each protein
        uniprot_response = requests.get(self.url, params=payload, timeout=10)
        # Raise error if issue with search
        uniprot_response.raise_for_status()

        # Uniprot response data
        uniprot_data = uniprot_response.json()
        # Extract results from Uniprot response data
        uniprot_results = uniprot_data.get("results", []) 

        # GO IDs under uniProtKBCrossReferences
        uniprot_df = pd.json_normalize(uniprot_results)

        # Adding gene name column (used to ensure index order matches original proteins file)
        uniprot_df['gene_name'] = gene

        return uniprot_df

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

        # Parallel API requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            # List of protein features data
            dfs = list(executor.map(self.fetch_gene, self.protein_genes))

        # Concatenate proteins data lists to single DataFrame
        uni_features_df = pd.concat(dfs, ignore_index=True)

        # Save DataFrame as csv file after concatenating
        uni_features_df.to_csv(save_path, index=False)

        return uni_features_df

    def clean_uniprot_features(self, features_dataframe=None, input_path: str=None):
        """
        Cleans extracted Uniprot features data to get GO IDs only for encoding

        Parameters:
            features_dataframe : pd DataFrame
                DataFrame of uniprot features data as nested dictionaries
            input_path : string
                File path to Uniprot features csv file
        
        Returns:
            cleaned_uniprot_features : np.array
                Array of lists of GO IDs for each protein
        """
        # Check if file path was given if not use passed DataFrame
        if input_path is not None:
            df = pd.read_csv(input_path)
        else:
            df = features_dataframe

        # Re-index DataFrame to ensure order matches original proteins file
        df = df.set_index('gene_name').reindex(self.protein_genes).reset_index()

        # Extract GO IDs data column and convert to numpy array
        go_data = df['uniProtKBCrossReferences'].to_numpy()
        
        # Pre-allocate array to store GO IDs lists
        cleaned_uniprot_features = np.empty(len(go_data), dtype=object)
        
        # Loop through go_data
        for i in range(len(go_data)):
            # Get current row
            row = go_data[i]

            # Check if row is NaN or empty
            if pd.isna(row) or row in ("", "[]"):
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
    
    def node_features_encoder(self, input_path= 'uniprot_features.csv'):
        """
        Encodes cleaned uniprot features and turns them into a torch tensor for GNN training.

            input_path : string
                File path to Uniprot features csv file

        Returns:
            x_tensor : torch tensor
                Torch tensor of encoded uniprot features for GNN training
        """
        # Get clean uniprot features
        cleaned_features = self.clean_uniprot_features(input_path)
        
        # Encode GO IDs into binary matrix for each protein
        mlb = MultiLabelBinarizer()
        x_matrix = mlb.fit_transform(cleaned_features)

        # Transform matrix of encoded features to a torch tensor for GNN
        x_tensor = torch.from_numpy(x_matrix).float()

        return x_tensor
