import requests
import pandas as pd
import ast
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from concurrent.futures import ThreadPoolExecutor

class feature_extracter:
    def __init__(self, proteins= None, protein_gene_file= "proteins.txt", url = "https://rest.uniprot.org/uniprotkb/search"):
        """
        Initializes class with proteins by their gene name as a list from a text file
        """
        if proteins == None:
            # Converts text file of proteins gene names to list
            with open(protein_gene_file, "r") as f:
                protein_genes = [line.strip() for line in f if line.strip()]
        else:
            protein_genes = proteins
        
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
            dfs = list(executor.map(self.get_protein_gene, self.protein_genes))

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
            cleaned_uniprot_features = pd.read_csv(input_path)
        else:
            cleaned_uniprot_features = features_dataframe.copy()

        # Pre-allocate space for GO features
        df_len = len(cleaned_uniprot_features)

        go_features_array = np.empty(df_len, dtype=object)

        # Parse GO Features
        col = cleaned_uniprot_features['uniProtKBCrossReferences'].values

        for i in range(df_len):
            # Get uniProtKBCrossReferences for current row
            row = col[i]

            # Check if row is NaN or empty
            # Empty if no GO features
            if row is None:
                go_features_array[i] = []
                continue

            if isinstance(row, float) and pd.isna(row):
                go_features_array[i] = []
                continue

            if isinstance(row, (list, np.ndarray)):
                row_items = row
            else:
                row_str = str(row)

                if row_str.strip() in ("", "[]"):
                    go_features_array[i] = []
                    continue
            
            try:
                # Converts row strings into list of dictionaries to parse GO IDs from 
                row_items = ast.literal_eval(row)
            except (ValueError, SyntaxError):
                # Empty if error
                go_features_array[i] = []
                continue
            
            go_ids = []
            for item in row_items:
                # Parse GO IDs from row
                if isinstance(item, dict) and item.get('database') == 'GO':
                    go_ids.append(item.get('id'))
            # Add GO Ids for row to features array
            go_features_array[i] = go_ids

        # Add GO Features back to dataframe
        cleaned_uniprot_features['go_terms'] = go_features_array

        # If duplicates genes merge
        cleaned_uniprot_features = (cleaned_uniprot_features.groupby('gene_name')['go_terms'].apply(lambda lists: list(set(sum(lists, [])))).reset_index())

        # Reindex to algin gene name with graph node ordering
        cleaned_uniprot_features = cleaned_uniprot_features.set_index('gene_name').reindex(self.protein_genes)

        # Add empty list ofor any empty genes
        cleaned_uniprot_features['go_terms'] = cleaned_uniprot_features['go_terms'].apply(lambda x: x if isinstance(x, list) else [])
        
        # Reset index
        cleaned_uniprot_features = cleaned_uniprot_features.reset_index()

        # Returns a numpy array of lists of GO strings for each protein
        return cleaned_uniprot_features
    
    def node_features_encoder(self, features_input):
        """
        Encodes cleaned uniprot features and turns them into a torch tensor for GNN training.

            input_path : string
                File path to Uniprot features csv file

        Returns:
            x_tensor : torch tensor
                Torch tensor of encoded uniprot features for GNN training
        """
        # Get clean uniprot features
        if isinstance(features_input, str):
            cleaned_features = self.clean_uniprot_features(input_path = features_input)
        else:
            cleaned_features = self.clean_uniprot_features(features_dataframe = features_input)
        
        # Extract GO features
        go_data = cleaned_features['go_terms'].values

        # Encode GO IDs into binary matrix for each protein
        mlb = MultiLabelBinarizer()
        x_matrix = mlb.fit_transform(go_data)

        # Transform matrix of encoded features to a torch tensor for GNN
        x_tensor = torch.from_numpy(x_matrix).float()

        return x_tensor
