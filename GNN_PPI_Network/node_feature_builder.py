import requests
import pandas as pd
import ast
import numpy as np
import mygene
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from concurrent.futures import ThreadPoolExecutor


class feature_extracter:
    """
    Extracts protein features to construct torch.tensor nodes object for protein-protein interaction 
    network (PPI) using Uniprot data.
    """

    def __init__(
        self,
        proteins=None,
        protein_gene_file="proteins.txt",
        url="https://rest.uniprot.org/uniprotkb/search",
    ):
        """
        Initializes class with proteins by their gene name as a list from a text file

        Parameters:
            proteins : list
                List of unique proteins to get features
            protein_gene_file : str
                File path to text file containing unique proteins
            url : str
                URL of Uniprot REST API for feature search
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

        # mygene 
        mg = mygene.MyGeneInfo()
        self.mg = mg

    def get_protein_gene(self, gene):
        """
        Fetch UniProt data for a single gene.

        Parameters:
            gene : string
                Gene name of protein to extract data

        Returns:
            uniprot_df : DataFrame
                DataFrame containing Uniprot features data for proteins
        """

        query = f'gene_exact:"{gene}" AND organism_id:9606 AND reviewed:true'

        # Specify payload Uniprot will use to search
        # Only extracts features that can be encoded to use for GNN training
        # Returns json format
        payload = {
            "query": query,
            "fields": "accession,ft_domain,cc_subcellular_location,go",
            "format": "json",
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
        uniprot_df["gene_name"] = gene

        return uniprot_df

    def get_uniprot(self, save_path="uniprot_features.csv"):
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

    def clean_uniprot_features(self, features_dataframe=None, input_path: str = 'uniprot_features.csv'):
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
        col = cleaned_uniprot_features["uniProtKBCrossReferences"].values

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
                if isinstance(item, dict) and item.get("database") == "GO":
                    go_ids.append(item.get("id"))
            # Add GO Ids for row to features array
            go_features_array[i] = go_ids

        # Add GO Features back to dataframe
        cleaned_uniprot_features["go_terms"] = go_features_array

        # If duplicates genes merge
        cleaned_uniprot_features = (
            cleaned_uniprot_features.groupby("gene_name")["go_terms"]
            .apply(lambda lists: list(set(sum(lists, []))))
            .reset_index()
        )

        # Reindex to algin gene name with graph node ordering
        cleaned_uniprot_features = cleaned_uniprot_features.set_index(
            "gene_name"
        ).reindex(self.protein_genes)

        # Add empty list ofor any empty genes
        cleaned_uniprot_features["go_terms"] = cleaned_uniprot_features[
            "go_terms"
        ].apply(lambda x: x if isinstance(x, list) else [])

        # Reset index
        cleaned_uniprot_features = cleaned_uniprot_features.reset_index()

        # Returns a DataFrame of lists of GO strings for each protein
        return cleaned_uniprot_features

    def ontology_split_go_terms(self, df, save_path= 'ontology_split_features.csv'):
        """
        Seperates GO IDs for each protein based on ontology
        biological_process (BP)
        molecular_function (MF)
        cellular_component (CC)

        Parameters
        ----------
        df : DataFrame
            Cleaned dataframe of all GO IDs for each protein

        Returns
        -------
        split_features_df : DataFrame
            Split features DataFrame by ontology for each protein
        """

        # Converts row of IDs into multiple rows 
        df_expl = df.explode("go_terms").rename(columns={"go_terms": "go_id"})

        # Drop empty rows early
        #exploded = exploded.dropna(subset=["go_id"])

        # Gets all unique go_ids for mapping
        unique_go_ids = df_expl["go_id"].unique()
        unique_go_ids = unique_go_ids.tolist()

        go_to_ontology = {}

        # Query mygene for all unique GO IDs
        mg_results = self.mg.querymany(
            unique_go_ids,
            scopes="go",
            fields="go",
            species="human"
        )

        # Loop through results
        for res in mg_results:
            # Extract data for GO ID
            go_data = res.get("go", {})

            # Check what ontology result is
            for ontology in ["BP", "MF", "CC"]:
                if ontology in go_data:
                    ont_data = go_data[ontology]

                    # Reformate all entries to be in list
                    if isinstance(ont_data, dict):
                        ont_data = [ont_data]

                    # Get GO ID for mapping
                    for data in ont_data:
                        go_id = data.get("id")
                        # Build GO ID to ontology mapping
                        if go_id:
                            go_to_ontology[go_id] = ontology

        # Map GO IDs to a ontology 
        df_expl["ontology"] = df_expl["go_id"].map(go_to_ontology)

        # Drop unmapped GO IDs
        df_expl = df_expl.dropna(subset=["ontology"])

        # Rebuild list for each protein by ontology
        ontology_features = (
            df_expl
            .groupby(["gene_name", "ontology"])["go_id"]
            .apply(list)
            .unstack(fill_value=[])
            .reset_index()
        )

        # Merge features back to original dataframe and drop go_terms column
        split_features_df = df.merge(ontology_features, on="gene_name", how="left")

        # Fill missing with empty lists
        for col in ["BP", "MF", "CC"]:
            split_features_df[col] = split_features_df[col].apply(lambda x: x if isinstance(x, list) else [])

        # Drop go_terms column after splitting
        split_features_df = split_features_df.drop(columns=["go_terms"])

        split_features_df.to_csv(save_path, index=False)

        return split_features_df

    def add_protein_sequences(self, df, save_path= 'feature_df.csv'):
        """
        Adds protein sequences to dataframe using MyGene.

        Requires:
            df["gene_name"]
        """

        genes = df["gene_name"].tolist()

        results = self.mg.querymany(
            genes,
            scopes="symbol",
            fields="uniprot.Swiss-Prot.sequence",
            species="human"
        )

        # Build mapping: gene → sequence
        gene_to_seq = {}

        for res in results:
            gene = res.get("query")

            seq = None
            uniprot = res.get("uniprot", {})

            if isinstance(uniprot, dict):
                swiss = uniprot.get("Swiss-Prot")

                if isinstance(swiss, dict):
                    seq = swiss.get("sequence")

            gene_to_seq[gene] = seq

        # Vectorized assignment
        df["protein_sequence"] = df["gene_name"].map(gene_to_seq)
        df["protein_sequence"] = df["protein_sequence"].apply(lambda x: "".join(x) if isinstance(x, list) else x)

        df.to_csv(save_path, index=False)

        return df

    def node_features_encoder(self, input_path=None, df=None):
        """
        Encodes cleaned/split uniprot features and turns them into a torch tensor for GNN training.

            input_path : string
                File path to Uniprot features csv file
            df : DataFrame
                Features DataFrame

        Returns:
            x_tensor : torch.tensor
                Torch tensor of encoded uniprot features for GNN training
        """
        # Check if df or file path was passed
        if df == None and input_path == None:
            raise TypeError(f'DataFrame or file path must be passed to node_features_encoder')
        elif df == None and isinstance(input_path, str):
            features_df = pd.read_csv('input_path')
        elif input_path == None and df != None:
            features_df = df
        else:
            raise TypeError(f'DataFrame or file path must be passed to node_features_encoder')
        
        # Check if features_df is split and if not split
        if 'BP' in features_df.columns:
            features_df = features_df
        elif 'go_terms' in features_df.columns:
            features_df = self.ontology_split_go_terms(df=features_df)
        else:
            features_df = self.clean_uniprot_features(features_dataframe=features_df)
            features_df = self.ontology_split_go_terms(df=features_df)

        # Extract GO features
        go_BP = features_df["BP"].values
        go_CC = features_df["CC"].values
        go_MF = features_df["MF"].values

        # Encode GO IDs into binary matrix for each protein
        mlb = MultiLabelBinarizer()
        x_matrix = mlb.fit_transform(go_data)

        # Transform matrix of encoded features to a torch tensor for GNN
        x_tensor = torch.from_numpy(x_matrix).float()

        return x_tensor

nf_builder = feature_extracter()
df_F = nf_builder.clean_uniprot_features(input_path= 'uniprot_features.csv')
split_df = nf_builder.ontology_split_go_terms(df=df_F)
final_df = nf_builder.add_protein_sequences(df= split_df)