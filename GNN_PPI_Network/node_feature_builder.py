import requests
import pandas as pd
import ast
import numpy as np
import os
import torch
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
from transformers import AutoTokenizer, EsmModel
from node2vec import Node2Vec
from goatools.obo_parser import GODag


class feature_extracter:
    """
    Extracts protein features to construct torch.tensor nodes object for protein-protein interaction
    network (PPI) using Uniprot data.
    """

    def __init__(
        self,
        proteins=None,
        protein_gene_file="features_files/proteins.txt",
        url="https://rest.uniprot.org/uniprotkb/search",
        obo_path="features_files/go-basic.obo",
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

        # Load ESM-2 model for amino acid sequence encoding
        model_name = "facebook/esm2_t6_8M_UR50D"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name)

        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model

        # Load GO Graph and Node2Vec for GO ID encoding
        godag = GODag(obo_path)
        G = nx.DiGraph()
        for go_id, term in godag.items():
            for parent in term.parents:
                G.add_edge(parent.id, go_id)

        # Training Node2Vec (this may take a moment on init)
        n2v = Node2Vec(G, dimensions=64, walk_length=20, num_walks=50, workers=4)
        self.n2v_model = n2v.fit(window=10, min_count=1)

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
            "fields": "accession,sequence,ft_domain,cc_subcellular_location,go",
            "format": "json",
        }

        # Uniprot response for each protein
        uniprot_response = requests.get(self.url, params=payload, timeout=(5, 30))
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

    def load_data(self, data=None):
        """
        Loads data from either a DataFrame or CSV file path as a DataFrame

        Parameters:
            data : DataFrame or str
                Either a DataFrame or a file path to a CSV

        Returns:
            df : DataFrame
                DataFrame of data
        """
        # Loads a copy if already a DataFrame
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            return df

        # Loads csv file if str file path
        elif isinstance(data, str):
            if not os.path.exists(data):
                raise FileNotFoundError(f"File not found: {data}")
            df = pd.read_csv(data)
            return df

        # No file loaded if no DataFrame or file path raise error
        elif data is None:
            raise ValueError(f"DataFrame or file path must be passed.")

        # Incorrect type of data is passed raise error
        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. Expected DataFrame or file path."
            )

    def get_uniprot(self, save_path="features_files/uniprot_features.csv"):
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

    def clean_uniprot_features(
        self,
        uniprot_data="features_files/uniprot_features.csv",
        save_path="features_files/cleaned_uniprot.csv",
    ):
        """
        Cleans extracted Uniprot features data to get GO IDs only for encoding

        Parameters:
            uniprot_data : DataFrame or str
                DataFrame or str file path of uniprot features data as nested dictionaries

        Returns:
            cleaned_uniprot_features : np.array
                Array of lists of GO IDs for each protein
        """
        # Loads data
        cleaned_uniprot_features = self.load_data(data=uniprot_data)

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

        # Save DataFrame as csv file after cleaning
        cleaned_uniprot_features.to_csv(save_path, index=False)

        # Returns a DataFrame of lists of GO strings for each protein
        return cleaned_uniprot_features

    def ontology_split_go_terms(
        self,
        cleaned_uniprot_data="features_files/cleaned_uniprot.csv",
        save_path="features_files/ontology_split_features.csv",
    ):
        """
        Seperates GO IDs for each protein based on ontology
        biological_process (BP)
        molecular_function (MF)
        cellular_component (CC)

        Parameters
        df : DataFrame or str
            Cleaned dataframe or str file path of all GO IDs for each protein

        Returns
        split_features_df : DataFrame
            Split features DataFrame by ontology for each protein
        """
        # Loads data
        df = self.load_data(data=cleaned_uniprot_data)

        # Converts row of IDs into multiple rows
        df_expl = df.explode("go_terms").rename(columns={"go_terms": "go_id"})

        # Drop empty rows early
        # exploded = exploded.dropna(subset=["go_id"])

        # Gets all unique go_ids for mapping
        unique_go_ids = df_expl["go_id"].unique()
        unique_go_ids = unique_go_ids.tolist()

        go_to_ontology = {}

        # Query mygene for all unique GO IDs
        mg_results = self.mg.querymany(
            unique_go_ids, scopes="go", fields="go", species="human"
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
            df_expl.groupby(["gene_name", "ontology"])["go_id"]
            .apply(list)
            .unstack(fill_value=[])
            .reset_index()
        )

        # Merge features back to original dataframe and drop go_terms column
        split_features_df = df.merge(ontology_features, on="gene_name", how="left")

        # Fill missing with empty lists
        for col in ["BP", "MF", "CC"]:
            split_features_df[col] = split_features_df[col].apply(
                lambda x: x if isinstance(x, list) else []
            )

        # Drop go_terms column after splitting
        split_features_df = split_features_df.drop(columns=["go_terms"])

        split_features_df.to_csv(save_path, index=False)

        return split_features_df

    def add_protein_sequence(
        self,
        uniprot_data="features_files/uniprot_features.csv",
        save_path="features_files/sequence_df.csv",
    ):
        """
        Parses protein sequence data from Uniprot Data

        Parameters
        uniprot_data : DataFrame or str
            DataFrame or str file path of Uniprot Data to parse sequences from
        save_path : str
            File path to save protein sequence data

        Returns
        protein_df : DataFrame
            DataFrame of protein sequences
        """
        # Loads data
        df = self.load_data(data=uniprot_data)

        seq_col_idx = 5
        gene_col_idx = 11

        # Temporary names for easier processing
        df = df.rename(
            columns={
                df.columns[seq_col_idx]: "sequence",
                df.columns[gene_col_idx]: "gene_name",
            }
        )

        # Group by gene and keep the longest sequence (standard for canonical proteins)
        protein_df = (
            df.groupby("gene_name")["sequence"]
            .apply(
                lambda seqs: (
                    max([str(s) for s in seqs], key=len) if any(pd.notna(seqs)) else ""
                )
            )
            .reset_index()
        )

        # Align with your graph node ordering
        protein_df = protein_df.set_index("gene_name").reindex(self.protein_genes)

        # Fill missing genes and finalize
        protein_df["sequence"] = protein_df["sequence"].fillna("")
        protein_df = protein_df.reset_index().rename(
            columns={"sequence": "amino_acid_sequence"}
        )

        protein_df.to_csv(save_path, index=False)
        return protein_df

    def merge_features(
        self,
        df1="features_files/ontology_split_features.csv",
        df2="features_files/sequence_df.csv",
        save_path="feature_df.csv",
    ):
        """
        Merges Ontology and Sequence Dataframes into single Dataframe

        Parameters
            df1 : DataFrame or str
                DataFrame or str file path of ontology feature data
            df2 : DataFrame or str
                DataFrame or str file path of sequence feature data
            save_path : str
                File path to save features dataframe

        Returns
            feature_df : DataFrame
                DataFrame of protein features
        """
        df1 = self.load_data(data=df1)
        df2 = self.load_data(data=df2)

        feature_df = pd.merge(df1, df2, on="gene_name")
        feature_df.to_csv(save_path, index=False)
        return feature_df

    def get_esm_embedding(self, sequence):
        # Process sequence through the model
        inputs = self.tokenizer(
            sequence, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Average hidden states (sequence length, hidden_dim) to get a single vector
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def get_go_embedding(self, go_list_str):
        # Handle string or list input
        go_ids = eval(go_list_str) if isinstance(go_list_str, str) else go_list_str
        vectors = [
            self.n2v_model.wv[go_id] for go_id in go_ids if go_id in self.n2v_model.wv
        ]
        return np.mean(vectors, axis=0) if vectors else np.zeros(64)

    def node_features_encoder(
        self,
        features_df="features_files/feature_df.csv",
        save_path="features_files/node_features.pt",
    ):
        """
        Encodes DataFrame features and stores them as a torch tensor for use in GNN training

        Parameters
            features_df : DataFrame or str
                Dataframe or str file path of protein features [GO IDs(BP, CC, MF), AA Sequence]
            save_path : str
                File path to save torch tensor

        Returns
            x : torch.tensor
                Torch.tensor of encoded protein features for torch.geometric data object
        """
        # Load Data
        df = self.load_data(features_df)

        # Pre-allocate space to store node features
        num_nodes = len(df)
        feature_dim = 384  # 320 (ESM) + 64 (GO)
        node_features = np.zeros((num_nodes, feature_dim), dtype=np.float32)

        # Loop through df proteins
        for i, (_, row) in enumerate(df.iterrows()):
            # Encode Sequence Features
            seq_feat = self.get_esm_embedding(row["amino_acid_sequence"])

            # Encode GO ID Features
            bp_feat = self.get_go_embedding(row["BP"])
            cc_feat = self.get_go_embedding(row["CC"])
            mf_feat = self.get_go_embedding(row["MF"])

            # Concatenate encoded features into single array
            features = np.concatenate([seq_feat, bp_feat, cc_feat, mf_feat])

            # Add features to pre-allocated array
            node_features[i] = features

        # Convert to torch tensor
        x = torch.from_numpy(node_features)
        # Save torch tensor
        torch.save(x, save_path)

        return x
