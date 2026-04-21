import torch
import stringdb
import pandas as pd


class PPIGraphBuilder:
    """
    Constructs protein-protein interaction (PPI) graphss using Biogrid and STRING DB based on a target protein.

    """

    def __init__(self, species=9606, score_threshold=400, add_nodes=10000):
        """
        Initialize PPIGraphBuilder

        Parameters:
            species : int
                Taxonomy ID for species (Default: 9606 for Homo Sapiens)
            score_threshold : int
                Minimum confidence for STRING DB protein interaction
            add_nodes : int
                Number of nodes to use for STRING DB network
        """
        self.species = species
        self.score_threshold = score_threshold
        self.add_nodes = add_nodes

    def get_stringdb_network(self, identifiers=["HMGCR"]):
        """
        Gets protein interaction network for target protein(s) from STRING DB

        Parameters:
            identifiers : list
                List of protein identifiers as gene symbol/IDs used for network seed

        Returns:
            PPI_df : DataFrame
                DataFrame containing interactions between proteins (edges)
                DataFrame Columns = ['preferredName_A', 'preferredName_B', 'score']
        """
        PPI_df = stringdb.get_network(
            identifiers=identifiers,
            species=self.species,
            required_score=self.score_threshold,
            caller_identity="PPI_GNN",
            add_nodes=self.add_nodes,
        )

        return PPI_df

    def get_biogrid_network(self, biogrid_file_path, target_protein="HMGCR"):
        """
        Loads BioGRID data and filter for HMGCR interactions.

        Parameters:
            biogrid_file_path : str
                Path to BioGRID dataset.
            target_protein : str
                Protein of interest (Default: "HMGCR").
        """
        df = pd.read_csv(biogrid_file_path, sep="\t", low_memory=False)

        # Filter to human proteins only (tax ID 9606)
        df = df[
            (df["Organism ID Interactor A"] == 9606)
            & (df["Organism ID Interactor B"] == 9606)
        ]

        # Filter to rows involving HMGCR
        mask = (df["Official Symbol Interactor A"] == target_protein) | (
            df["Official Symbol Interactor B"] == target_protein
        )
        df = df[mask]

        # Standardize column names to match STRING output
        biogrid_edges = pd.DataFrame(
            {
                "preferredName_A": df["Official Symbol Interactor A"].values,
                "preferredName_B": df["Official Symbol Interactor B"].values,
                "score": 700,  # treat all experimental as high confidence
            }
        )

        return biogrid_edges

    def merge_networks(self, string_df, biogrid_df):
        """
        Combines STRING and BioGRID, drop duplicate edges

        Parameters:
            string_df : DataFrame
                STRING DB interaction DataFrame.
            biogrid_df : DataFrame
                BioGRID interaction DataFrame.
        """
        # Concatenate STRING DB and BioGRID DataFrames
        combined = pd.concat(
            [string_df[["preferredName_A", "preferredName_B", "score"]], biogrid_df],
            ignore_index=True,
        )

        # Removes duplicate edges
        combined = combined.drop_duplicates(
            subset=["preferredName_A", "preferredName_B"]
        )

        return combined

    def protein_extraction(self, df, save_path="proteins.txt"):
        """
        Extracts unique proteins from DataFrame and creates index mapping

        Parameters:
            df : DataFrame
                DataFrame of protein interactions
            save_path : str
                File path to save proteins lsit to

        Returns:
            proteins : list
                List of all unique proteins
            protein_mapping : dict
                Mapping of each protein name to an index
        """

        # Extract all unique proteins in df
        proteinA = set(df["preferredName_A"])
        proteinB = set(df["preferredName_B"])
        proteins = proteinA.union(proteinB)

        # Sort proteins to ensure order matches original proteins file
        proteins = sorted(list(proteins))

        # Map proteins to create edges
        protein_mapping = {name: i for i, name in enumerate(proteins)}

        with open(save_path, "w") as f:
            for protein in sorted(proteins):
                f.write(protein + "\n")

        return proteins, protein_mapping

    def build_edges(self, df, mapping):
        """
        Creates edge_index tensor

        Parameters:
            df : DataFrame
                DataFrame of protein interactions
            mapping : dict
                Mapping of each protein name to an index

        Returns:
            edge_index : torch.tensor
                Edge index tensor for torch geometric data object
                Shape [2, num_edges * 2] with undirected edges
        """
        # Source proteins
        src = [mapping[name] for name in df["preferredName_A"]]

        # Destination protein
        dst = [mapping[name] for name in df["preferredName_B"]]

        # Undirected torch tensor of edges
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

        return edge_index

    def edge_weights(self, df):
        """
        Create edge weights to be stored as edge_attributes

        Parameters:
            df : DataFrame
                DataFrame of protein interactions

        Returns:
            edge_weights : torch.tensor
                Torch tensor of edgeweights for torch geometric data object
        """
        # Normalize edge weights to between 0-1
        scores = df["score"].values / 1000.0
        # Torch tensor of scores to be used as edge attributes
        edge_weights = torch.tensor(list(scores) * 2, dtype=torch.float)
        return edge_weights
