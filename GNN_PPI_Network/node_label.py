import pandas as pd
import torch


class node_labeler:
    """
    Generates node labels for protein-protein interaction network (PPI) using
    disease association data from DisGeNET
    """

    def __init__(self, proteins=None, protein_gene_file="proteins.txt"):
        """
        Initializes list of proteins by their gene name

        Parameters:
            proteins : list
                List of unique proteins to get disease association
            protein_gene_file : str
                File path to text file containing unique proteins
        """
        if proteins == None:
            # Converts text file of proteins gene names to list
            with open(protein_gene_file, "r") as f:
                protein_genes = [line.strip() for line in f if line.strip()]
        else:
            protein_genes = proteins

        self.protein_genes = protein_genes

    def get_disgenet(self, input_path="disgenet_features.csv"):
        """
        Loads DisGeNET dataset containing gene (protein) disease association scores.

        Parameters:
            input_path : str
                File path to the DisGeNET dataset

        Returns:
            disgenet_df : DataFrame
                DisGeNET data with 'score' (disease association score) column
        """
        disgenet_df = pd.read_csv(input_path)

        # Replace invalid values with 0 and convert scores to a numeric value
        disgenet_df["score"] = pd.to_numeric(
            disgenet_df["score"], errors="coerce"
        ).fillna(0)

        return disgenet_df

    def clean_disgenet(self):
        """
        Cleans DisGeNET data to have single score for each gene

        Returns:
            disgenet_scores_df : DataFrame
                DataFrame containing only gene symbol and score
        """

        disgenet_df = self.get_disgenet()

        # Aggregate gene by the maximum score per gene
        disgenet_scores_df = disgenet_df.groupby("geneSymbol", as_index=False)[
            "score"
        ].max()

        return disgenet_scores_df

    def node_labels(self, disease_score_threshold=0.3):
        """
        Creates binary node labels for GNN training based on disease association

        1 = Strong disease association (High Risk)
        0 = Weak or no association

        Parameter:
            disease_score_threshold : float
                Threshold for classifying a gene as associated with a disease

        Returns:
            y_tensor : torch.tensor
                Torch tensor of binary node labels
        """
        proteins_df = self.clean_disgenet()

        # Create label for proteins based on disease score
        proteins_df["label"] = (proteins_df["score"] >= disease_score_threshold).astype(
            int
        )

        # Map gene symbol to label for aligning to protein order
        gene_to_label = dict(zip(proteins_df["geneSymbol"], proteins_df["label"]))

        # Ensures labels are aligned with protein order
        labels = [gene_to_label.get(gene, 0) for gene in self.protein_genes]

        # Convert labels to torch tensor
        y_tensor = torch.tensor(labels, dtype=torch.long)

        return y_tensor
