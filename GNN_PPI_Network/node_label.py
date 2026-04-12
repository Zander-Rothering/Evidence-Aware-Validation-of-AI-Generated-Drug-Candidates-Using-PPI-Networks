import pandas as pd
import torch
class node_labeler():
    def __init__(self, protein_gene_file= "proteins.txt"):
        """
        Initializes list of proteins by their gene name
        """
        # Converts text file of proteins gene names to list
        with open(protein_gene_file, "r") as f:
            protein_genes = [line.strip() for line in f if line.strip()] 
        
        self.protein_genes = protein_genes

    def get_disgenet(self, input_path= 'disgenet_features.csv'):
        
        disgenet_df = pd.read_csv(input_path)

        disgenet_df['score'] = pd.to_numeric(disgenet_df['score'], errors='coerce').fillna(0)
        
        return disgenet_df

    def clean_disgenet(self):

        disgenet_df = self.get_disgenet()

        disgenet_scores_df = disgenet_df.groupby('geneSymbol', as_index=False)['score'].max()

        return disgenet_scores_df

    def node_labels(self, disease_score_threshold=0.3):
        """
        Creates label for GNN training.
        1 = Strong disease association (High Risk)
        0 = Weak or no association
        """
        proteins_df = self.clean_disgenet()

        # Create label for proteins based on disease score
        proteins_df['label'] = (proteins_df['score'] >= disease_score_threshold).astype(int)

        # Map gene symbol to label for aligning to protein order
        gene_to_label = dict(zip(proteins_df['geneSymbol'], proteins_df['label']))

        # Ensures labels are aligned with protein order
        labels = [gene_to_label.get(gene, 0) for gene in self.protein_genes]

        # Convert labels to torch tensor
        y_tensor = torch.tensor(labels, dtype=torch.long)

        return y_tensor
