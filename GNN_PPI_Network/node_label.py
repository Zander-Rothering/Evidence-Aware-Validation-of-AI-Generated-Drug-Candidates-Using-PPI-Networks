import pandas as pd

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