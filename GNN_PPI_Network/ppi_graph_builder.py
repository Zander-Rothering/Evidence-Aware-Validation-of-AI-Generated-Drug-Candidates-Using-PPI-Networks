import torch
import stringdb
import pandas as pd

class PPIGraphBuilder:
    """
    
    """
    def __init__(self, species=9606, score_threshold=400, add_nodes=10000):
        self.species = species
        self.score_threshold = score_threshold
        self.add_nodes = add_nodes

    def get_stringdb_network(self, identifiers= ["HMGCR"]):
        PPI_df = stringdb.get_network(identifiers = identifiers, 
                                      species= self.species, 
                                      required_score = self.score_threshold, 
                                      caller_identity = "PPI_GNN", 
                                      add_nodes= self.add_nodes)
        
        return PPI_df
    
    def get_biogrid_network(self, biogrid_file_path, target_protein="HMGCR"):
        """Load BioGRID data and filter for HMGCR interactions."""
        df = pd.read_csv(biogrid_file_path, sep='\t', low_memory=False)
        
        # Filter to human proteins only (tax ID 9606)
        df = df[(df['Organism ID Interactor A'] == 9606) & 
                (df['Organism ID Interactor B'] == 9606)]
        
        # Filter to rows involving HMGCR
        mask = ((df['Official Symbol Interactor A'] == target_protein) | 
                (df['Official Symbol Interactor B'] == target_protein))
        df = df[mask]
        
        # Standardize column names to match STRING output
        biogrid_edges = pd.DataFrame({
            'preferredName_A': df['Official Symbol Interactor A'].values,
            'preferredName_B': df['Official Symbol Interactor B'].values,
            'score': 700  # treat all experimental as high confidence
        })
        
        return biogrid_edges
    
    def merge_networks(self, string_df, biogrid_df):
        """Combine STRING and BioGRID, drop duplicate edges."""
        combined = pd.concat([
            string_df[['preferredName_A', 'preferredName_B', 'score']], 
            biogrid_df
        ], ignore_index=True)
    
        combined = combined.drop_duplicates(
            subset=['preferredName_A', 'preferredName_B']
        )
    
        return combined
    
    def protein_extraction(self, df, save_path= "proteins.txt"):
        # Extract all unique proteins in df
        proteinA = set(df['preferredName_A'])
        proteinB = set(df['preferredName_B'])
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
        """Create edge_index tensor."""
        # Source proteins
        src = [mapping[name] for name in df['preferredName_A']]

        # Destination protein
        dst = [mapping[name] for name in df['preferredName_B']]

        # Undirected torch tensor of edges
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

        return edge_index

    def edge_weights(self, df):
        """Create edge weights to be stored as edge_attributes"""
        # Normalize edge weights to between 0-1
        scores = df['score'].values / 1000.0
        # Torch tensor of scores to be used as edge attributes
        edge_weights = torch.tensor(list(scores) * 2, dtype=torch.float)
        return edge_weights
"""
GraphBuilder = PPIGraphBuilder()
PPI__STRING_DF = GraphBuilder.get_stringdb_network()
#adding Biogrid part
PPI_BIOGRID_DF = GraphBuilder.get_biogrid_network("BIOGRID-ALL-5.0.256.tab3.txt")

#merge string and biogrid
PPI_COMBINED_DF = GraphBuilder.merge_networks(PPI__STRING_DF, PPI_BIOGRID_DF)
proteins, protein_mapping = GraphBuilder.protein_extraction(PPI_COMBINED_DF)
edge_index = GraphBuilder.build_edges(PPI_COMBINED_DF, protein_mapping)
edge_weights = GraphBuilder.edge_weights(PPI_COMBINED_DF)
print(edge_index[0:3][0:3])
print(edge_weights[0:3])
print(PPI__STRING_DF.shape)
print(PPI__STRING_DF.head())

#my sanity checks (Shivani)
print(f"STRING edges: {len(PPI__STRING_DF)}")
print(f"BioGRID edges: {len(PPI_BIOGRID_DF)}")
print(f"Combined edges after dedup: {len(PPI_COMBINED_DF)}")
print(f"Total unique proteins: {len(proteins)}")
"""
