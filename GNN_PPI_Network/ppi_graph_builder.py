import torch
import stringdb
import pandas as pd

#2stringdb.api.get_network(identifiers, species=9606, required_score=400, caller_identity='https://github.com/gpp-rnd/stringdb', add_nodes=0)[source]¶
#Get the ppi network for a list of string ids

#Parameters:	
#identifiers (list) – list of string ids
#species (int, optional) – species NCBI identifier
#required_score (int, optional) – score cutoff for edges, corresponds to probability of belonging to same kegg pathway (0 -1000) 400 is medium confidence
#caller_identity (str, optional) – personal identifier for string
#add_nodes (int, optional) – number of nodes to add to the network based on confidence
#Returns:	
#network edges

# Output (Dataframe) 
#stringId_A stringId_B preferredName_A preferredName_B  ncbiTaxonId  score  nscore  fscore  pscore  ascore  escore  dscore  tscore

#stringId_A: Internal STRING identifier for protein
#stringId_B: Internal STRING identifier for protein 
#preferredName_A: Common name of protein in interaction 
#preferredName_B: Common name of protein in interaction   
#ncbiTaxonId: Taxonomy identifier for the species (9606 for Humans) 
#score: Probabilistic measure of how likely the interaction is to be true  
#nscore: (Neighbor) Computed from the proximity of genes on the genome (inter-gene nucleotide count).  
#fscore: (Fusion) Derived from proteins that are fused into a single polypeptide chain in other species.
#pscore: (Co-occurrence) Derived from similar absence or presence patterns of genes across different species.
#ascore: (Co-expression) Based on similar patterns of mRNA expression (e.g., from microarrays or RNA-seq).
#escore: (Experimental) Derived from high-throughput lab data like affinity chromatography or yeast two-hybrid screens.
#dscore: (Database) Extracted from curated knowledge in other public databases (e.g., KEGG, Reactome).
#tscore: (Textmining) Derived from the statistical co-occurrence of protein names in scientific abstracts (PubMed).
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
        
    def protein_extraction(self, df):
        # Extract all unique proteins in df
        proteinA = set(df['preferredName_A'])
        proteinB = set(df['preferredName_B'])
        proteins = proteinA.union(proteinB)

        # Map proteins to create edges
        protein_mapping = {name: i for i, name in enumerate(proteins)}

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
    

PPIGraphBuilder = PPIGraphBuilder()
PPI_DF = PPIGraphBuilder.get_stringdb_network()
proteins, protein_mapping = PPIGraphBuilder.protein_extraction(PPI_DF)
edge_index = PPIGraphBuilder.build_edges(PPI_DF, protein_mapping)
edge_weights = PPIGraphBuilder.edge_weights(PPI_DF)
print(edge_index[0:3][0:3])
print(edge_weights[0:3])
print(PPI_DF.shape)
print(PPI_DF.head())