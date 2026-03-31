import torch
from torch_geometric.data import Data
import stringdb
import pandas as pd

#stringdb.api.get_network(identifiers, species=9606, required_score=400, caller_identity='https://github.com/gpp-rnd/stringdb', add_nodes=0)[source]¶
#Get the ppi network for a list of string ids

#Parameters:	
#identifiers (list) – list of string ids
#species (int, optional) – species NCBI identifier
#required_score (int, optional) – score cutoff for edges, corresponds to probability of belonging to same kegg pathway (0 -1000) 400 is medium confidence
#caller_identity (str, optional) – personal identifier for string
#add_nodes (int, optional) – number of nodes to add to the network based on confidence
#Returns:	
#network edges

PPI_df = stringdb.get_network(identifiers = ["HMGCR"], species=9606, required_score = 400, caller_identity = "HMGCR", add_nodes=1000)

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

print(PPI_df.shape)
print(PPI_df.head())

# Extract all unique proteins in df
ProteinA = set(PPI_df['preferredName_A'])
ProteinB = set(PPI_df['preferredName_B'])
Proteins = ProteinA.union(ProteinB)

# Map proteins to create edges
Protein_mapping = {name: i for i, name in enumerate(Proteins)}

# Source proteins
src = [Protein_mapping[name] for name in PPI_df['preferredName_A']]

# Desitnation protein 
dst = [Protein_mapping[name] for name in PPI_df['preferredName_B']]

# Undirected torch tensor of edges
edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

# Normalize edge weights to between 0-1 
scores = PPI_df['score'].values/1000.0

# Torch tensor of scores to be used as edge attributes
edge_weights = torch.tensor(list(scores)*2, dtype=torch.float)

# Create torch geomtric object to use as graph
# x = node features (Protein features)
# edge_index = interactions between proteins
# edge_attr = interaction score normalized
data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights)