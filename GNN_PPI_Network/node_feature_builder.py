import requests
import pandas as pd
import numpy as np
from ppi_graph_builder import PPIGraphBuilder

def get_uniprot(proteins):
    url = "https://rest.uniprot.org/uniprotkb/search"

    uniprot_features = [] 

    for p in proteins:
        query = f'gene_exact:"{p}" AND organism_id:9606'
        payload = {
            'query': query,
            'fields': 'accession,gene_names,id,ft_domain,cc_function,cc_subcellular_location,go,xref_kegg,xref_reactome',
            'format': 'json'
        }

        uniprot_response = requests.get(url, params=payload)
        uniprot_response.raise_for_status()

        uniprot_data = uniprot_response.json()
        uniprot_results = uniprot_data.get("results", []) 
        
        uniprot_df = pd.json_normalize(uniprot_results)
        uniprot_features.append(uniprot_df)

    return pd.concat(uniprot_features, ignore_index=True)

def get_disgenet(entrez_ids, api_key):
    base_url = "https://api.disgenet.com/api/v1"
    headers = {"Authorization": f"Bearer {api_key}"}
    disgenet_results = []

    for id in entrez_ids:

        DisGeNET_response = requests.get(f"{base_url}/gda/gene/{int(id)}", headers=headers)
        
        DisGeNET_response.raise_for_status()

        DisGeNET_data = DisGeNET_response.json()
        DisGeNET_data['entrez_id'] = id
        disgenet_results.append(DisGeNET_data)
            
    return pd.concat(disgenet_results, ignore_index=True)

def get_biological_features(protein_list, disgenet_api_key):
    uniprot_df = get_uniprot(protein_list)
    
    entrez_ids = uniprot_df['entrez_id'].dropna().unique()

    disgenet_df = get_disgenet(entrez_ids, disgenet_api_key)

    uniprot_df['entrez_id'] = uniprot_df['entrez_id'].astype(str)
    disgenet_df['entrez_id'] = disgenet_df['entrez_id'].astype(str)
        
    protein_features = pd.merge(uniprot_df, disgenet_df, on='entrez_id', how='left')

    return protein_features

GraphBuilder = PPIGraphBuilder()
PPI__STRING_DF = GraphBuilder.get_stringdb_network()

PPI_BIOGRID_DF = GraphBuilder.get_biogrid_network("BIOGRID-ALL-5.0.256.tab3.txt")

#merge string and biogrid
PPI_COMBINED_DF = GraphBuilder.merge_networks(PPI__STRING_DF, PPI_BIOGRID_DF)
proteins, protein_mapping = GraphBuilder.protein_extraction(PPI_COMBINED_DF)

proteins = list(proteins)
proteins_features = get_uniprot(proteins[0:7])
print(proteins_features.head())