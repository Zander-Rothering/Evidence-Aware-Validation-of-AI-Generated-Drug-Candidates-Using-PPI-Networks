import requests
import pandas as pd
from ppi_graph_builder import PPIGraphBuilder

def chunk_list(lst, size=100):
    """Yield successive chunks from a list."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def get_uniprot_chunk(proteins):
    # Query UniProt for functional data and cross-references
    url = "https://rest.uniprot.org/uniprotkb/search"

    gene_query = " OR ".join([f'gene_exact:"{p}"' for p in proteins])
    query = f'({gene_query}) AND organism_id:9606'

    payload = {
        'query': query,
        'fields': 'accession,gene_names,lineage,cc_subcellular_location,xref_kegg,xref_reactome',
        'format': 'json'
    }

    uniprot_response = requests.get(url, params=payload)

    if uniprot_response.status_code != 200:
        return None

    protein_data = uniprot_response.json()
    protein_results = protein_data.get("results", [])
    protein_df = pd.json_normalize(protein_results)

    return protein_df

def get_uniprot_data(proteins, chunk_size=100):
    proteins = list(set(proteins))  # deduplicate

    all_dfs = []
    failed_genes = []

    for i, chunk in enumerate(chunk_list(proteins, chunk_size)):
        print(f"Chunk {i+1} ({len(chunk)} genes)")

        # Batch query first
        gene_query = " OR ".join([f'gene:"{p}"' for p in chunk])
        query = f"({gene_query}) AND organism_id:9606"

        df_chunk = get_uniprot_chunk(query)

        if df_chunk is not None:
            all_dfs.append(df_chunk)
            continue

        # If batch fails → fallback to per-gene queries
        print("Chunk failed, falling back to per-gene queries...")

        for gene in chunk:
            single_query = f'gene:"{gene}" AND organism_id:9606'
            df_gene = get_uniprot_chunk(single_query)

            if df_gene is None or df_gene.empty:
                failed_genes.append(gene)
            else:
                all_dfs.append(df_gene)

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
    else:
        full_df = pd.DataFrame()

    return full_df, failed_genes