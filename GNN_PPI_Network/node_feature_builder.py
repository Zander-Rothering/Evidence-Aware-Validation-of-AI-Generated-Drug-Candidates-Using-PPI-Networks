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
        print(f"🔄 Chunk {i+1} ({len(chunk)} genes)")

        # ✅ Batch query first
        gene_query = " OR ".join([f'gene:"{p}"' for p in chunk])
        query = f"({gene_query}) AND organism_id:9606"

        df_chunk = get_uniprot_chunk(query)

        if df_chunk is not None:
            all_dfs.append(df_chunk)
            continue

        # ❌ If batch fails → fallback to per-gene queries
        print("⚠️ Chunk failed, falling back to per-gene queries...")

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

if __name__ == "__main__":
    builder = PPIGraphBuilder()

    # Step 1: Build PPI network
    PPI_DF = builder.get_stringdb_network()
    proteins, protein_mapping = builder.protein_extraction(PPI_DF)

    protein_df, failed = get_uniprot_data(proteins, chunk_size=50)

    print("✅ Data shape:", protein_df.shape)
    print("❌ Failed genes:", failed[:])
    print(f"Total failed: {len(failed)}")


def get_disgenet_data(gene_symbol):
    # DisGeNET API endpoint for Gene-Disease Associations
    url = f"https://disgenet.org{gene_symbol}"
    # This usually requires headers={'Authorization': 'Bearer YOUR_TOKEN'}
    
    try:
        response = requests.get(url)
        data = response.json()
        # Extract disease names
        return [item['disease_name'] for item in data]
    except:
        return []
    
def get_biological_features(protein_list):
    # Step A: Get UniProt (Pathways + Locations)
    all_data = get_uniprot_data(protein_list)
    
    # Step B: Loop through and add DisGeNET (Diseases)
    for gene in protein_list:
        if gene in all_data:
            diseases = get_disgenet_data(gene)
            all_data[gene]['diseases'] = diseases
        else:
            # Fallback for genes not found in UniProt search
            all_data[gene] = {'pathways': [], 'locations': [], 'diseases': []}
            
    return all_dat