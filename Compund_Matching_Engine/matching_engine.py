from rdkit import Chem

def parse_smiles(smiles: str):
    # Step 1: Ensure that entry is not empty
    if not smiles or not smiles.strip():
        return None, "ERROR: empty input"
    
    # Step 2: pass to RDKit's parser
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    
    # Step 3: if parser returns nothing, stop
    if mol is None:
        return None, "ERROR: invalid SMILES string"
    
    # Step 4 and then 5: run sanitisation
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        return None, f"ERROR: sanitisation failed — {str(e)}"
    
    # Step 6: return molecule and OK status
    return mol, "OK"


# Test with Atorvastatin
if __name__ == "__main__":
    smiles = "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O"
    mol, status = parse_smiles(smiles)
    print(status)
    # Test 2 AI-generated molecule (from MolGPT paper, HMGCR-like)
    ai_generated = "O=C(O)CC(O)CC(O)CCn1c(C(C)C)c(-c2ccc(F)cc2)c(C(=O)Nc2ccccc2)c1-c1ccccc1"
    mol2, status2 = parse_smiles(ai_generated)
    print(f"AI-generated molecule: {status2}")

    # Test 3 deliberately broken SMILES to confirm error handling works
    broken = "CC(C)c1ccc(INVALID!!!)cc1"
    mol3, status3 = parse_smiles(broken)
    print(f"Broken SMILES: {status3}")
