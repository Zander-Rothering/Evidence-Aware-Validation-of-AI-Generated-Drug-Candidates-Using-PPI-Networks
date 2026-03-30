
# Loads all classes and libraries for Step 1
# Import this file and everything in Step 1 is available

# ── External Libraries ────────────────────────────────────────

# RDKit — all molecular chemistry operations
from rdkit import Chem                          # SMILES parsing
from rdkit.Chem import AllChem                  # Morgan fingerprints
from rdkit.Chem import Descriptors              # MW, LogP, TPSA
from rdkit.Chem import QED                      # drug-likeness score
from rdkit.Chem import rdFMCS                   # MCS scaffold comparison
from rdkit.Chem import DataStructs              # Tanimoto similarity
from rdkit.Chem import FilterCatalog            # PAINS / Brenk filters
from rdkit.Chem import FilterCatalogParams      # filter catalog builder
from rdkit.Chem.Scaffolds import MurckoScaffold # core scaffold extraction
from rdkit.Chem import Draw                     # 2D structure images
from rdkit.Chem import rdMolDescriptors         # HBD, HBA, rotatable bonds

# ChEMBL — reference compound database
from chembl_webresource_client.new_client import new_client  # ChEMBL API

# Standard Python libraries
import numpy as np        # fingerprint array operations
import pandas as pd       # compound data as dataframes
import requests           # HTTP fallback for API calls
import logging            # error and status logging
import json               # parsing API responses
import os                 # file path operations
