# fingerprint_encoder.py

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np

class FingerprintEncoder:

    def __init__(self, radius=2, n_bits=2048):
        self.radius = radius
        self.n_bits = n_bits

    def encode(self, mol):
        # takes one mol object
        # returns one fingerprint

    def encode_batch(self, mol_list):
        # takes a list of mol objects
        # returns a list of fingerprints

    def to_numpy(self, fingerprint):
        # converts fingerprint to numpy array
