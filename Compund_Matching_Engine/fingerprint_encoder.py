# fingerprint_encoder.py


from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import numpy as np

class FingerprintEncoder:

    def __init__(self, radius=2, n_bits=2048):
        self.radius = radius
        self.n_bits = n_bits

    def encode(self, mol):
        # takes one mol object
        # returns one fingerprint
        if mol is None:
            return None
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)
        return generator.GetFingerprint(mol)

    def encode_batch(self, mol_list):
        # takes a list of mol objects
        # returns a list of fingerprints
        fingerprints = []
        for mol in mol_list:
            fp = self.encode(mol)
            fingerprints.append(fp)
        return fingerprints

    def to_numpy(self, fingerprint):
        # converts fingerprint to numpy array
        if fingerprint is None:
            return None
        array = np.zeros((self.n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array

# Test
mol = Chem.MolFromSmiles("CC(C)c1ccccc1")
encoder = FingerprintEncoder()
fp = encoder.encode(mol)
print(fp)
print(type(fp))
print(len(fp))
print(f"Bits ON: {fp.GetNumOnBits()}")

arr = encoder.to_numpy(fp)
print(f"Numpy array shape: {arr.shape}")
print(f"Numpy array type: {arr.dtype}")









## you receive a receive:** a molecule object and return a fingerprint — a row of 2048 zeros and ones mol     
#Test your code with this #
# Chem.MolFromSmiles("CC(C)c1ccccc1")
#encoder = FingerprintEncoder()
#fp      = encoder.encode(mol)
#print(fp)           # fingerprint object
#print(type(fp))     # should be rdkit fingerprint type
