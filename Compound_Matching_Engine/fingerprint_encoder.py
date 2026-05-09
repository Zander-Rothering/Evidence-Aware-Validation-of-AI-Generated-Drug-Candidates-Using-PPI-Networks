# fingerprint_encoder.py


from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import numpy as np

class FingerprintEncoder:

    def __init__(self, radius=2, n_bits=2048):
        self.radius = radius
        self.n_bits = n_bits
        # Single shared generator instance — MorganGenerator construction
        # is non-trivial, so we build it once per encoder.
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, fpSize=self.n_bits
        )

    def encode(self, mol):
        # takes one mol object
        # returns one fingerprint
        if mol is None:
            return None
        return self._generator.GetFingerprint(mol)

    def encode_batch(self, mol_list):
        # takes a list of mol objects
        # returns a list of fingerprints
        fingerprints = []
        for mol in mol_list:
            fp = self.encode(mol)
            fingerprints.append(fp)
        return fingerprints

    def to_numpy(self, fingerprint_or_mol):
        """Convert a fingerprint (or mol) to a 2048-bit uint8 numpy array.

        Accepts either an ExplicitBitVect or an RDKit Mol for caller convenience.
        """
        if fingerprint_or_mol is None:
            return None
        if isinstance(fingerprint_or_mol, Chem.Mol):
            fingerprint = self.encode(fingerprint_or_mol)
        else:
            fingerprint = fingerprint_or_mol
        if fingerprint is None:
            return None
        array = np.zeros((self.n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array


if __name__ == "__main__":
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
