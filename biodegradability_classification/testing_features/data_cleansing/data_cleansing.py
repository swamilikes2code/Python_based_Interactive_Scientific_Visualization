import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
import numpy as np

# Read the original CSV file into a DataFrame
df = pd.read_csv("./original_dataset_all.csv")

# Columns to be removed
columns_to_remove = ['Source', 'CAS Number', 'Name type', 'Index']

# Drop the specified columns
df_cleaned = df.drop(columns=columns_to_remove)

# Create fingerprints here
def smiles_to_molecule(smiles): #function that converts Smiles to RDKit molecule object
    try:
        molecule = Chem.MolFromSmiles(smiles)
        return molecule
    except:
        return None  # Return None if unable to convert

# Apply the conversion function to the SMILES column
RDLogger.DisableLog('rdApp.*') #ignoring hydrogen warning
df['Molecule'] = df['Smiles'].apply(smiles_to_molecule)
df_cleaned['Fingerprint'] = [MACCSkeys.GenMACCSKeys(mol) for mol in df['Molecule']]

'''
# creating list representations of the fingerprint vectors
temp = np.array([list(fp) for fp in df['Fingerprint']]).tolist()
df_cleaned['Fingerprint'] = np.array([list(fp) for fp in df['Fingerprint']])
'''

# Save the cleaned DataFrame to a new CSV file
df_cleaned.to_csv("../biodegrad.csv", index=False)