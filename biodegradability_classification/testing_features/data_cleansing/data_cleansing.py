import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, DataStructs
# from rdkit.DataStructs.cDataStructs import ExplicitBitVect
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
df['MACCS Fingerprint Object'] = [MACCSkeys.GenMACCSKeys(mol) for mol in df['Molecule']]

# a list of fingerprint arrays, one array per molec
fp_vec = [(np.array(fp)) for fp in df['MACCS Fingerprint Object']]

# add the lists for table display purposes
df_cleaned['Fingerprint List'] = fp_vec

# an array from fp_vec --> a df of fp_array --> to concat to main df
fp_array = np.stack(fp_vec, axis = 0)
fp_df = pd.DataFrame(fp_array)

# df_new holds each bit in separate column to be read by the model later
df_new = pd.concat([df_cleaned, pd.DataFrame(fp_array)], axis = 1)

# dropping data from the non-biodegradable class to balance the data
df_new = df_new.drop(df_new[df_new['Class'] == 0].sample(frac = .46).index)

# checking
# df_check = df_new.sort_values(by = ['Class'])
# df_check.hist(column = ['Class'])
print(df_new['Class'].value_counts())

# for testing
# print(type(fp_vec))
# print(fp_vec[0])
# print(type(fp_array))
# print(fp_array.shape)
# print(fp_array[:5])
# print(fp_df[:5])
# print(df_new.shape)
# print(df_cleaned.shape)

# Save the cleaned DataFrame to a new CSV file
df_new.to_csv("../../biodegrad.csv", index=False)
