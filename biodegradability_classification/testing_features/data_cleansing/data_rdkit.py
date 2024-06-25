import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, DataStructs, Descriptors
# from rdkit.DataStructs.cDataStructs import ExplicitBitVect
import numpy as np

# Read the original CSV file into a DataFrame
df = pd.read_csv("./original_dataset_all.csv")

# Drop the specified columns
df = df.loc[:, ['Substance Name', 'Smiles', 'Class']]
# print(df)

# dropping data from the non-biodegradable class to balance the data
df = df.drop(df[df['Class'] == 0].sample(frac = .46).index)
df.reset_index(drop=True, inplace=True)
# print(df)

# making molecules
def smiles_to_molecule(smiles): #function that converts Smiles to RDKit molecule object
    try:
        molecule = Chem.MolFromSmiles(smiles)
        return molecule
    except:
        return None  # Return None if unable to convert

# Apply the conversion function to the SMILES column
RDLogger.DisableLog('rdApp.*') #ignoring hydrogen warning
molecs = df['Smiles'].apply(smiles_to_molecule)
# print(df)

# generating descriptors (using all descriptors, can specify list to generate)
desc = [Descriptors.CalcMolDescriptors(mol) for mol in molecs]
desc_df = pd.DataFrame(desc)
# print(desc_df)

df = pd.concat([df, desc_df], axis = 1)
# print(df)

fp_obj = [MACCSkeys.GenMACCSKeys(mol) for mol in molecs]

# a list of fingerprint arrays, one array per molec
fp_vec = [(np.array(fp)) for fp in fp_obj]

# add the lists for table display purposes
df['Fingerprint List'] = fp_vec

# an array from fp_vec --> a df of fp_array --> to concat to main df
fp_array = np.stack(fp_vec, axis = 0)
fp_df = pd.DataFrame(fp_array)

# df_new holds each bit in separate column to be read by the model later
df = pd.concat([df, pd.DataFrame(fp_array)], axis = 1)

# # checking
# # df_check = df_new.sort_values(by = ['Class'])
# # df_check.hist(column = ['Class'])
# print(df_new['Class'].value_counts())

# # for testing
# # print(type(fp_vec))
# # print(fp_vec[0])
# # print(type(fp_array))
# # print(fp_array.shape)
# # print(fp_array[:5])
# # print(fp_df[:5])
# # print(df_new.shape)
# # print(df_cleaned.shape)

# Save the cleaned DataFrame to a new CSV file
df.to_csv("../../rdkit_table.csv", index=False)