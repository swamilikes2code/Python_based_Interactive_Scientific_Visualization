import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, DataStructs, Descriptors, AllChem
# from rdkit.DataStructs.cDataStructs import ExplicitBitVect
import numpy as np

# Read the original CSV file into a DataFrame
df = pd.read_csv("./original_dataset_all.csv")

# only keep three identifying cols
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
desc_df = desc_df.drop(columns=['MaxPartialCharge', 'MaxAbsPartialCharge', 'Ipc', 'MinPartialCharge', 'MinAbsPartialCharge', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW'])
# print(desc_df.isnull().any().any())
# print(desc_df)

option_1 = pd.concat([df, desc_df], axis = 1)
# print(df)

# from 2016 rdkit ugm github
def molecule_to_morgan(mol):
    a = np.zeros(2048)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, radius=1),a)
    return a

def molecule_to_ecfp(mol):
    a = np.zeros(2048)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2), a)
    return a

def molecule_to_pathfp(mol):
    a = np.zeros(2048)
    DataStructs.ConvertToNumpyArray(Chem.RDKFingerprint(mol, maxPath=2), a)
    return a

mfp_vec = [molecule_to_morgan(mol) for mol in molecs]
ECFP_vec = [molecule_to_ecfp(mol) for mol in molecs]
pathfp_vec = [molecule_to_pathfp(mol) for mol in molecs]
# MACCSfp_obj = [MACCSkeys.GenMACCSKeys(mol) for mol in molecs]

# a list of fingerprint arrays, one array per molec >> mfp and ecfp are already np arrays
# MACCSfp_vec = [(np.array(fp)) for fp in MACCSfp_obj]


# # add the lists for table display purposes
# df['Fingerprint List'] = fp_vec

# an array from fp_vec --> a df of fp_array --> to concat to main df
# MACCSfp_array = np.stack(MACCSfp_vec, axis=0)
# MACCSfp_df = pd.DataFrame(MACCSfp_array)

mfp_array = np.stack(mfp_vec, axis=0)
mfp_df = pd.DataFrame(mfp_array)
option_2 = pd.concat([df, mfp_df], axis=1)

ECFP_array = np.stack(ECFP_vec, axis=0)
ECFP_df = pd.DataFrame(ECFP_array)
option_3 = pd.concat([df, ECFP_df], axis=1)

pathfp_array = np.stack(pathfp_vec, axis=0)
pathfp_df = pd.DataFrame(pathfp_array)
option_4 = pd.concat([df, pathfp_df], axis=1)

# # df_new holds each bit in separate column to be read by the model later
# df = pd.concat([df, pd.DataFrame(fp_array)], axis = 1)

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
option_1.to_csv("../../data/option_1.csv", index=False)
option_2.to_csv("../../data/option_2.csv", index=False)
option_3.to_csv("../../data/option_3.csv", index=False)
option_4.to_csv("../../data/option_4.csv", index=False)