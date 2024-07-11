import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, DataStructs, Descriptors, AllChem
# from rdkit.DataStructs.cDataStructs import ExplicitBitVect
import numpy as np
from math import nan
import pubchempy

# Read the original CSV file into a DataFrame
df_og = pd.read_csv("./original_dataset_all.csv")

df = df_og.loc[:, ['Substance Name', 'Smiles', 'Class']]
df.rename(columns={'Smiles':'SMILES'}, inplace=True)

# dropping data for dummy data purposes
df = df.drop(df[df['Class'] == 0].sample(frac = .99).index)
df = df.drop(df[df['Class'] == 1].sample(frac = .99).index)

print(df['Substance Name'])
print('-------------------------------------------------------')

# making additional molecular names
def smiles_to_name(smiles): #function that converts SMILES to IUPAC name
    try:
        compound = pubchempy.get_compounds(smiles, namespace='smiles')
        name = compound[0].iupac_name
        if name == None:
            return "\'N/A\'"
        else:
            return name
    except:
        return "\'N/A\'"  # Return 'N/A' if unable to convert
    
df['Substance Name'] = df.apply(lambda row: smiles_to_name(row['SMILES']) if pd.isna(row['Substance Name']) else row['Substance Name'], axis=1)

print(df['Substance Name'])
print('-------------------------------------------------------')

df.drop(df[df['Substance Name'] == "\'N/A\'"].index, inplace = True)

df.reset_index(drop=True, inplace=True)
df['Substance Name'] = [str(sub) for sub in df['Substance Name']]

print(df['Substance Name'])