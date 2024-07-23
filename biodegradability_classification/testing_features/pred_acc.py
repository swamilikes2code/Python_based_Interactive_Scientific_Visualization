import numpy as np
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit import DataStructs
from rdkit import RDLogger

'''came from the original study's code,
uses their values to produce a prediction accuracy'''
def calc_acc(similarity):
    if similarity >= 0.9:
        accuracy = 0.886
    elif 0.8 <= similarity <= 0.9:
        accuracy = 0.827
    elif 0.7 <= similarity <= 0.8:
        accuracy = 0.862
    elif 0.6 <= similarity <= 0.7:
        accuracy = 0.800
    elif 0.5 <= similarity <= 0.6:
        accuracy = 0.732
    else:
        accuracy = '-'
    return accuracy

#from data_cleansing.py
def smiles_to_molecule(smiles): #function that converts SMILES to RDKit molecule object
    try:
        molecule = Chem.MolFromSmiles(smiles)
        return molecule
    except:
        return None  # Return None if unable to convert
    
similarity_list = []
accuracy_list = []

#now using morgan generator bc deprectation warnings
fpg = rdFingerprintGenerator.GetMorganGenerator(radius=1, fpSize=2048)

fp = fpg.GetFingerprint(smiles_to_molecule('C=C(C)C(=O)O'))
# print(fp.shape)

model_df = pd.read_csv('../data/option_1.csv')

# Apply the conversion function to the SMILES column
RDLogger.DisableLog('rdApp.*') #ignoring hydrogen warning
model_mol = model_df['SMILES'].apply(smiles_to_molecule)
model_fp = [fpg.GetFingerprint(mol) for mol in model_mol]

# print(type(model_fp[0]))
# print(model_fp.shape)
def store_acc(fp):
    similarities = DataStructs.BulkTanimotoSimilarity(fp, model_fp)
    print(similarities)
    similarities.sort()
    similarity = round(similarities[-1], 2)
    accuracy = calc_acc(similarity)
    similarity_list.append(similarity)
    accuracy_list.append(accuracy)
store_acc(fp)

print(similarity_list)
print(accuracy_list)