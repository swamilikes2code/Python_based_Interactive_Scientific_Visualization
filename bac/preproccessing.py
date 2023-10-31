# welcome to the first step of the new module!
# tenatively naming it bac, "Bio-activity Classification", since that's the gist of the data and the principle applied
# here, just going to load in the data sets from the papers, separate out our smile strings, and make see about generating morgan fingerprints for each molecule

# import statements
import pandas as pd
import numpy as np
import os
import sys
import time
import pickle
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

# import the data, saved as xlsx files in the data folder
# for now, lets use the full CF dataset
cf_data = pd.read_excel('data/CFDataset.xlsx')

# for now, I just need the list of smile strings, it's the last column
smiles_list = cf_data.iloc[:, -1]
# rdkit doesn't get smiles, it wants mol objs, so we need to convert
mols_list = [Chem.MolFromSmiles(smile) for smile in smiles_list]
# note: on a test run, got a 'not removing hydrogen atom without neighbors'
# warning. come back if this causes problems later

#finally, we need to generate the morgan fingerprints for each molecule
# this is a list of lists, each list is a fingerprint
# the first argument is the molecule, the second is the radius of the fingerprint
# the third is the length of the bit vector
fp_list = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in mols_list]
# can save this list as a pickle for later use
pickle.dump(fp_list, open('data/CF_fp_list.pkl', 'wb'))