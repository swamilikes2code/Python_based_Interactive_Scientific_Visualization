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
from rdkit import RDLogger
# 
#suppress warnings 
RDLogger.DisableLog('rdApp.*')

# import the data, saved as xlsx files in the data folder
# swapping to the smaller dataset for faster testing
cf_data = pd.read_excel('data/original/C3F6Dataset.xlsx')

# for now, I just need the list of smile strings, it's the last column
smiles_list = cf_data.iloc[:, -1]
# rdkit doesn't get smiles, it wants mol objs, so we need to convert
mols_list = [Chem.MolFromSmiles(smile) for smile in smiles_list]
# note: on a test run, got a 'not removing hydrogen atom without neighbors'
# warning. come back if this causes problems later
#path fingerprints, depth 3-4 as an alternative? depth first

#offers a fingerprint that provides depth. 
#**ecfp, depth 1-2, breadth-first search 
# generate ecfp from mols_list
# collect a union of fragments from the dataset, extended fp creates our full set of possibilities
# can also do a fragment search on the fps, see both if it exists and how many times 
#finally, we need to generate the morgan fingerprints for each molecule
# this is a list of lists, each list is a fingerprint
# the first argument is the molecule, the second is the radius of the fingerprint
# the third is the length of the bit vector
fp_list = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in mols_list]
# can save this list as a pickle for later use
#pickle.dump(fp_list, open('data/CF_fp_list.pkl', 'wb'))
# write fplist to a vector
fp_vec = np.array(fp_list)
#display head of fp_vec
print(fp_vec[:5])

#create a column for random classifiers
rand_class = np.random.randint(0, 2, (len(fp_vec), 1))
#append to fp_vec
fp_vec = np.append(fp_vec, rand_class, axis=1)

#save as csv
np.savetxt('data/CF_fp_vec.csv', fp_vec, delimiter=',')
#save as pickle
pickle.dump(fp_vec, open('data/CF_fp_vec.pkl', 'wb'))
