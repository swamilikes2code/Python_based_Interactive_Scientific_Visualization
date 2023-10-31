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

# import the data, saved as xlsx files in the data folder
# for now, lets use the full CF dataset
cf_data = pd.read_excel('data/CFDataset.xlsx')
# print out the first few rows to make sure it loaded correctly
print(cf_data.head())
