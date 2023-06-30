# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import joblib
import pathlib as path
import copy

### Static
rawData = pd.read_csv('STEMVisualsSynthData.csv', header=0)
#remove unneeded column
rawData.drop('Index_within_Experiment', axis = 1, inplace = True)
#X is inputs--the three Concentrations, F_in, I0 (light intensity), and c_N_in (6)
X = rawData[['Time', 'C_X', 'C_N', 'C_L', 'F_in', 'C_N_in', 'I0']]
Y = X.copy(deep=True)
#drop unnecessary rows in Y
Y.drop('F_in', axis = 1, inplace = True)
Y.drop('C_N_in', axis = 1, inplace = True)
Y.drop('I0', axis = 1, inplace = True)
Y.drop('Time', axis = 1, inplace = True)
#Y vals should be X concentrations one timestep ahead, so remove the first index
Y.drop(index=0, inplace=True)
#To keep the two consistent, remove the last index of X
X.drop(index=19999, inplace=True)

#user defined parameters: current values can serve as a default
#splits - expects 3 floats that add to 1
trainSplit = 0.6
valSplit = 0.1
testSplit = 0.3
#model params
initNeuronNum = 50 #number of neurons in the first layer, 0 < int < 100
loss = 0 #0 = MSE, 1 = MAE
optimizer = 0 #0 = Adam, 1 = SGD
learnRate = 0.001 #0 < float < 0.1
#training params
epochs = 100 #0 < int < 1000
batchSize = 100 #0 < int < 1000

