#imports
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
import modularNN as mnn

#the below code is designed to drag and drop into the bokeh visualization
#static section should run once on launch, dynamic section should run on each change
### Static (run once)
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
#set device
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

#user defined parameters: current values can serve as a default
#splits - expects 3 floats that add to 1
trainSplit = 0.6
#model params
initNeuronNum = 18 #number of neurons in the first layer, 7 < int < 100
loss = 1 #0 = MSE, 1 = MAE
optimizer = 0 #0 = Adam, 1 = SGD
learnRate = 0.001 #0.0001 < float < 0.01
#training params
epochs = 100 #0 < int < 200
batchSize = 25 #0 < int < 200

### Dynamic (run on each change)
#TODO: upon running, check params are valid then update these values
#test the all-in-one function
model, Y_test_tensor, testPreds, XTestTime = mnn.trainAndSaveModel(X, Y, trainSplit, initNeuronNum, loss, optimizer, learnRate, epochs, batchSize, device)
#read in the loss CSV
lossCSV = pd.read_csv('models/losses.csv', header=0)
#TODO:plot the losses against epochs (stored as indexes)
#TODO:update the prediction side of the bokeh visualization