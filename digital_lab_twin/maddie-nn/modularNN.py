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

#dataSplitter takes in the data and the split params and returns the split data
#XTrainTime, XValTime, and XTestTime are the time columns for each respective set
def dataSplitter(X, Y, trainSplit, valSplit, testSplit):
    #split the data into train, val, and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=testSplit)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=valSplit/(trainSplit+valSplit), random_state=42)
    #pop time column
    XTrainTime = X_train.pop('Time')
    XValTime = X_val.pop('Time')
    XTestTime = X_test.pop('Time')
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, XTrainTime, XValTime, XTestTime

#scaleData takes in the data and returns the scaled data
def scaleData(X_train, X_val, X_test, Y_train, Y_val, Y_test):
    #scale the data
    stScalerX = preprocessing.StandardScaler().fit(X_train)
    stScalerY = preprocessing.StandardScaler().fit(Y_train)
    X_train_scaled = stScalerX.transform(X_train)
    X_val_scaled = stScalerX.transform(X_val)
    X_test_scaled = stScalerX.transform(X_test)
    Y_train_scaled = stScalerY.transform(Y_train)
    Y_val_scaled = stScalerY.transform(Y_val)
    Y_test_scaled = stScalerY.transform(Y_test)
    return stScalerX, stScalerY, X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled

#tensors takes in the scaled data and returns the tensors
def tensors(X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled):
    #convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val_scaled, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test_scaled, dtype=torch.float32)
    return X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor

#modelCreator takes in the model params and returns the model
def modelCreator(initNeuronNum, loss, optimizer, learnRate):
    #create the model
    model = nn.Sequential(
        nn.Linear(6, initNeuronNum), 
        nn.LeakyReLU(), 
        nn.Linear(initNeuronNum, (initNeuronNum/2)),
        nn.LeakyReLU(),
        nn.Linear((initNeuronNum/2), 3))
    #define the loss function
    if loss == 0:
        lossFunction = nn.MSELoss()
    elif loss == 1:
        lossFunction = nn.L1Loss()
    #define the optimizer
    if optimizer == 0:
        optimizer = optim.Adam(model.parameters(), lr=learnRate)
    elif optimizer == 1:
        optimizer = optim.SGD(model.parameters(), lr=learnRate)
    return model, lossFunction, optimizer

#trainModel takes in the model, loss function, optimizer, training params, and data and returns the trained model, training loss, and validation loss
def trainModel(model, lossFunction, optimizer, epochs, batchSize, X_train_tensor, X_val_tensor, Y_train_tensor, Y_val_tensor):
    batch_start = torch.arange(0, len(X_train_tensor), batchSize)
    
    # Hold the best model
    best_mse = np.inf   # init to infinity
    best_weights = None
    trainLoss = []
    valLoss = []
    
    # training loop
    for epoch in range(epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train_tensor[start:start+batchSize]
                y_batch = Y_train_tensor[start:start+batchSize]
                # forward pass
                y_pred = model(X_batch)
                loss = lossFunction(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val_tensor)
        mse = lossFunction(y_pred, Y_val_tensor)
        mse = float(mse)
        trainLoss.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())
        
        #validation loss
        y_pred = model(X_val_tensor)
        mse = lossFunction(y_pred, Y_val_tensor)
        mse = float(mse)
        valLoss.append(mse)
    
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return model, trainLoss, valLoss

#testModel takes in the model and data and returns the test loss
def testModel(model, lossFunction, X_test_tensor, Y_test_tensor):
    #test the model
    model.eval()
    y_pred = model(X_test_tensor)
    mse = lossFunction(y_pred, Y_test_tensor)
    mse = float(mse)
    return mse

#plotter takes in the training and validation loss and plots them
def plotter(trainLoss, valLoss):
    #plot the loss
    plt.plot(trainLoss, label='Training Loss')
    plt.plot(valLoss, label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()