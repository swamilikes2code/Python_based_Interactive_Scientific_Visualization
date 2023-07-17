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
import math

#dataSplitter takes in the data and the split params and returns the split data
#XTrainTime, XValTime, and XTestTime are the time columns for each respective set
def dataSplitter(X, Y, trainSplit):
    #presume trainsplit is between .1 and .7
    #val split and test split should be equal to the other, and half of the remaining percentage
    valSplit = testSplit = (1-trainSplit)/2
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
        nn.Linear(initNeuronNum, (initNeuronNum//2)),
        nn.LeakyReLU(),
        nn.Linear((initNeuronNum//2), 3))
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
                for param in model.parameters():
                    param.grad = None
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_train_tensor)
        mse = lossFunction(y_pred, Y_train_tensor)
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

#plotPredictions takes in the real data and labels as well as the model to predict the test data and plot the predictions
#quick visual just to see our data
def plot_predictions(train_data, 
                     train_labels, 
                     test_data, 
                     test_labels, 
                     predictions):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7)) #create the base of our figure, figsize feeds in width/height in inches
  #if needed, move data to cpu
  if torch.cuda.is_available():
        #train_data = train_data.cpu()
        train_labels = train_labels.cpu()
        #test_data = test_data.cpu()
        test_labels = test_labels.cpu()
        #predictions = predictions.cpu()
  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data") #c for color, s for size. 
  
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="deeppink", s=4, label="Testing data")

  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="seagreen", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14});

#saveModel saves both model and the scalers
def saveModel(model, scalerX, scalerY):
    torch.save(model, 'models/model.pt')
    joblib.dump(scalerX, 'models/StScalerX.pkl')
    joblib.dump(scalerY, 'models/StScalerY.pkl')

#savelosses saves the training and validation losses as csv files
def saveLosses(trainLoss, valLoss):
    #convert losses to a single DF
    losses = pd.DataFrame({'trainLoss': trainLoss, 'valLoss': valLoss})
    #save the losses
    return losses
    #losses.to_csv('models/losses.csv')


#all in one function to split data, train and save model
#models, scalers, and losses are saved to the models folder
def trainAndSaveModel(X, Y, trainSplit, initNeuronNum, loss, optimizer, learnRate, epochs, batchSize, device):
    #split the data
    X_train, X_val, X_test, Y_train, Y_val, Y_test, XTrainTime, XValTime, XTestTime = dataSplitter(X, Y, trainSplit)
    #scale the data
    stScalerX, stScalerY, X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled= scaleData(X_train, X_val, X_test, Y_train, Y_val, Y_test)
    #tensorize the data
    X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor = tensors(X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled)
    #if possible, move the tensors to the GPU
    X_train_tensor = X_train_tensor.to(device)
    X_val_tensor = X_val_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    Y_train_tensor = Y_train_tensor.to(device)
    Y_val_tensor = Y_val_tensor.to(device)
    Y_test_tensor = Y_test_tensor.to(device)

    #create the model
    model, lossFunction, optimizer = modelCreator(initNeuronNum, loss, optimizer, learnRate)
    #if possible, move the model to the GPU
    model = model.to(device)
    #train the model
    model, trainLoss, valLoss = trainModel(model, lossFunction, optimizer, epochs, batchSize, X_train_tensor, X_val_tensor, Y_train_tensor, Y_val_tensor)
    lossDF = saveLosses(trainLoss, valLoss)
    #saveModel(model, stScalerX, stScalerY)
    testPreds, mse, rmse = testPredictions(model, X_test_tensor, lossFunction, Y_test_tensor) #3 columns, 1 for each output (biomass/nitrate/lutein)
    #return model, Y_test_tensor, testPreds, XTestTime for plotting
    return model, Y_test_tensor, testPreds, XTestTime, lossDF, stScalerX, stScalerY, testPreds, mse, rmse
#testPredictions takes in the model, test data, loss function, and test labels and returns the predictions as well as test loss and RMSE
def testPredictions(model, X_test_tensor, lossFunction, Y_test_tensor):
    #test the model
    model.eval()
    y_pred = model(X_test_tensor)
    #if possible, move the predictions to the CPU
    if torch.cuda.is_available():
        y_pred = y_pred.cpu()
    #calculate test loss
    mse = lossFunction(y_pred, Y_test_tensor)
    mse = float(mse)
    #calculate test RMSE
    rmse = math.sqrt(mse)
    #create parity dataframe
    y_pred = y_pred.detach().numpy()
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = ['Biomass_Predicted', 'Nitrate_Predicted', 'Lutein_Predicted']
    #add the actual values to the dataframe
    y_pred['Biomass_Actual'] = Y_test_tensor[:,0]
    y_pred['Nitrate_Actual'] = Y_test_tensor[:,1]
    y_pred['Lutein_Actual'] = Y_test_tensor[:,2]

    return y_pred, mse, rmse


