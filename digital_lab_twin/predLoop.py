#needed for this to work: working directory should have a models folder with mmscalerX.pkl and mmscalerY.pkl and model.pt, and an outputs folder
# imports, put these at the very top of everything
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import joblib
import pathlib as path
import os
# initialize everything 
# note that these load funcs will need you to change to your current directory here!
model = torch.load('models\model.pt')
model.eval()
#scalers
mmscalerX = joblib.load('models\mmscalerX.pkl')
mmscalerY = joblib.load('models\mmscalerY.pkl')

XTimes = np.linspace(0, 150, 200)
XDF = pd.DataFrame(XTimes, columns=['Time'])
#generate empty columns
XDF['C_X'] = np.zeros(200)
XDF['C_N'] = np.zeros(200)
XDF['C_L'] = np.zeros(200)
XDF['F_in'] = np.zeros(200)
XDF['C_N_in'] = np.zeros(200)
XDF['I0'] = np.zeros(200)
XTimes = XDF.pop('Time') #popped for plotting
#set initial conditions by changing these vals
C_X_init = .2
C_N_init = .2
C_L_init = 0.0
F_in_init = 0.001
C_N_in_init = 10
I0_init = 150

#function takes in initial conditions and runs the model
#overwrites XDF with the predicted values
#updates bokeh plot with new values
#call when run button is hit
def predLoop(C_X, C_N, C_L, F_in, C_N_in, I0):
    #write init conditions to df
    #Only write to the first row for these 3, they'll be filled in thru the loop
    XDF['C_X'][0] = C_X
    XDF['C_N'][0] = C_N
    XDF['C_L'][0] = C_L
    #write to all rows for these 3, they won't be filled in thru the loop
    XDF['F_in'] = F_in
    XDF['C_N_in'] = C_N_in
    XDF['I0'] = I0

    #loop through the experiment and predict each timestep
    for i in range(0, 199):
        #get the current timestep
        X_current = XDF.iloc[i]
        #scale the current timestep
        X_current_scaled = mmscalerX.transform([X_current])
        #predict the next timestep
        Y_current_scaled = model(torch.tensor(X_current_scaled, dtype=torch.float32))
        #unscale the prediction
        Y_current_scaled = Y_current_scaled.detach().numpy()
        Y_current = mmscalerY.inverse_transform(Y_current_scaled)
        #store the prediction
        nextTimeStep = i+1
        XDF.iloc[nextTimeStep, 0] = Y_current[0,0]
        XDF.iloc[nextTimeStep, 1] = Y_current[0,1]
        XDF.iloc[nextTimeStep, 2] = Y_current[0,2]
    #after this loop, XDF should be filled with the predicted values
    #export XDF to csv
    #add times back in
    XDF['Time'] = XTimes
    XDF.to_csv('outputs\prediction.csv', index=False)
    #TODO: re-call the plotting function to show results to user

#testing with default values
predLoop(C_X_init, C_N_init, C_L_init, F_in_init, C_N_in_init, I0_init)
