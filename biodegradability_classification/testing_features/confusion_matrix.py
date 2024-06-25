import pandas as pd
from bokeh.plotting import figure
from bokeh.io import curdoc

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

df = pd.read_csv("../biodegrad.csv", low_memory=False)
df = df.iloc[:, 2:]
print(df)

X = df.drop(columns=['Class', 'Fingerprint List'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0)

cmatrix = figure(title = "Confusion Matrix", x_range = (-1,1), y_range = (-1,1))

#squares 1234 = top left, top right, bottom left, bottom right
green_sq_x = [-.5, .5]
green_sq_y = [.5, -.5]
green_sq = cmatrix.scatter(x=green_sq_x, y=green_sq_y, marker='square', size=275, fill_color = 'green')
red_sq_x = [-.5, .5]
red_sq_y = [-.5, .5]
red_sq = cmatrix.scatter(x=red_sq_x, y=red_sq_y, marker='square', size = 270, fill_color = 'red')
curdoc().add_root(cmatrix)