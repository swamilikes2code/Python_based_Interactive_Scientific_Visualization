import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

'''
investigating why precision recall curves are duplicating in our main file

ex 1
DT 2, DT 2, DT 7, KNN 20, DT 7, DT 2, DT 2
The first three decision trees produce the same pr list and curve 
'''

df = pd.read_csv('./../../data/option_1.csv')
np.random.seed(123)

X = df.iloc[:, 3:]
y = df['Class']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=.25)
test_split = 25/50
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_split)

model = DecisionTreeClassifier(max_depth=2)
# have to train first !!!!!
model.fit(X_train, y_train)

score1 = model.predict_proba(X_test)[:, 1]
print(score1)

model = DecisionTreeClassifier(max_depth = 7)
model.fit(X_train, y_train)

score2 = model.predict_proba(X_test)[:, 1]
print(score2)
