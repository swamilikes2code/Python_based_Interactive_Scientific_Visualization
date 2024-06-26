import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource

# Step 1: Load data from CSV
data = pd.read_csv('C:/Users/Devin/OneDrive - Lehigh University/Desktop/MountainTop/Python_based_Interactive_Scientific_Visualization/biodegradability_classification/data/option_1.csv')

# Assuming 'X' contains your features and 'y' contains your target
columns_to_drop = ['Class', 'Substance Name', 'Smiles']
X = data.drop(columns_to_drop, axis=1)
y = data['Class']

# Step 2: Train Decision Tree Classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

# Step 3: Calculate ROC Curve
y_scores = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Step 4: Plot ROC Curve using Bokeh
source = ColumnDataSource(data=dict(fpr=fpr, tpr=tpr))

plot = figure(title='Receiver Operating Characteristic (ROC) Curve',
              x_axis_label='False Positive Rate',
              y_axis_label='True Positive Rate')

plot.line('fpr', 'tpr', source=source, legend_label=f'AUC = {roc_auc:.2f}')
plot.line([0, 1], [0, 1], line_dash='dashed', color='gray')

plot.legend.location = 'bottom_right'
plot.legend.click_policy = 'hide'

# Show the plot
show(column(plot))
