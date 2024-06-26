import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool

df = pd.read_csv('../rdkit_test.csv') # Load dataset


features = df[['MolWt', 'NumValenceElectrons', 'NumRadicalElectrons', 'HeavyAtomMolWt','ExactMolWt']] #independent variables / features 
target = df['Class'] #target variable

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42) # split data into training and testing sets

train_sizes = np.linspace(0.1, 1.0, 15) # Defines array from 0.1 to 1.0 in 10 evenly spaced out increments. The result is 10 numbers are generated... 0.1, 0.2, 0.3 ...  to 1.0

train_scores = [] # allocate space on machine to hold the test and train results to later plot on the learning curve
test_scores = []

# Compute the learning curve
for size in train_sizes: #for each size in the array train_size 
    X_train_subset = X_train.sample(frac=size, random_state=42) #from X_train (which holds 80% of the rows from the original dataset), take frac % of the rows from there to get a smaller percentage. state = 42 for reproduciblity
    y_train_subset = y_train.loc[X_train_subset.index] #X_train_subset.index gives you the index of the points taken in the line above. Then it extracts the corresponding class results for the smaller subset taken above so we can calculate the accuracy
    
    model = KNeighborsClassifier(n_neighbors=1) # Train the model with new subsets
    model.fit(X_train_subset, y_train_subset)
    
    train_score = accuracy_score(y_train_subset, model.predict(X_train_subset)) # Evaluate on training and test sets
    test_score = accuracy_score(y_test, model.predict(X_test))
    
    train_scores.append(train_score) #add the scores to the arrays 
    test_scores.append(test_score)

train_scores = np.array(train_scores) # Convert lists to numpy arrays
test_scores = np.array(test_scores)

output_file('learning_curve_custom_split.html') # Create a Bokeh plot

# Create a ColumnDataSource
source = ColumnDataSource(data=dict(
    train_size=train_sizes,
    train_score=train_scores,
    test_score=test_scores
))

# Create a figure with y_range set to (0, 1)
p = figure(
    title='Learning Curve with Custom Split', 
    x_axis_label='Training Size (Fraction)', 
    y_axis_label='Accuracy',
    tools='pan, wheel_zoom, box_zoom, reset',
    x_range=(0, 1),
    y_range=(0, 1)  # Set y-axis range from 0 to 1
)

# Plot training scores
p.line('train_size', 'train_score', source=source, line_width=2, legend_label='Training Score', color='blue')
p.circle('train_size', 'train_score', source=source, size=8, color='blue')

# Plot testing scores
p.line('train_size', 'test_score', source=source, line_width=2, legend_label='Test Score', color='orange')
p.circle('train_size', 'test_score', source=source, size=8, color='orange')

# Add hover tool
hover = HoverTool()
hover.tooltips = [
    ("Training Size", "@train_size"),
    ("Training Score", "@train_score"),
    ("Test Score", "@test_score")
]
p.add_tools(hover)

# Show the legend
p.legend.location = 'bottom_right'

# Show the plot
show(p)
