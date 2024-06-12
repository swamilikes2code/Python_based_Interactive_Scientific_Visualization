import pandas as pd
from bokeh.io import curdoc
from bokeh.models import Select, Button, Div, ColumnDataSource, Whisker
from bokeh.models.callbacks import CustomJS
from bokeh.layouts import column
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data in from temp_data.csv (this is just a copy of biodegrad.csv, but in the testing_features folder)
# this is just placeholder, will use data from phase_one.py
file_path = r'temp_data.csv'
global df
df = pd.read_csv(file_path)

global df_new
df_new = df


# creating dropdown menu
global my_alg

# default algorithm
my_alg = 'Decision Tree'

# Create status message Div
status_message = Div(text='Not running', styles={'color': 'red', 'font-size': '16px'})

# Select button
select = Select(title="ML Algorithm:", value="Decision Tree", options=["Decision Tree", "K-Nearest Neighbor", "Support Vector Classification"])


def update_algorithm(attr, old, new):
    global my_alg
    my_alg = new
    status_message.text = 'Not running'
    status_message.styles = {'color': 'red', 'font-size': '16px'}

# Attach callback to Select widget
select.on_change('value', update_algorithm)

# creating widgets
accuracy_display = Div(text="<div>Validation Accuracy: N/A | Test Accuracy: N/A</div>")
global val_accuracy
val_accuracy = []

global test_accuracy
test_accuracy = []

def run_config():
    status_message.text = f'Running {my_alg}'
    status_message.styles = {'color': 'green', 'font-size': '16px'}

    # Assigning model based on selected ML algorithm, using default hyperparameters
    if my_alg == "Decision Tree":
        model = DecisionTreeClassifier()
    elif my_alg == "K-Nearest Neighbor":
        model = KNeighborsClassifier()
    else:
        model = LinearSVC()



    # function to split data and train model
    def split_and_train_model(train_percentage, val_percentage, test_percentage):
        
        val_accuracy = []
        test_accuracy = []

        # run and shuffle five times, and save result in list
        for i in range(5):

            # Was not running properly with fingerprint
            X = df_new.drop(columns=['Substance Name', 'Smiles', 'Class', 'Fingerprint'])
            y = df_new['Class']

            # splitting
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=100-train_percentage)
            test_split = test_percentage / (100-train_percentage)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_split)
            
            # train model
            model.fit(X_train, y_train)
            
            # calculating accuracy
            # FIXME: implement tuning and validation

            y_val_pred = model.predict(X_val)
            val_accuracy.append(accuracy_score(y_val, y_val_pred))
            y_test_pred = model.predict(X_test)
            test_accuracy.append(accuracy_score(y_test, y_test_pred))

        return f"<div><b>Validation Accuracy:</b> {str(val_accuracy)} | <b>Test Accuracy:</b> {str(test_accuracy)}</div>"



    # initializing model training and accuracy display
    accuracy_display.text = split_and_train_model(50, 25, 25)

# Run button
run_button = Button(label="Run chosen ML algorithm", button_type="success")

# Attach callback to the run button
run_button.on_click(run_config)

# Create boxplot of prediction accuracy
# Note: this will likely not be in the run tab



layout = column(select, run_button, status_message, accuracy_display)

curdoc().add_root(layout)