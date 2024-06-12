from bokeh.io import curdoc
from bokeh.models import CustomJS, Select, Button, Div
from bokeh.layouts import column
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import LinearSVC
# from sklearn.neighbors import KNeighborsClassifier


# creating dropdown menu

global my_alg

# default algorithm
my_alg = 'DecisionTree'

# Select button
select = Select(title="ML Algorithm:", value="DecisionTree", options=["DecisionTree", "NearestNeighbor", "SVM"])

# Create status message Div
status_message = Div(text='Not running', styles={'color': 'red', 'font-size': '16px'})

def update_algorithm(attr, old, new):
    global my_alg
    my_alg = new
    status_message.text = 'Not running'
    status_message.styles = {'color': 'red', 'font-size': '16px'}

# Attach callback to the Select widget
select.on_change('value', update_algorithm)

def run_config():
    status_message.text = f'Running {my_alg}'
    status_message.styles = {'color': 'green', 'font-size': '16px'}

# Run button
run_button = Button(label="Run chosen ML algorithm", button_type="success")

# Attach callback to the run button
run_button.on_click(run_config)

layout = column(select, run_button, status_message)


curdoc().add_root(layout)