import pandas as pd
import numpy as np
from bokeh.models import ColumnDataSource, DataTable, TableColumn, CheckboxButtonGroup, Button, Div, RangeSlider
from bokeh.io import curdoc, output_notebook
from bokeh.layouts import column, row
from bokeh.models.callbacks import CustomJS
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


output_notebook()

# Load data from the CSV file
file_path = r'biodegrad.csv'
df = pd.read_csv(file_path)

# Columns that should always be shown
mandatory_columns = ['Substance Name', 'Smiles', 'Class']

# Ensure mandatory columns exist in the dataframe (if not, create dummy columns)
for col in mandatory_columns:
    if col not in df.columns:
        df[col] = "N/A"

# Global saved list
global saved_list
saved_list = []

def create_table():
    # Limit the dataframe to the first 10 rows
    df_subset = df.head(10)

    df_dict = df_subset.to_dict("list")
    cols = list(df_dict.keys())

    # Separate mandatory and optional columns
    optional_columns = [col for col in cols if col not in mandatory_columns]

    # Create column datasource
    source = ColumnDataSource(data=df_subset)

    # Create figure
    columns = [TableColumn(field=col, title=col) for col in cols]
    figure = DataTable(source=source, columns=columns, width=1800)

    # Create widget excluding mandatory columns
    checkbox_button_group = CheckboxButtonGroup(labels=optional_columns, active=list(range(len(optional_columns))))

    # Create status message Div
    status_message = Div(text='Columns saved', styles={'color': 'green', 'font-size': '16px'})

    # Update columns to display
    def update_cols(display_columns):
        # Always include mandatory columns
        all_columns = mandatory_columns + display_columns
        figure.columns = [col for col in columns if col.title in all_columns]
        figure.width = np.size(all_columns) * 90

    def update(attr, old, new):
        cols_to_display = [checkbox_button_group.labels[i] for i in checkbox_button_group.active]
        update_cols(display_columns=cols_to_display)
        status_message.text = 'Columns not saved'
        status_message.styles = {'color': 'red', 'font-size': '16px'}

    checkbox_button_group.on_change('active', update)

    # Save columns to global saved list
    def save_columns():
        saved_columns = [checkbox_button_group.labels[i] for i in checkbox_button_group.active]
        saved_columns = sorted(saved_columns)
        saved_list.clear()
        saved_list.append(saved_columns)
        status_message.text = 'Columns saved'
        status_message.styles = {'color': 'green', 'font-size': '16px'}
        print('Columns saved')

    # Save button
    save_button = Button(label="Save Selected Columns", button_type="success")

    # Attach callback to the save button
    save_button.on_click(save_columns)

    curdoc().add_root(column(checkbox_button_group, figure, save_button, status_message))

create_table()



# --------------- DATA SPLIT---------------


# set features and target
X = df.loc[saved_list] #all pka, alpha, fingerprint (to be added)
y = df['Class']    

            
# initialize model
model = DecisionTreeClassifier()

# function to split data and train model
def split_and_train_model(train_percentage, val_percentage, test_percentage):
    # print("test percentage:", test_percentage)

    # splitting
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=100-train_percentage, random_state=1)
    test_split = test_percentage / (100-train_percentage)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_split, random_state=1)
    
    # train model
    model.fit(X_train, y_train)
    
    # calculating accuracy
    # FIXME: implement tuning and validation
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    return f"<div>Training split: {train_percentage}</div><div>Validation split: {val_percentage}</div><div>Testing split: {test_percentage}</div><div><b>Validation Accuracy:</b> {val_accuracy:.2f} | <b>Test Accuracy:</b> {test_accuracy:.2f}</div>"

'''
# attempted python function to limit the range slider
def callback(attrname, old, new):
    if tvt.value[0] <= 10:
        tvt.value = (10, tvt.value[1])
    elif tvt.value[1] >= 90:
        tvt.value = (tvt.value[1], 90)
    if tvt.value[1]-tvt.value[0] < 10:
        if tvt.value[0] <= 50:
            tvt.value = (tvt.value[0], tvt.value[0]+10)
        elif tvt.value[1] > 50:
            tvt.value = (tvt.value[1]-10, tvt.value[1])
'''
            
# function to update model and accuracy
def update_model_and_accuracy(attrname, old, new):
    train_percentage, temp = tvt.value #first segment is train, range of the slider is validate, the rest is test
    val_percentage = temp - train_percentage
    test_percentage = 100 - temp
    if train_percentage < 10 or val_percentage < 10 or test_percentage < 10:
        return
    # print("train_percentage and temp:", train_percentage, temp, "val and test:", val_percentage, test_percentage)
    accuracy_text = split_and_train_model(train_percentage, val_percentage, test_percentage)
    accuracy_display.text = accuracy_text

# js function to limit the range slider
callback = CustomJS(args = dict(), 
             code = """
                if (cb_obj.value[0] <= 10) {
                    cb_obj.value = [10, cb_obj.value[1]];
                }
                else if (cb_obj.value[1] >= 90) {
                    cb_obj.value = [cb_obj.value[0], 90];
                }
                if (cb_obj.value[1] - cb_obj.value[0] < 10) {
                    if (cb_obj.value[0] <= 50){
                        cb_obj.value = [cb_obj.value[0], cb_obj.value[0]+10]
                    }
                    else if (cb_obj.value[1] > 50) {
                        cb_obj.value = [cb_obj.value[1]-10, cb_obj.value[1]];
                    }
                }
                """,
            )

# creating widgets
tvt = RangeSlider(title="Train-Validate-Test (%)", value=(50, 75), start=0, end=100, step=5, tooltips = False, show_value = False)
tvt.bar_color = '#FAFAFA' # may change later, just so that the segments of the bar look the same
accuracy_display = Div(text="<div>Validation Accuracy: N/A | Test Accuracy: N/A</div>")

# adding update functions to slider
tvt.js_on_change('value', callback)
tvt.on_change('value', update_model_and_accuracy)


# initializing model training and accuracy display
accuracy_display.text = split_and_train_model(50, 25, 25)

# add and display
layout = column(tvt, accuracy_display)
curdoc().add_root(layout)