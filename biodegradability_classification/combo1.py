import pandas as pd
import numpy as np
from bokeh.models import ColumnDataSource, DataTable, TableColumn, CheckboxButtonGroup, Button, Div, RangeSlider, Select, Whisker
from bokeh.io import curdoc, output_notebook
from bokeh.layouts import column, row
from bokeh.models.callbacks import CustomJS
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
'''
This file is a draft for combining tab 1 (data) with tab 2(model selection and first run based on data config)
'''

#for ref:
# df is original csv, holds fingerprint list and 167 cols of fingerprint bits
# df_display > df_subset > df_dict are for displaying table

# --------------- DATA SELECTION ---------------
# Load data from the csv file
file_path = r'biodegrad.csv'
df = pd.read_csv(file_path)
df_display = df.iloc[:,:22] #don't need to display the other 167 rows of fingerprint bits
df = df.drop(columns=['Fingerprint List']) #removing the display column, won't be useful in training

# Columns that should always be shown
mandatory_columns = ['Substance Name', 'Smiles', 'Class']

# Ensure mandatory columns exist in the dataframe (if not, create dummy columns) (hopefully shouldn't have to apply)
for col in mandatory_columns:
    if col not in df_display.columns:
        df_display[col] = "N/A"

# saved list to append to
saved_col_list = []

# Limit the dataframe to the first 10 rows
df_subset = df_display.head(10)

df_dict = df_subset.to_dict("list")
cols = list(df_dict.keys())

# Separate mandatory and optional columns
optional_columns = [col for col in cols if col not in mandatory_columns]

# Create column datasource
source = ColumnDataSource(data=df_subset)

# Create figure
columns = [TableColumn(field=col, title=col) for col in cols]
datatable = DataTable(source=source, columns=columns, width=1800)

# Create widget excluding mandatory columns
checkbox_button_group = CheckboxButtonGroup(labels=optional_columns, active=list(range(len(optional_columns))))

# Create status message Div
data_save_message = Div(text='Configuration saved', styles={'color': 'green', 'font-size': '16px'})

# Update columns to display
def update_cols(display_columns):
    # Always include mandatory columns
    all_columns = mandatory_columns + display_columns
    datatable.columns = [col for col in columns if col.title in all_columns]
    datatable.width = np.size(all_columns) * 90

def update_table(attr, old, new):
    cols_to_display = [checkbox_button_group.labels[i] for i in checkbox_button_group.active]
    update_cols(display_columns=cols_to_display)
    data_save_message.text = 'Configuration not saved'
    data_save_message.styles = {'color': 'red', 'font-size': '16px'}



# --------------- DATA SPLIT ---------------

# saved split list to write to
saved_split_list = [50,25,25] #0-train, 1-val, 2-test

# helper function to produce string
def update_text(train_percentage, val_percentage, test_percentage):
    split_display.text = f"<div>Training split: {train_percentage}</div><div>Validation split: {val_percentage}</div><div>Testing split: {test_percentage}</div>"

# function to update model and accuracy
def update_values(attrname, old, new):
    train_percentage, temp = tvt.value #first segment is train, range of the slider is validate, the rest is test
    val_percentage = temp - train_percentage
    test_percentage = 100 - temp
    if train_percentage < 10 or val_percentage < 10 or test_percentage < 10:
        return
    # print("train_percentage and temp:", train_percentage, temp, "val and test:", val_percentage, test_percentage)
    data_save_message.text = 'Configuration not saved'
    data_save_message.styles = {'color': 'red', 'font-size': '16px'}
    update_text(train_percentage, val_percentage, test_percentage)
    global saved_split_list
    saved_split_list = [train_percentage, val_percentage, test_percentage]

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
split_display = Div(text="<div>Training split: 50</div><div>Validation split: 25</div><div>Testing split: 25</div>")

# --------------- SAVE BUTTON ---------------
# table on change
checkbox_button_group.on_change('active', update_table)

# range slider on change
tvt.js_on_change('value', callback)
tvt.on_change('value', update_values)

# Save columns to saved list (split already saved)
def save_config():
    # print('Configuration saved')

    saved_columns = [checkbox_button_group.labels[i] for i in checkbox_button_group.active]
    # saved_columns = sorted(saved_columns)
    global saved_col_list
    saved_col_list.clear()
    saved_col_list = saved_columns

    #saved_split_list isn't located here as the split values update in the list upon the change of the range slider
    #the collective save button is to make the design more cohesive

    data_save_message.text = 'Configuration saved'
    data_save_message.styles = {'color': 'green', 'font-size': '16px'}

    # print(saved_col_list)
    # print(saved_split_list)
    
# Save button
save_config_button = Button(label="Save Current Configuration", button_type="success")

# Attach callback to the save button
save_config_button.on_click(save_config)

# --------------- ALGORITHM SELECT AND RUN ---------------

# algorithm name holder
my_alg = 'Decision Tree'

# Create status message Div
train_status_message = Div(text='Not running', styles={'color': 'red', 'font-size': '16px'})

# Create select button
alg_select = Select(title="ML Algorithm:", value="Decision Tree", options=["Decision Tree", "K-Nearest Neighbor", "Support Vector Classification"])


def update_algorithm(attr, old, new):
    global my_alg
    my_alg = new
    train_status_message.text = 'Not running'
    train_status_message.styles = {'color': 'red', 'font-size': '16px'}

# Attach callback to Select widget
alg_select.on_change('value', update_algorithm)

# creating widgets
accuracy_display = Div(text="<div>Validation Accuracy: N/A | Test Accuracy: N/A</div>")
# global val_accuracy
val_accuracy = []
# global test_accuracy
test_accuracy = []

# Create empty list
# global combo_list
combo_list = [0, 0, 0, 0, 0, None, None, None, None, None]

def run_config():
    train_status_message.text = f'Running {my_alg}'
    train_status_message.styles = {'color': 'green', 'font-size': '16px'}
    global model

    # Assigning model based on selected ML algorithm, using default hyperparameters
    if my_alg == "Decision Tree":
        model = DecisionTreeClassifier()
    elif my_alg == "K-Nearest Neighbor":
        model = KNeighborsClassifier()
    else:
        model = LinearSVC()
    
    split_and_train_model(saved_split_list[0],saved_split_list[1],saved_split_list[2])

    # Changing the list used to create boxplot
    saved_list = [None, None, None, None, None]
    global combo_list
    combo_list.clear()
    combo_list = val_accuracy + saved_list
    print("val", val_accuracy)
    print("test",test_accuracy)
    print("combo",combo_list)

    # Updating accuracy display
    accuracy_display.text = f"<div><b>Validation Accuracy:</b> {val_accuracy} | <b>Test Accuracy:</b> {test_accuracy}</div>"


def split_and_train_model(train_percentage, val_percentage, test_percentage):

    # run and shuffle five times, and save result in list
    global val_accuracy, test_accuracy

    for i in range(5):
        # Was not running properly with fingerprint
        # TODO: implement fingerprint decoder so model can read them
        X = df[saved_col_list]
        y = df['Class']

        # splitting
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=100-train_percentage)
        test_split = test_percentage / (100-train_percentage)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_split)
        
        # train model
        model.fit(X_train, y_train)
        
        # calculating accuracy
        y_val_pred = model.predict(X_val)
        val_accuracy.append(round(accuracy_score(y_val, y_val_pred), 2))
        y_test_pred = model.predict(X_test)
        test_accuracy.append(round(accuracy_score(y_test, y_test_pred), 2))
        print(i, '. val', val_accuracy)
        print(i, '. test', test_accuracy)


# Run button
train_button = Button(label="Run ML algorithm", button_type="success")

# Attach callback to the run button
train_button.on_click(run_config)


# --------------- VISUALIZATIONS ---------------
'''
# Create empty plot
global p
p = figure(x_range=['current', 'saved'], tools="", toolbar_location=None,
            title="Validation Accuracy saved vs. current",
            background_fill_color="#eaefef", y_axis_label="accuracy")



def update_boxplot():
    d = {'kind': ['current', 'current', 'current', 'current', 'current',
                'saved', 'saved', 'saved', 'saved', 'saved'],
        'accuracy': combo_list
        }
    df_box = pd.DataFrame(data=d)

    kinds = df_box.kind.unique()

    # compute quantiles
    qs = df_box.groupby("kind").accuracy.quantile([0.25, 0.5, 0.75])
    qs = qs.unstack().reset_index()
    qs.columns = ["kind", "q1", "q2", "q3"]
    df_box = pd.merge(df_box, qs, on="kind", how="left")

    # compute IQR outlier bounds
    iqr = df_box.q3 - df_box.q1
    df_box["upper"] = df_box.q3 + 1.5*iqr
    df_box["lower"] = df_box.q1 - 1.5*iqr

    source = ColumnDataSource(df_box)

    # outlier range
    whisker = Whisker(base="kind", upper="upper", lower="lower", source=source)
    whisker.upper_head.size = whisker.lower_head.size = 20
    p.add_layout(whisker)

    # quantile boxes
    cmap = factor_cmap("kind", "Paired3", kinds)
    p.vbar("kind", 0.7, "q2", "q3", source=source, color=cmap, line_color="black")
    p.vbar("kind", 0.7, "q1", "q2", source=source, color=cmap, line_color="black")

    # outliers
    outliers = df_box[~df_box.accuracy.between(df_box.lower, df_box.upper)]
    p.scatter("kind", "accuracy", source=outliers, size=6, color="black", alpha=0.3)

    p.xgrid.grid_line_color = None
    p.axis.major_label_text_font_size="14px"
    p.axis.axis_label_text_font_size="12px"

# Update plot button
update_plot_button = Button(label="Update boxplot", button_type="warning")

# Attach callback to the update_plot button
update_plot_button.on_click(update_boxplot)
'''
# --------------- LAYOUTS ---------------

# creating widget layouts
table_layout = column(checkbox_button_group, datatable)
slider_layout = column(tvt, split_display, save_config_button, data_save_message)
tab2_layout = column(alg_select, train_button, train_status_message, accuracy_display)

# just to see the elements
test_layout = column(slider_layout, tab2_layout) #p, update_plot_button
curdoc().add_root(row(test_layout, table_layout))

# FIXME: update_plot_button and update_boxplot() currently do not work!