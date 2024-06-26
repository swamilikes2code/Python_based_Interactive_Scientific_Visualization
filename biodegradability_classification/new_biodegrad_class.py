import pandas as pd
import numpy as np
from math import nan
from bokeh.io import curdoc, show
from bokeh.layouts import column, row, Spacer, layout
from bokeh.models import ColumnDataSource, DataTable, TableColumn, CheckboxButtonGroup, Button, Div, RangeSlider, Select, Whisker, Slider, Checkbox, Tabs, TabPanel, TextInput, PreText, HelpButton, Tooltip, MultiSelect, HoverTool
from bokeh.models.callbacks import CustomJS
from bokeh.models.dom import HTML
from bokeh.models.ui import SVGIcon
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from bokeh.models import Div
from bokeh.util.warnings import BokehUserWarning, warnings

warnings.simplefilter(action='ignore', category=BokehUserWarning)


#CONTENTS/HEADERS throughout this code
# message styles, accuracy lists, status messages, buttons
# instructions
# data selection, data split, interactive data exploration, save data button
# algorithm select and run
# hyperparameter tuning + button, box plot and save
# testing
# visibility, layouts

# ---------------MESSAGE STYLES-----------------

not_updated = {'color': 'red', 'font-size': '14px'}
loading = {'color': 'orange', 'font-size': '14px'}
updated = {'color': 'green', 'font-size': '14px'}

# ---------------ACCURACY LISTS-----------------
# Create empty list - declare at the top to use everywhere
val_accuracy = []

# ---------------STATUS MESSAGES-----------------

save_config_message = Div(text='Configuration not saved', styles=not_updated)
train_status_message = Div(text='Not running', styles=not_updated)
tune_status_message = Div(text='Not running', styles=not_updated)
temp_test_status_message = Div(text='Not running', styles=not_updated)
predict_status_message = Div(text = 'Not running', styles=not_updated)
delete_status_message = Div(text='Changes not saved', styles = not_updated)

# -------------------BUTTONS--------------------

save_config_button = Button(label="Save Current Configuration", button_type="warning")
train_button = Button(label="Run ML algorithm", button_type="success", width=150)
tune_button = Button(label="Tune", button_type="success", width=150)
delete_button = Button(label = "Delete", button_type = 'danger')
test_button = Button(label = "Test", button_type = "success")
predict_button = Button(label = 'Predict')

#svg icons for buttons
up_arrow = SVGIcon(svg = '''<svg  xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-chevron-up"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M6 15l6 -6l6 6" /></svg>''')
down_arrow = SVGIcon(svg = '''<svg  xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-chevron-down"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M6 9l6 6l6 -6" /></svg>''')

data_exp_vis_button = Button(label="Show Data Exploration*", button_type="primary", icon = down_arrow)

# -----------------INSTRUCTIONS-----------------

intro_instr = Div(
    text="""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f4f4f4;
            }
            .container {
                background-color: #ffffff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1 {
                text-align: center;
                color: #333;
            }
            h2 {
                color: #444;
                border-bottom: 2px solid #ddd;
                padding-bottom: 5px;
            }
            p {
                margin: 15px 0;
            }
            .section {
                margin-bottom: 20px;
                padding: 10px;
                background-color: #fafafa;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .highlight {
                background-color: #e7f3fe;
                border-left: 5px solid #2196F3;
                padding: 2px 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="section">
                <h2>Data Tab</h2>
                <p>Start by opening the <span class="highlight">Data Tab</span>. This tab is used for preparing the biodegradability data for training. In a dropdown menu, you will be given four options containing various <b>fingerprints</b> and <b>molecular features</b> that will be used to train the model. You will also have the option to split the data into <b>training</b>, <b>validating</b>, and <b>testing</b>. Once you are done preparing your data, save your choices, and continue on to the next tab.</p>
            </div>
            <div class="section">
                <h2>Train and Validate Tab</h2>
                <p>On the <span class="highlight">Train and Validate Tab</span>, you will first select the <b>machine learning algorithm</b> of your choice, and run it. This will display the validation accuracy of this run in a datatable. Once you set and run your chosen algorithm, you will be able to fine-tune its <b>hyperparameters</b>, and compare these runs' validation accuracies with your past saved ones in the table. Once you have run a model at least once, you may continue to the next tab.</p>
            </div>
            <div class="section">
                <h2>Test Tab</h2>
                <p>The <span class="highlight">Test Tab</span> is where you will be able to complete your final test of the saved model of your choice. This will display your testing accuracy, as well as both a <b>confusion matrix</b> and <b>tsne plot</b>, which visually display certain performance aspects of your model. Finally, once testing your model, you can continue to the final tab.</p>
            </div>
            <div class="section">
                <h2>Predict Tab</h2>
                <p>The <span class="highlight">Predict Tab</span> is where you will be able to test any of the saved models by inputting a <b>SMILES string</b>. </p>
                <p>FINISH THIS WHEN TEST TAB IS READY.</p>
            </div>
            <div class="section">
                <h2>Additional Information</h2>
                <p>For more information about each of the <b>bolded</b> vocab words, see the above navigation menu.</p>
            </div>
        </div>
    </body>
    </html>
    """,
    width=750,
    height=500
)

show(layout([intro_instr]))


splitter_help = HelpButton(tooltip=Tooltip(content=Div(text="""
                 <div style='background-color: #DEF2F1; padding: 16px; font-family: Arial, sans-serif;'>
                 Use this <b>slider</b> to split the data into <i>train/validate/test</i> percentages.
                 </div>
                 <div style='background-color: #DEF2F1; padding: 16px; font-family: Arial, sans-serif;'>
                 CAUTION: Adjusting the percentages can impact the model's performance, leading to overfitting or underfitting if the splits are not well-balanced of the overall dataset.
                 </div>""", width=280), position="right"))


datatable_help = HelpButton(tooltip=Tooltip(content=Div(text="""
                 <div style='background-color: #DEF2F1; padding: 16px; font-family: Arial, sans-serif;'>
                 <div>Select which group of <b>features</b> you wish to train the model with.</div>
                                                        <div>You can also select the <b>molecular fingerprint</b>, if any.</div>
                 </div>""", width=280), position="right"))

datavis_help = HelpButton(tooltip=Tooltip(content=Div(text="""
                 <div style='background-color: #DEF2F1; padding: 16px; font-family: Arial, sans-serif;'>
                 UPDATE THIS WHEN HISTOGRAMS ARE IN
                 </div>""", width=280), position="right"))

train_help = HelpButton(tooltip=Tooltip(content=Div(text="""
                  <div style='background-color: #DEF2F1; padding: 20px; font-family: Arial, sans-serif;'>
                  Select one of the following <b>Machine Learning algorithms</b>, and click <b>run</b>. This will display its initial 
                                                    <b>validation accuracy</b> in the table on the right.
                  </div>""", width=280), position="right"))

tune_help = HelpButton(tooltip=Tooltip(content=Div(text="""
                 <div style='background-color: #DEF2F1; padding: 20px; font-family: Arial, sans-serif;'>
                 Based on the ML algorithm chosen above, fine-tune its <b>hyperparameters</b> to improve the model's validation accuracy, 
                                                   and click <b>tune</b>. This will also add the run's validation accuracy to the data table.
                 </div>""", width=280), position="right"))

test_help = HelpButton(tooltip=Tooltip(content=Div(text="""
                <div style='background-color: #DEF2F1; padding: 20px; font-family: Arial, sans-serif;'>
                <div>Select the save from the previous tab to test the model.</div>
                <div>‎</div>     
                <div>NOTE: This should be considered the <b>final</b> test of your model.</div>
                <div>You are encouraged to keep exploring the module by continuing to the next tab, or
                starting again from the <b>data</b> tab.</div>
                <div>However, this is NOT intended for validation.</div>
                </div>""", width=280), position="right"))

predict_instr = Div(text="""
                 <div style='background-color: #DEF2F1; padding: 20px; font-family: Arial, sans-serif;'>
                    To create your own SMILES String, go to 
                    <a href="http://pubchem.ncbi.nlm.nih.gov//edit3/index.html" target="_blank">
                    http://pubchem.ncbi.nlm.nih.gov//edit3/index.html </a>
                    (Additional instructions are located on 'Help' button)
                 </div>""",
width=300, height=120)


# --------------- DATA SELECTION ---------------

#for ref:
# df is original csv, holds fingerprint list and 167 cols of fingerprint bits
# df_display > df_subset > df_dict are for displaying table

# Load data from the csv file
file_path = r'rdkit_table.csv'
df = pd.read_csv(file_path, low_memory=False)
df_display = df.iloc[:,:220]  #don't need to display the other 167 rows of fingerprint bits
# df = df.drop(columns=['Fingerprint List'])  #removing the display column, won't be useful in training

# Columns that should always be shown
mandatory_columns = ['Substance Name', 'Smiles', 'Class']

# Ensure mandatory columns exist in the dataframe (if not, create dummy columns) (hopefully shouldn't have to apply)
for col in mandatory_columns:
    if col not in df_display.columns:
        df_display[col] = "N/A"

# for storing data
user_data = ''
user_columns = []
data_opts = ["Fragments", "Molecular/Electronic", "Fingerprint List"]

# Limit the dataframe to the first 15 rows
df_subset = df_display.head(15)

df_rounded = df_subset.round(3)

df_dict = df_rounded.to_dict("list")
cols = list(df_dict.keys())

# Separate mandatory and optional columns
# optional_columns = [col for col in cols if col not in mandatory_columns]

# Create 3 options for columns
option_one = ['fr_COO', 'fr_COO2', 'fr_SH', 'fr_Ar_NH', 'NumHeteroatoms']
option_two = ['MolWt', 'NumValenceElectrons', 'NumRadicalElectrons', 'MaxEStateIndex', 'MinEStateIndex', 'NumAromaticCarbocycles']
option_three = ['Fingerprint List']


# Create column datasource
data_tab_source = ColumnDataSource(data=df_rounded)

# Create figure
data_tab_columns = [TableColumn(field=col, title=col, width=150) for col in cols]
data_tab_table = DataTable(source=data_tab_source, columns=data_tab_columns, width=1000, height_policy = 'auto', autosize_mode = "none")

# Create widget excluding mandatory columns
# checkbox_button_group = CheckboxButtonGroup(labels=optional_columns, active=list(range(len(optional_columns))), orientation = 'vertical')

# menu = [("Item 1", "item_1"), ("Item 2", "item_2"), None, ("Item 3", "item_3")]

data_select = Select(title="Select Features:", options=data_opts)
data_select.js_on_change("value", CustomJS(code="""
    console.log('select: value=' + this.value, this.toString())
"""))

# Update columns to display
def update_cols(display_columns):
    # Always include mandatory columns
    all_columns = mandatory_columns + display_columns
    data_tab_table.columns = [col for col in data_tab_columns if col.title in all_columns]

def update_table(attr, old, new):
    # cols_to_display = [checkbox_button_group.labels[i] for i in checkbox_button_group.active]
    if data_select.value == 'Fragments':
        cols_to_display = option_one
    elif data_select.value == 'Molecular/Electronic':
        cols_to_display = option_two
    elif data_select.value == 'Fingerprint List':
        cols_to_display = option_three
    update_cols(display_columns=cols_to_display)
    save_config_message.text = 'Configuration not saved'
    save_config_message.styles = not_updated

# --------------- DATA SPLIT ---------------

# saved split list to write to
split_list = [50,25,25] #0-train, 1-val, 2-test

# helper function to produce string
def update_text(train_percentage, val_percentage, test_percentage):
    split_display.text = f"""<div style='background-color: #FBE9D0; padding: 20px; font-family: Arial, sans-serif;'>
    Training split: {train_percentage}% | Validation split: {val_percentage}% | Testing split: {test_percentage}%
    </div>"""

# function to update model and accuracy
def update_values(attrname, old, new):
    train_percentage, temp = tvt_slider.value #first segment is train, range of the slider is validate, the rest is test
    val_percentage = temp - train_percentage
    test_percentage = 100 - temp
    if train_percentage < 10 or val_percentage < 10 or test_percentage < 10:
        return

    save_config_message.text = 'Configuration not saved'
    save_config_message.styles = not_updated
    update_text(train_percentage, val_percentage, test_percentage)
    global split_list
    split_list = [train_percentage, val_percentage, test_percentage]

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
tvt_slider = RangeSlider(value=(50, 75), start=0, end=100, step=5, tooltips = False, show_value = False)
tvt_slider.bar_color = '#FAFAFA' # may change later, just so that the segments of the bar look the same
split_display = Div(text="""
                    <div style='background-color: #FBE9D0; padding: 20px; font-family: Arial, sans-serif;'>
                    Training split: 50% | Validation split: 25% | Testing split: 25%
                    </div>""")

# --------------- INTERACTIVE DATA EXPLORATION --------------- 

# get columns
data_exp_columns = df.columns.tolist()[:21]

#columns to exclude
data_exp_columns = [col for col in data_exp_columns if col not in ["Class", "Smiles", "Substance Name", "Fingerprint"]]

#convert the class columns to a categorical column if it's not
df['Class'] = df['Class'].astype('category')

# Create a ColumnDataSource
data_exp_source = ColumnDataSource(data=dict(x=[], y=[], class_color=[], names = []))

# configure hovertool
tooltips = [
    ("name", "@names"),
    ("index", "$index")
]

# Create a figure
data_exp = figure(title="Data Exploration: search for correlations between properties", width = 600, height = 320, x_axis_label='X', y_axis_label='Y', 
           tools="pan,wheel_zoom,box_zoom,reset,save", tooltips = tooltips)


# Create an initial scatter plot
data_exp_scatter = data_exp.scatter(x='x', y='y', color='class_color', source=data_exp_source, legend_field='class_label')

# legend
data_exp.add_layout(data_exp.legend[0], 'right')

# Create dropdown menus for X and Y axis
select_x = Select(title="X Axis", value=data_exp_columns[0], options=data_exp_columns)
select_y = Select(title="Y Axis", value=data_exp_columns[1], options=data_exp_columns)

# Update the data based on the selections
def update_data_exp(attrname, old, new):
    x = select_x.value
    y = select_y.value
    new_vis_data = {
        'x': df[x],
        'y': df[y],
        'names' : df['Substance Name'],
        'class_color': ['#900C3F' if cls == df['Class'].cat.categories[0] else '#1DBD4D' for cls in df['Class']],
        'class_label': ['Not readily biodegradable' if cls == df['Class'].cat.categories[0] else 'Readily biodegradable' for cls in df['Class']]
    }
        
    # Update the ColumnDataSource with a plain Python dict
    data_exp_source.data = new_vis_data
    
    # Update existing scatter plot glyph if needed
    data_exp_scatter.data_source.data = new_vis_data
    
    data_exp.xaxis.axis_label = x
    data_exp.yaxis.axis_label = y

# Attach the update_data function to the dropdowns
select_x.on_change('value', update_data_exp)
select_y.on_change('value', update_data_exp)

update_data_exp(None, None, None)


# --------------- SAVE DATA BUTTON ---------------

# table on change
# checkbox_button_group.on_change('active', update_table)
data_select.on_change('value', update_table)

# range slider on change
tvt_slider.js_on_change('value', callback)
tvt_slider.on_change('value', update_values)

def set_columns():
    global user_columns, user_data
    if user_data == 'Fragments':
        user_columns = option_one
    elif user_data == 'Molecular/Electronic':
        user_columns = option_two
    elif user_data == 'Fingerprint List':
        user_columns = option_three

# Save columns to saved list (split already saved)
def save_config():
    # temp_columns = [checkbox_button_group.labels[i] for i in checkbox_button_group.active]
    global user_data, user_columns

    if data_select.value == 'Fragments':
        user_data = 'Fragments'
        user_columns = option_one
    elif data_select.value == 'Molecular/Electronic':
        user_data = 'Molecular/Electronic'
        user_columns = option_two
    elif data_select.value == 'Fingerprint List':
        user_data = 'Fingerprint List'
        user_columns = option_three
    else:
        save_config_message.text = 'Error: select an option before saving'
        save_config_message.styles = not_updated
        return

    #split_list isn't located here as the split values update in the list upon the change of the range slider
    #the collective save button is to make the design more cohesive

    #split data when saved to withold the test set and always use the same train val sets
    split_data(split_list[0],split_list[1],split_list[2],user_columns)

    save_config_message.text = 'Configuration saved'
    save_config_message.styles = updated

def load_config():
    save_config_message.text = "Loading config..."
    save_config_message.styles = loading

    train_status_message.text='Not running'
    train_status_message.styles=not_updated


    tune_status_message.text='Not running'
    tune_status_message.styles=not_updated

    curdoc().add_next_tick_callback(save_config)

# Attach callback to the save button
save_config_button.on_click(load_config)

# --------------- ALGORITHM SELECT AND RUN ---------------

# algorithm name holder
my_alg = 'Decision Tree'

# list of the models to use
model_list = [DecisionTreeClassifier(), KNeighborsClassifier(), SVC()]

# Create select button
alg_select = Select(title="Select ML Algorithm:", value="Decision Tree", options=["Decision Tree", "K-Nearest Neighbor", "Support Vector Classification"])

def update_algorithm(attr, old, new):
    global my_alg
    my_alg = new
    train_status_message.text = 'Not running'
    train_status_message.styles = not_updated

# Attach callback to Select widget
alg_select.on_change('value', update_algorithm)

# creating widgets
accuracy_display = Div(text="""<div style='background-color: #FBE9D0; padding: 20px; font-family: Arial, sans-serif;'>
                       <div><b>Your Data Split:</b> N/A </div><div><b>Your Selected columns:</b> N/A</div><div><b>Validation Accuracy:</b> N/A</div><div><b>Test Accuracy:</b> N/A</div>
                       </div>""", width=600)
test_accuracy = []


def run_ML():
    if save_config_message.styles == not_updated:
        train_status_message.text = 'Error: must save data configuration before training'
        train_status_message.styles = not_updated
        return

    #stage can be train or tune, determines which list to write to
    global stage
    global model
    stage = 'Train'
    train_status_message.text = f'Algorithm: {my_alg}'
    train_status_message.styles = updated

    # Assigning model based on selected ML algorithm, using default hyperparameters
    if my_alg == "Decision Tree":
        model = model_list[0]
    elif my_alg == "K-Nearest Neighbor":
        model = model_list[1]
    else:
        model = model_list[2]
    
    set_hyperparameter_widgets()

    train_validate_model()

    # Updating accuracy display
    accuracy_display.text = f"""<div style='background-color: #FBE9D0; padding: 20px; font-family: Arial, sans-serif;'>
    <div><b>Your Data Split:</b> {split_list} </div><div><b>Your Selected columns:</b> {user_columns}<div><b>Validation Accuracy:</b> {val_accuracy}</div><div><b>Test Accuracy:</b> {test_accuracy}</div>
    </div>"""

def split_data(train_percentage, val_percentage, test_percentage, columns):
    global X_train, X_val, X_test, y_train, y_val, y_test

    train_columns = []
    train_columns += columns
    if 'Fingerprint List' in columns:
        train_columns.remove("Fingerprint List")
        train_columns += [str(i) for i in range(167)]

    X = df[train_columns]
    y = df['Class']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(100-train_percentage)/100)
    test_split = test_percentage / (100-train_percentage)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_split)

def train_validate_model():
    np.random.seed(123)

    # train model
    model.fit(X_train, y_train)
    
    # calculating accuracy
    y_val_pred = model.predict(X_val)

    val_accuracy.append(round(accuracy_score(y_val, y_val_pred), 3))

    save_model()

def load_ML():
    train_status_message.text = f'Running {my_alg}...'
    train_status_message.styles = loading

    tune_status_message.text='Not running'
    tune_status_message.styles=not_updated

    curdoc().add_next_tick_callback(run_ML)

# Attach callback to the run button
train_button.on_click(load_ML)


# --------------- HYPERPARAMETER TUNING + BUTTON ---------------

# a list of an int an string
## decision tree - int/nan, string
## KNN - int, string
## SVC - int, ""

# define to be default: decision tree
hyperparam_list = [2, "random"]

# create displays
tuned_accuracy_display = Div(text = """
                             <div style='background-color: #FBE9D0; padding: 20px; font-family: Arial, sans-serif;'>
                             <div><b>Tuned Validation Accuracy:</b> N/A</div>
                             <div>‎</div>
                             <div>‎</div>
                             <div><b>Tuned Test Accuracy:</b> N/A</div>
                             </div>""")

def run_tuned_config():
    if save_config_message.styles == not_updated:
        tune_status_message.text = 'Error: must save data configuration before tuning'
        tune_status_message.styles = not_updated
        return
    elif train_status_message.styles == not_updated:
        tune_status_message.text = 'Error: must train model before tuning'
        tune_status_message.styles = not_updated
        return

    global my_alg, stage
    stage = 'Tune'
    tune_status_message.text = f'Algorithm: {my_alg}'
    tune_status_message.styles = updated
    global model

    train_validate_model()

    # Updating accuracy display
    tuned_accuracy_display.text = f"""<div style='background-color: #FBE9D0; padding: 20px; font-family: Arial, sans-serif;'>
    <div><b>Tuned Validation Accuracy:</b> {val_accuracy[-1]}</div> 
    </div>"""

# hyperparameter tuning widgets, default to decision tree
hp_slider = Slider(
    title = "Max Depth of Tree",
    start= 1,
    end = 15,
    value = 2,
    step = 1,
    width=200
)
hp_select = Select(
    title = "Splitter strategy",
    value = "random",
    options = ["random", "best"],
    width= 200
)
hp_toggle = Checkbox(
    label = "None",
    visible = True,
    active = False
)
#hp_toggle.margin = (24, 10, 5, 10)

# setting widget callbacks
def print_vals():
    global my_alg
    if my_alg == 'Decision Tree':
        print("slider", hp_slider.value)
        print("switch", hp_toggle.active)
        print("model", model.max_depth)
        print("model splitter", model.splitter)
    elif my_alg == 'K-Nearest Neighbor':
        print("slider", hp_slider.value)
        print("n_neighbors", model.n_neighbors)
        print("weights", model.weights)
    elif my_alg == 'Support Vector Classification':
        print("slider", hp_slider.value)
        print("C", model.C)
        print("kernel", model.kernel)

def hp_slider_callback(attr, old, new):
    if hp_slider.disabled == True:
        return

    global my_alg
    hyperparam_list[0] = new

    if my_alg == 'Decision Tree':
        if hp_slider.disabled == True:
            hyperparam_list[0] = None
            return
        model.max_depth = new
    elif my_alg == 'K-Nearest Neighbor':
        model.n_neighbors = new
    elif my_alg == 'Support Vector Classification':
        model.C = new

def hp_select_callback(attr, old, new):
    global my_alg
    hyperparam_list[1] = new
    if my_alg == 'Decision Tree':
        model.splitter = new
    elif my_alg == 'K-Nearest Neighbor':
        model.weights = new
    elif my_alg == 'Support Vector Classification':
        model.kernel = new

def hp_toggle_callback(attr, old, new):
    if my_alg == 'Decision Tree':
        if new == True:
            hp_slider.update(disabled = True, show_value = False)
            model.max_depth = None
            hyperparam_list[0] = None
        elif new == False:
            hp_slider.update(disabled = False, bar_color = '#e6e6e6', show_value = True)
            model.max_depth = hp_slider.value
            hyperparam_list[0] = hp_slider.value

def set_hyperparameter_widgets():
    global model
    global my_alg
    if my_alg == 'Decision Tree':
        #hyperparameters are 
        # splitter strategy (splitter, best vs. random, select)
        # max_depth of tree (max_depth, int slider)

        hp_slider.update(
            title = "Max Depth of Tree",
            disabled = False,
            show_value = True,
            start= 1,
            end = 15,
            value = 2,
            step = 1
        )
        hp_toggle.update(
            label = "None",
            visible = True,
            active = False
        )
        hp_select.update(
            title = "Splitter strategy",
            value = "random",
            options = ["random", "best"]
        )

        model.max_depth = hp_slider.value
        model.splitter = hp_select.value
    elif my_alg == 'K-Nearest Neighbor':
        #hyperparameters are 
        # K (n_neighbors, int slider)
        # weights (weights, uniform vs. distance, select)
        
        hp_slider.update(
            title = "Number of neighbors",
            disabled = False,
            show_value = True,
            start = 1,
            end = 30,
            value = 20,
            step = 2
        )

        hp_toggle.visible = False

        hp_select.update(
            title = "Weights",
            value = "uniform",
            options = ["uniform", "distance"]
        )
    elif my_alg == 'Support Vector Classification':
        #hyperparameters are 
        # kernel (linear, poly, rbf, sigmoid) 
        # regularization parameter C (float slider)
        # model = SVC()

        hp_slider.update(
            title = "C, regularization parameter",
            disabled = False,
            show_value = True,
            start = 1,
            end = 100,
            value = 50,
            step = 1
        )

        hp_toggle.visible = False

        hp_select.update(
            title = "kernel",
            value = "linear",
            options = ["linear", "poly", "rbf", "sigmoid"]
        )

hp_slider.on_change('value', hp_slider_callback)
hp_select.on_change('value', hp_select_callback)
hp_toggle.on_change('active', hp_toggle_callback)


def load_tuned_config():
    tune_status_message.text = "Loading tuned config..."
    tune_status_message.styles = loading
    
    curdoc().add_next_tick_callback(run_tuned_config)

# Can connect to the old funcs
tune_button.on_click(load_tuned_config)


# --------------- BOX PLOT AND SAVE ---------------

# making select to choose save num to display/use
delete_multiselect = MultiSelect(title = "Choose saves to delete", options = [], margin=(5, 40, 5, 5))
test_save_select = Select(title = "Choose a save to test", options = [], margin=(5, 40, 5, 5))
predict_select = Select(title = 'Choose a save to predict with', options = [])

new_save_number = 0

# Define an empty data source
saved_data = dict(
    save_number = [],
    train_val_test_split = [],
    saved_data_choice = [],
    saved_algorithm = [],
    saved_hyperparams = [],
    saved_val_acc = []
)
save_source = ColumnDataSource(saved_data)

# Define table columns
saved_columns = [
    TableColumn(field="save_number", title="#", width = 25),
    TableColumn(field="train_val_test_split", title="Train/Val/Test split", width = 260),
    TableColumn(field="saved_data_choice", title="Saved col."),
    TableColumn(field="saved_algorithm", title="Saved alg.", width = 140),
    TableColumn(field="saved_hyperparams", title="Saved hp.", width = 220),
    TableColumn(field="saved_val_acc", title="Pred. accuracy")
]

# Create a DataTable
saved_data_table = DataTable(source=save_source, columns=saved_columns, width=600, height=280, index_position=None)


def save_model():
    global hyperparam_list
    global new_save_number
    global new_train_val_test_split
    global new_saved_data_choice
    global new_saved_algorithm
    global new_saved_hyperparams
    global new_saved_val_acc

    new_saved_val_acc = val_accuracy[new_save_number]  # access before save num is incremented

    new_save_number += 1
    test_save_select.options.append(str(new_save_number))
    delete_multiselect.options.append(str(new_save_number))
    predict_select.options.append(str(new_save_number))

    new_train_val_test_split = str(split_list[0]) + '/' + str(split_list[1]) + '/' + str(split_list[2])

    new_saved_data_choice = user_data

    if my_alg == 'Decision Tree':
        new_saved_algorithm = 'DT'
    elif my_alg == 'K-Nearest Neighbor':
        new_saved_algorithm = 'KNN'
    elif my_alg == 'Support Vector Classification':
        new_saved_algorithm = 'SVC'
    else:
        new_saved_algorithm = my_alg
    new_saved_hyperparams = str(hyperparam_list) # convert back to list for usage when loading a saved profile

    add_row()

# Add new row to datatable every time a plot is saved
def add_row():
    new_saved_data = {
        'save_number': [new_save_number],
        'train_val_test_split': [new_train_val_test_split],
        'saved_data_choice': [new_saved_data_choice],
        'saved_algorithm': [new_saved_algorithm],
        'saved_hyperparams': [new_saved_hyperparams],
        'saved_val_acc' : [new_saved_val_acc]
    }
    save_source.stream(new_saved_data)

# def display_save():
#     global saved_accuracy, saved_test_acc
#     index = int(test_save_select.value) - 1
#     saved_accuracy = saved_test_acc[index]


# def load_display_save():
#     curdoc().add_next_tick_callback(display_save)

# # callback to display_save button
# display_save_button.on_click(load_display_save)

def delete_save():
    saves_to_del = [int(i) for i in delete_multiselect.value]

    if len(delete_multiselect.options) <= 1 or len(saves_to_del) == len(delete_multiselect.options):
        delete_status_message.text = 'Error: must have at least one save'
        delete_status_message.styles = not_updated
        return

    temp = [save for save in delete_multiselect.options if int(save) not in saves_to_del]
    delete_multiselect.update(
        options = temp.copy(),
        value = []
    )
    test_save_select.update(
        options = temp.copy(),
        value = None
    )

    for col in save_source.data:
        save_source.data[col] = [val for index, val in enumerate(save_source.data[col]) if (index+1) not in saves_to_del]

    delete_status_message.text = 'Deleted'
    delete_status_message.styles = updated

def load_delete_save():
    delete_status_message.text = 'Deleting...'
    delete_status_message.styles = loading
    curdoc().add_next_tick_callback(delete_save)

def del_multiselect_callback(attr, old, new):
    delete_status_message.text = 'Changes not saved'
    delete_status_message.styles = not_updated

# callback to display_save button
delete_button.on_click(load_delete_save)
delete_multiselect.on_change('value', del_multiselect_callback)

# --------------- TESTING -----------------
def train_test_model():
    np.random.seed(123)

    save_num = int(test_save_select.value)
    save_index = save_num-1
    temp_split = [int(split) for split in save_source.data['train_val_test_split'][save_index].split("/")]
    temp_data_choice = save_source.data['saved_data_choice'][save_index]
    temp_alg = save_source.data['saved_algorithm'][save_index]
    temp_hyperparams = eval(save_source.data['saved_hyperparams'][save_index])

    if temp_data_choice == 'Fragments':
        temp_cols = option_one
    elif temp_data_choice == 'Molecular/Electronic':
        temp_cols = option_two
    elif temp_data_choice == 'Fingerprint List':
        temp_cols = option_three
    
    split_data(temp_split[0], temp_split[1], temp_split[2],temp_cols)

    if temp_alg == 'DT':
        model = model_list[0]
        model.max_depth = temp_hyperparams[0]
        model.splitter = temp_hyperparams[1]
    elif temp_alg == 'KNN':
        model = model_list[1]
        model.n_neighbors = temp_hyperparams[0]
        model.weights = temp_hyperparams[1]
    elif temp_alg == 'SVC':
        model = model_list[2]
        model.C = temp_hyperparams[0]
        model.kernel = temp_hyperparams[1]


    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)

    test_accuracy.append(round(accuracy_score(y_test, y_test_pred), 3))


def run_test():
    global my_alg, stage
    stage = 'Test'
    global model

    train_test_model()

    # Updating accuracy display
    temp_test_status_message.text = f"""{test_accuracy}</div> 
    </div>"""
    temp_test_status_message.styles = updated

def load_test():
    temp_test_status_message.text = "Testing..."
    temp_test_status_message.styles = loading
    
    curdoc().add_next_tick_callback(run_test)

test_button.on_click(load_test)


# --------------- PREDICTING ---------------

user_smiles_input = TextInput(title = 'Enter a SMILES string:')
# test in dataset C=C(C)C(=O)O

# def predict_biodegrad():
#     temp_tvt_list = new_train_val_test_split.split("/")
#     temp_train = int(temp_tvt_list[0])
#     temp_val = int(temp_tvt_list[1])
#     temp_test = int(temp_tvt_list[2])
#     temp_columns = save_source.data['saved_columns'][int(predict_select.value)-1]

#     train_validate_model(temp_train,temp_val,temp_test, temp_columns)

#     user_molec = Chem.MolFromSmiles(user_smiles_input.value)

#     user_fp = np.array(MACCSkeys.GenMACCSKeys(user_molec))

#     user_df = pd.DataFrame(user_fp)

#     user_df = user_df.transpose() #each bit has its own column

#     # --------------TAB WORKS UP UNTIL HERE-----------------------
#     # The model is not receiving the actual saved columns it needs, except for fingerprint. 
#     # For example, has 167 features, but is expecting 185 features as input (there are 18 columns, excluding fingerprint)
#     # If I get to it I'll try to fix this this weekend

#     user_biodegrad = model.predict(user_df)

#     print(user_biodegrad)

#     predict_status_message.styles = updated
#     # if user_biodegrad == 0:
#     #     predict_status_message.text = 'Molecule is not readily biodegradable (class 0)'
#     # elif user_biodegrad == 1:
#     #     predict_status_message.text = 'Molecule is readily biodegradable (class 1)'
#     # else:
#     #     predict_status_message.text = 'error'

#     return

# def load_predict():
#     predict_status_message.text = 'Predicting...'
#     predict_status_message.styles = loading
#     curdoc().add_next_tick_callback(predict_biodegrad)

# # callback for predict button
# predict_button.on_click(load_predict)


# ---------------- VISIBILITY --------------

# Data exploration plot
datavis_help.visible = False
data_exp.visible = False
select_x.visible = False
select_y.visible = False

# Callback function to toggle visibility
def toggle_data_exp_visibility():
    datavis_help.visible = not datavis_help.visible
    data_exp.visible = not data_exp.visible
    select_x.visible = not select_x.visible
    select_y.visible = not select_y.visible
    data_exp_vis_button.label = "Show Data Exploration*" if not data_exp.visible else "Hide Data Exploration*"
    data_exp_vis_button.icon = down_arrow if not data_exp.visible else up_arrow

# Link the button to the callback
data_exp_vis_button.on_click(toggle_data_exp_visibility)

# --------------- LAYOUTS ---------------

height_spacer = Spacer(height = 30)
small_height_spacer = Spacer(height = 15)
large_height_spacer = Spacer(height = 45)
button_spacer = Spacer(height = 30, width = 54)
top_page_spacer = Spacer(height = 10)
left_page_spacer = Spacer(width = 10)

# creating widget layouts
tab0_layout = row(left_page_spacer, column(top_page_spacer, intro_instr))

data_config_layout = layout(
    [datatable_help, data_select],
    [small_height_spacer],
    [splitter_help, column(tvt_slider, split_display)],
    [small_height_spacer],
    [button_spacer, column(save_config_button, save_config_message)]
)
interactive_graph = column(data_exp_vis_button, row(datavis_help, column(data_exp, row(select_x, select_y)))) #create data graph visualization 
tab1_layout = row(left_page_spacer, column(top_page_spacer, row(data_tab_table, data_config_layout), small_height_spacer, interactive_graph))

hyperparam_layout = layout(
    [hp_slider],
    [hp_toggle],
    [hp_select],
    [tune_button, tune_help],
    [tune_status_message],
    [height_spacer]
)

delete_layout = layout(
    [delete_multiselect],
    [delete_button],
    [delete_status_message]
)

tab2_layout = row(left_page_spacer, column(top_page_spacer,  alg_select, row( train_button, train_help), train_status_message, height_spacer, hyperparam_layout, delete_layout), left_page_spacer, saved_data_table)

# save_layout = row(column(test_save_select, display_save_button), saved_data_table)
tab3_layout = row(left_page_spacer, column(top_page_spacer, test_help, test_save_select, test_button, temp_test_status_message))

tab4_layout = row(left_page_spacer, column(top_page_spacer, predict_instr, user_smiles_input, predict_button, predict_status_message))

tabs = Tabs(tabs = [TabPanel(child = tab0_layout, title = 'Instructions'),
                    TabPanel(child = tab1_layout, title = 'Data'),
                    TabPanel(child = tab2_layout, title = 'Train and Validate'),
                    TabPanel(child = tab3_layout, title = 'Test'),
                    TabPanel(child = tab4_layout, title = 'Predict')
                ])

curdoc().add_root(tabs)