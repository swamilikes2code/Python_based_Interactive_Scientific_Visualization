import pandas as pd
import numpy as np
import io
from io import BytesIO
import openpyxl
import random
import base64
from math import nan
from bokeh.events import ButtonClick
from bokeh.io import curdoc, show
from bokeh.layouts import column, row, Spacer, layout
from bokeh.models import Div, ColumnDataSource, DataTable, TableColumn, CheckboxButtonGroup, Button, RangeSlider, Select, Whisker, Slider, Checkbox, Tabs, TabPanel, TextInput, PreText, HelpButton, Tooltip, MultiSelect, HoverTool, LinearColorMapper, ColorBar, BasicTicker, PrintfTickFormatter
from bokeh.models.callbacks import CustomJS
from bokeh.models.dom import HTML
from bokeh.models.ui import SVGIcon
from bokeh.palettes import Viridis256
from bokeh.plotting import figure, show
from bokeh.transform import factor_cmap, transform
import pubchempy
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
from rdkit.Chem import MACCSkeys, MolFromSmiles, DataStructs, Descriptors, AllChem
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
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
completed = {'color': 'black', 'font-size': '14px'}

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

save_config_button = Button(label="Save Current Configuration", button_type="warning", width = 250)
train_button = Button(label="Run ML algorithm", button_type="success", width=150, height = 31)
tune_button = Button(label="Tune", button_type="success", width=150, height = 31)
delete_button = Button(label = "Delete", button_type = 'danger', width = 200, height = 31)
test_button = Button(label = "Test", button_type = "success", width = 150, height = 31)
predict_button = Button(label = 'Predict', button_type = "success", width = 200, height = 31)

#svg icons for buttons
up_arrow = SVGIcon(svg = '''<svg  xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-chevron-up"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M6 15l6 -6l6 6" /></svg>''')
down_arrow = SVGIcon(svg = '''<svg  xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-chevron-down"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M6 9l6 6l6 -6" /></svg>''')

data_exp_vis_button = Button(label="Show Data Exploration*", button_type="primary", icon = down_arrow)

export_excel = Button(label="Download Full Table to Excel (.xlsx)", width=200, height=31)
export_csv = Button(label="Download Full Table to CSV (.csv)", width=200, height=31)

# -----------------HTML TEMPLATES-----------------
html_val_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f4f4f4;
        }}
        .container {{
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        h2 {{
            color: #444;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
        }}
        p {{
            margin: 15px 0;
        }}
        .section {{
            margin-bottom: 20px;
            padding: 10px;
            background-color: #fafafa;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .highlight {{
            background-color: #e7f3fe;
            border-left: 5px solid #2196F3;
            padding: 2px 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="section">
            <h2>Latest Validation Accuracy:</h2>
            <p>{}</p>
        </div>
    </div>
</body>
</html>
"""

html_test_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.0;
            margin: 0;
            padding: 20px;
            background-color: #f4f4f4;
        }}
        .container {{
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .column_4 {{
            float: left;
            width: 25%;
        }}

        .column_2 {{
            float: left;
            width: 50%;
        }}

        .row:after {{
            content: "";
            display: table;
            clear: both;
        }}

        h1 {{
            text-align: center;
            color: #333;
        }}
        h2 {{
            color: #444;
            border-bottom: 2px solid #ddd;
            padding-bottom: 1px;
        }}
        h3 {{
            color: #555;
        }}
        p {{
            margin: 5px 0;
            max-width: 575px;
        }}
        .section {{
            margin-bottom: 20px;
            padding: 10px;
            background-color: #fafafa;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .highlight {{
            background-color: #e7f3fe;
            border-left: 5px solid #2196F3;
            padding: 2px 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="section">
            <h2>Testing Accuracy:</h2>
            <p>{}</p>
        </div>

        <div class="row">
            <h2>Counts:</h2>
            <div class="column_4">
                <h3>True Positives:</h3>
                <p>{}</p>
            </div>
            <div class="column_4">
                <h3>False Positives:</h3>
                <p>{}</p>
            </div>
            <div class="column_4">
                <h3>False Negatives:</h3>
                <p>{}</p>
            </div>
            <div class="column_4">
                <h3>True Negatives:</h3>
                <p>{}</p>
            </div>
        </div>

        <div class-"row">
            <h2>Percentages:</h2>
            <div class="column_2">
                <h3>Precision for Ready Biodegradability:</h3>
                <p>{}</p>
            </div>
            <div class="column_2">
                <h3>Precision for Non-ready Biodegradability:</h3>
                <p>{}</p>
            </div>
        </div>

        <div class="row">
            <h2>Performance:</h2>
            <p>On average, your model was {}* at classifying ready biodegradabiliy, and {}* at classifying non-ready biodegradability,
            resulting in an overall {}* performance.</p>
        </div>

        <div class="row">
            <h2>NOTE:</h2>
            <p><b>Precision for Ready Biodegradability:</b> true positives/(true postives + false positives)--> Percentage of predicted positives
              that were true (actual) positives.</p>
            <p><b>Precision for Non-ready Biodegradability:</b> true negatives/(true negatives + false negatives)--> Percentage of predicted negatives
              that were true (actual) negatives.</p>
            <p><b>Testing accuracy:</b> Average of the two precision values.</p>
            <p>*'POOR' performance--> accuracy value of less than 50%</p> 
            <p>*'FAIR' performance--> accuracy value between 50% and 75%</p> 
            <p>*'EXCELLENT' performance--> accuracy value greater than 75%</p>
        </div>
    </div>
</body>
</html>
"""

html_predict_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f4f4f4;
        }}
        .container {{
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        h2 {{
            color: #444;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
        }}
        p {{
            margin: 15px 0;
        }}
        .section {{
            margin-bottom: 20px;
            padding: 10px;
            background-color: #fafafa;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .highlight {{
            background-color: #e7f3fe;
            border-left: 5px solid #2196F3;
            padding: 2px 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="section">
            <h2>Substance Name:</h2>
            <p>{}</p>
            <h2>Smiles String:</h2>
            <p>{}</p>
            <h2>Predicted Class</h2>
            <p>{}</p>
            <h2>Actual Class</h2>
            <p>{}</p>
        </div>
    </div>
</body>
</html>
"""

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

formatted_val_html = html_val_template.format('N/A')
val_acc_display = Div(text=formatted_val_html)

formatted_test_html = html_test_template.format('N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A')
test_acc_display = Div(text=formatted_test_html)

formatted_predict_html = html_predict_template.format('N/A', 'N/A', 'N/A', 'N/A')
predict_display = Div(text=formatted_predict_html)


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
                <div>Select the save from the previous tab to test the model, and view the number of true/false postives/negatives
                                                   in the <b>confusion matrix</b> below.</div>
                <div>‎</div>     
                <div>NOTE: This should be considered the <b>final</b> test of your model.</div>
                <div>You are encouraged to keep exploring the module by continuing to the next tab, or
                starting again from the <b>data</b> tab.</div>
                <div>However, this is NOT intended for validation.</div>
                </div>""", width=280), position="right"))

predict_instr = Div(text="""
                 <div style='background-color: #DEF2F1; padding: 1px; font-family: Arial, sans-serif;'>
                    To create a SMILES String, visit
                    <a href="http://pubchem.ncbi.nlm.nih.gov//edit3/index.html" target="_blank">
                    pubchem.ncbi.nlm.nih.gov//edit3/index.html </a>
                    (Instructions on 'Help' button)
                 </div>""",
width=160, height=60)


# --------------- DATA SELECTION ---------------

#for ref:
# df is original csv, holds fingerprint list and 167 cols of fingerprint bits
# df_display > df_subset > df_dict are for displaying table

# Load data from the csv file
df1 = pd.read_csv("./data/option_1.csv", low_memory=False)
df2 = pd.read_csv("./data/option_2.csv", low_memory=False)
df3 = pd.read_csv("./data/option_3.csv", low_memory=False)
df4 = pd.read_csv("./data/option_4.csv", low_memory=False)

all_df = [df1, df2, df3, df4]

# just holding mandatory cols
df = df1.iloc[:, :4]

# Columns that should always be shown
mandatory_columns = ['Substance Name', 'Smiles', 'Class']

# for storing data choice
user_data = ''
user_index = 0
data_opts = ['Molecular Properties', 'Morgan Fingerprint', 'ECFP2', 'Path Fingerprint']

# Limit the dataframe to the first 15 rows
df1_subset = df1.iloc[:15, :10].round(3)
df2_subset = df2.iloc[:15, :10]
df3_subset = df3.iloc[:15, :10]
df4_subset = df4.iloc[:15, :10]

df1_tab_source = ColumnDataSource(df1_subset)
df2_tab_source = ColumnDataSource(df2_subset)
df3_tab_source = ColumnDataSource(df3_subset)
df4_tab_source = ColumnDataSource(df4_subset)

df1_dict = df1.to_dict("list")
df2_dict = df2.to_dict("list")
df3_dict = df3.to_dict("list")
df4_dict = df4.to_dict("list")

cols1 = [key for key in df1_dict.keys() if key not in mandatory_columns]
cols2 = [key for key in df2_dict.keys() if key not in mandatory_columns]
cols3 = [key for key in df3_dict.keys() if key not in mandatory_columns]
cols4= [key for key in df4_dict.keys() if key not in mandatory_columns]

all_cols = [cols1, cols2, cols3, cols4]

# Create figure
data_tab_columns = [TableColumn(field=col, title=col, width=150) for col in (mandatory_columns+cols1[:7])]
data_tab_table = DataTable(source=df1_tab_source, columns=data_tab_columns, width=1000, height_policy = 'auto', autosize_mode = "none")

data_select = Select(title="Select Features:", options=data_opts, width = 195)

# Update columns to display
def update_cols(display_columns, table_source):
    all_columns = mandatory_columns + display_columns
    data_tab_table.source = table_source
    data_tab_table.columns = [TableColumn(field=col, title=col, width=150) for col in all_columns]

def update_table(attr, old, new):
    if data_select.value == 'Molecular Properties':
        cols_to_display = cols1[:7]
        table_source = df1_tab_source
    elif data_select.value == 'Morgan Fingerprint':
        cols_to_display = cols2[:7]
        table_source = df2_tab_source
    elif data_select.value == 'ECFP2':
        cols_to_display = cols3[:7]
        table_source = df3_tab_source
    elif data_select.value == 'Path Fingerprint':
        cols_to_display = cols4[:7]
        table_source = df4_tab_source
    update_cols(display_columns=cols_to_display, table_source=table_source)
    save_config_message.text = 'Configuration not saved'
    save_config_message.styles = not_updated

# table on change
data_select.on_change('value', update_table)

# --------------- DATA SPLIT ---------------

# saved split list to write to
split_list = [50, 25, 25] #0-train, 1-val, 2-test

# helper function to produce string
def update_split_text(train_percentage, val_percentage, test_percentage):
    split_display.text = f"""<div style='background-color: #FBE9D0; padding: 20px; font-family: Arial, sans-serif;'>
    Train: {train_percentage}% || Validate: {val_percentage}% || Test: {test_percentage}%
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
    update_split_text(train_percentage, val_percentage, test_percentage)
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
tvt_slider = RangeSlider(title= 'Split Data', value=(50, 75), start=0, end=100, step=5, tooltips = False, show_value = False, width = 195)
tvt_slider.bar_color = '#FAFAFA' # may change later, just so that the segments of the bar look the same
split_display = Div(text="""
                    <div style='background-color: #FBE9D0; padding: 20px; font-family: Arial, sans-serif;'>
                    Train: 50% || Validate: 25% || Test: 25%
                    </div>""", width = 250)

# range slider on change
tvt_slider.js_on_change('value', callback)
tvt_slider.on_change('value', update_values)

# --------------- SAVE DATA BUTTON ---------------

# Save columns to saved list (split already saved)
def save_config():
    global user_data, user_index
    user_data = data_select.value

    if user_data not in data_opts:
        save_config_message.text = 'Error: select an option before saving'
        save_config_message.styles = not_updated
        return
    
    user_index = data_opts.index(user_data)

    #split_list isn't located here as the split values update in the list upon the change of the range slider
    #the collective save button is to make the design more cohesive

    #split data when saved to withold the test set and always use the same train val sets
    split_data(split_list[0],split_list[1],split_list[2], user_index)

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

# --------------- INTERACTIVE DATA EXPLORATION --------------- 

# # get columns
# data_exp_columns = df.columns.tolist()[:21]

# #columns to exclude
# data_exp_columns = [col for col in data_exp_columns if col not in ["Class", "Smiles", "Substance Name", "Fingerprint"]]

# #convert the class columns to a categorical column if it's not
# df['Class'] = df['Class'].astype('category')

# # Create a ColumnDataSource
# data_exp_source = ColumnDataSource(data=dict(x=[], y=[], class_color=[], names = []))

# # configure hovertool
# tooltips = [
#     ("name", "@names"),
#     ("index", "$index")
# ]

# # Create a figure
# data_exp = figure(title="Data Exploration: search for correlations between properties", width = 600, height = 320, x_axis_label='X', y_axis_label='Y', 
#            tools="pan,wheel_zoom,box_zoom,reset,save", tooltips = tooltips)


# # Create an initial scatter plot
# data_exp_scatter = data_exp.scatter(x='x', y='y', color='class_color', source=data_exp_source, legend_field='class_label')

# Create a figure
# data_exp = figure(title="Data Exploration: search for correlations between properties", width = 800, height = 520, x_axis_label='X', y_axis_label='Y', 
#            tools="pan,wheel_zoom,box_zoom,reset,save", tooltips = tooltips)


# Create an initial scatter plot
# data_exp_scatter = data_exp.scatter(x='x', y='y', color='class_color', source=data_exp_source, legend_field='class_label', size = 2, alpha = 0.6)

# # legend
# data_exp.add_layout(data_exp.legend[0], 'right')

# # Create dropdown menus for X and Y axis
# select_x = Select(title="X Axis", value=data_exp_columns[0], options=data_exp_columns)
# select_y = Select(title="Y Axis", value=data_exp_columns[1], options=data_exp_columns)

# # Update the data based on the selections
# def update_data_exp(attrname, old, new):
#     x = select_x.value
#     y = select_y.value
#     new_vis_data = {
#         'x': df[x],
#         'y': df[y],
#         'names' : df['Substance Name'],
#         'class_color': ['#900C3F' if cls == df['Class'].cat.categories[0] else '#1DBD4D' for cls in df['Class']],
#         'class_label': ['Not readily biodegradable' if cls == df['Class'].cat.categories[0] else 'Readily biodegradable' for cls in df['Class']]
#     }
        
#     # Update the ColumnDataSource with a plain Python dict
#     data_exp_source.data = new_vis_data
    
#     # Update existing scatter plot glyph if needed
#     data_exp_scatter.data_source.data = new_vis_data
    
#     data_exp.xaxis.axis_label = x
#     data_exp.yaxis.axis_label = y

# # Attach the update_data function to the dropdowns
# select_x.on_change('value', update_data_exp)
# select_y.on_change('value', update_data_exp)

# update_data_exp(None, None, None)

sizes = np.linspace(.01, 1.0, 15)
train_scores = []
val_scores = []
learning_curve_source = ColumnDataSource()

def set_lc_source():
    global learning_curve_source
    learning_curve_source.data=dict(
        train_size=sizes,
        train_score=train_scores,
        val_score=val_scores
    )
set_lc_source()

learning_curve = figure(
    title='Learning Curve with Custom Split', 
    x_axis_label='Training Size (Fraction)', 
    y_axis_label='Accuracy',
    tools='pan, wheel_zoom, box_zoom, reset',
    x_range=(0, 1),
    y_range=(0, 1),  # Set y-axis range from 0 to 1
    height=500
)

curve1 = learning_curve.line('train_size', 'train_score', source=learning_curve_source, line_width=2, legend_label='Training Score', color='blue')
curve2 = learning_curve.scatter('train_size', 'train_score', source=learning_curve_source, size=8, color='blue')

curve3 = learning_curve.line('train_size', 'val_score', source=learning_curve_source, line_width=2, legend_label='Validation Score', color='orange')
curve4 = learning_curve.scatter('train_size', 'val_score', source=learning_curve_source, size=8, color='orange')

learning_curve_hover = HoverTool(tooltips = [
    ("Training Size", "@train_size"),
    ("Training Score", "@train_score"),
    ("Val Score", "@val_score")
])
learning_curve.add_tools(learning_curve_hover)
learning_curve.legend.location = 'bottom_right'

# --------------- ALGORITHM SELECT AND RUN ---------------
# algorithm name holder
my_alg = 'Decision Tree'

# Create select button
alg_select = Select(title="Select ML Algorithm:", value="Decision Tree", options=["Decision Tree", "K-Nearest Neighbor", "Support Vector Classification"])

# define to be default: decision tree
hyperparam_list = [2, "random"]

def print_vals():
    global my_alg, model, hyperparam_list
    print("hp", hyperparam_list)
    print("model", model)
    if my_alg == 'Decision Tree':
        print("depth", model.max_depth)
        print("splitter", model.splitter)
    elif my_alg == 'K-Nearest Neighbor':
        print("n_neighbors", model.n_neighbors)
        print("weights", model.weights)
    elif my_alg == 'Support Vector Classification':
        print("C", model.C)
        print("kernel", model.kernel)

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
    title = "Splitter strategy:",
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

def set_hyperparameter_widgets():
    global model
    global my_alg
    global hyperparam_list
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

        model.n_neighbors = hp_slider.value
        model.weights = hp_select.value
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
            value = 20,
            step = 1
        )
        hp_toggle.visible = False
        hp_select.update(
            title = "kernel",
            value = "linear",
            options = ["linear", "poly", "rbf", "sigmoid"]
        )

        model.C = hp_slider.value
        model.kernel = hp_select.value


# list of the models to use
np.random.seed(123)
# model_list = [DecisionTreeClassifier(), KNeighborsClassifier(), SVC()]
# model = model_list[0]
model_list = []
model = DecisionTreeClassifier()
# model_list.append(model)  #temp solution, trying to store all trained models to access for testing later
set_hyperparameter_widgets()

def update_algorithm(attr, old, new):
    global my_alg, model
    my_alg = new

    # Assigning model based on selected ML algorithm, using default hyperparameters
    if my_alg == "Decision Tree":
        # model = model_list[0]
        model = DecisionTreeClassifier()
    elif my_alg == "K-Nearest Neighbor":
        # model = model_list[1]
        model = KNeighborsClassifier()
    else:
        # model = model_list[2]
        model = SVC()
    set_hyperparameter_widgets()
    train_status_message.text = 'Not running'
    train_status_message.styles = not_updated

# creating widgets
test_accuracy = 0.0

# Attach callback to Select widget
alg_select.on_change('value', update_algorithm)
# alg_select.on_change('value', set_hyperparameter_widgets)


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
    # set_hyperparameter_widgets()
    train_validate_model()

def split_data(train_percentage, val_percentage, test_percentage, data_index):
    global X_train, X_val, X_test, y_train, y_val, y_test

    temp_df = all_df[data_index]
    temp_cols = all_cols[data_index]
    #print(temp_df)
    

    X = temp_df[temp_cols]
    y = df['Class']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(100-train_percentage)/100)
    test_split = test_percentage / (100-train_percentage)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_split)

def train_validate_model():
    global model
    global train_scores, val_scores
    train_scores.clear()
    val_scores.clear()

    sizes = np.linspace(.01, 1.0, 15)
    train_scores.append(0)
    val_scores.append(0)
    if my_alg == "K-Nearest Neighbor":
        if X_train.shape[0]*.01 < model.n_neighbors:
            sizes = np.linspace(.025, 1.0, 15)
        if X_train.shape[0]*.025 < model.n_neighbors:
            sizes = np.linspace(.05, 1.0, 15)
        if X_train.shape[0]*.05 < model.n_neighbors:
            sizes = np.linspace(.075, 1.0, 15)
 
    for size in sizes:
        np.random.seed(123)
        X_train_subset = X_train.sample(frac=size, random_state=42)
        y_train_subset = y_train.loc[X_train_subset.index]

        # train model
        model.fit(X_train_subset, y_train_subset)

        train_score = accuracy_score(y_train_subset, model.predict(X_train_subset)) # Evaluate on training and test sets
        val_score = accuracy_score(y_val, model.predict(X_val))
        
        train_scores.append(train_score) #add the scores to the arrays 
        val_scores.append(val_score)

    val_accuracy.append(round(val_scores[-1], 3))

    set_learning_curve()
    save_model()
    model_list.append(model)

def load_ML():
    train_status_message.text = f'Running {my_alg}...'
    train_status_message.styles = loading

    tune_status_message.text='Not running'
    tune_status_message.styles=not_updated

    curdoc().add_next_tick_callback(run_ML)

# Attach callback to the run button
train_button.on_click(load_ML)

def set_learning_curve():
    set_lc_source()
    curve1.data_source = learning_curve_source
    curve2.data_source = learning_curve_source
    curve3.data_source = learning_curve_source
    curve4.data_source = learning_curve_source

# --------------- HYPERPARAMETER TUNING + BUTTON ---------------

# a list of an int an string
## decision tree - int/nan, string
## KNN - int, string
## SVC - int, ""


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

# setting widget callbacks
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
    tune_status_message.text = "Not running"
    tune_status_message.styles = not_updated

def hp_select_callback(attr, old, new):
    global my_alg
    hyperparam_list[1] = new
    if my_alg == 'Decision Tree':
        model.splitter = new
    elif my_alg == 'K-Nearest Neighbor':
        model.weights = new
    elif my_alg == 'Support Vector Classification':
        model.kernel = new
    tune_status_message.text = "Not running"
    tune_status_message.styles = not_updated

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


hp_slider.on_change('value', hp_slider_callback)
hp_select.on_change('value', hp_select_callback)
hp_toggle.on_change('active', hp_toggle_callback)


def load_tuned_config():
    tune_status_message.text = "Loading tuned config..."
    tune_status_message.styles = loading
    
    curdoc().add_next_tick_callback(run_tuned_config)

# Can connect to the old funcs
tune_button.on_click(load_tuned_config)

# --------------- SAVE ---------------

# making select to choose save num to display/use
delete_multiselect = MultiSelect(title = "Choose saves to delete:", options = [], margin=(5, 40, 5, 5), width = 200)
test_save_select = Select(title = "Choose a save to test:", options = [], margin=(5, 40, 5, 5), width = 200)

def update_test_message(attr, old, new):
    temp_test_status_message.text = "Not running"
    temp_test_status_message.styles = not_updated

test_save_select.on_change('value', update_test_message)

predict_select = Select(title = 'Choose a save to predict with:', options = [])

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
    TableColumn(field="saved_data_choice", title="Data"),
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

    new_saved_val_acc = val_accuracy[new_save_number] # access before save num is incremented

    new_save_number += 1
    test_save_select.options.append(str(new_save_number))
    temp_test_status_message.text = 'Not running'
    temp_test_status_message.styles = not_updated
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
    

    new_formatted_val_html = html_val_template.format(f'{round((new_saved_val_acc*100), 1)}%')

    val_acc_display.text = new_formatted_val_html

    save_source.stream(new_saved_data)
    

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
true_pos = nan
false_pos = nan
false_neg = nan
true_neg = nan


scale = 4
confus_d = {'T_range': ['Positive', 'Positive',
                 'Negative', 'Negative'],
    'Subject': ['Negative', 'Positive',
                'Negative', 'Positive'],
    'count': [true_pos, false_pos,
            false_neg, true_neg],
    'count_scaled': [true_pos / scale, false_pos / scale,
            false_neg / scale, true_neg / scale],
    'title': ['False Negative', 'True Positive', 'True Negative', 'False Positive'],
    'color': ['#FF7F50', '#6495ED', '#6495ED', '#FF7F50']     
           }

confus_df = pd.DataFrame(data = confus_d)
confus_source = ColumnDataSource(confus_df)
bubble = figure(x_range = confus_df['T_range'].unique(), y_range = confus_df['Subject'].unique(), 
                title = 'Confusion Matrix', width = 325, height = 300, tools='', toolbar_location = None)

# color_mapper = LinearColorMapper(palette = Viridis256, low = confus_df['count'].min(), high = confus_df['count'].max())
# color_bar = ColorBar(color_mapper = color_mapper,
#                     location = (0, 0),
#                     ticker = BasicTicker())
# bubble.add_layout(color_bar, 'right')
bubble.scatter(x = 'T_range', y = 'Subject', size = 'count_scaled', color = 'color', source = confus_source)
bubble.grid.visible = False
bubble.xaxis.axis_label = 'Actual Values'
bubble.yaxis.axis_label = 'Predicted Values'
bubble.add_tools(HoverTool(tooltips = [('Type', '@title'), ('Count', '@count')]))

cmatrix = figure(title = "Confusion Matrix", x_range = (-1,1), y_range = (-1,1))

def update_cmatrix(attrname, old, new):
    new_confus_d = {'T_range': ['Positive', 'Positive',
                    'Negative', 'Negative'],
                    'Subject': ['Negative', 'Positive',
                                'Negative', 'Positive'],
                    'count': [new_false_neg, new_true_pos, 
                            new_true_neg, new_false_pos],
                    'count_scaled': [new_false_neg / scale, new_true_pos / scale,
                            new_true_neg / scale, new_false_pos / scale],
                    'title': ['False Negative', 'True Positive', 'True Negative', 'False Positive'],
                    'color': ['#FF7F50', '#6495ED', '#6495ED', '#FF7F50']    
           }
    
    precision_1 = new_true_pos/(new_true_pos + new_false_pos)

    precision_2 = new_true_neg/(new_true_neg + new_false_neg)

    if precision_1 <= 0.50:
        performance_1 = 'POOR'
    elif precision_1 > 0.50 and precision_1 < 0.75:
        performance_1 = 'FAIR'
    else:
        performance_1 = 'EXCELLENT'

    if precision_2 <= 0.50:
        performance_2 = 'POOR'
    elif precision_2 > 0.50 and precision_2 < 0.75:
        performance_2 = 'FAIR'
    else:
        performance_2 = 'EXCELLENT'
    
    if test_accuracy <= 0.50:
        performance_3 = 'POOR'
    elif test_accuracy > 0.50 and test_accuracy < 0.75:
        performance_3 = 'FAIR'
    else:
        performance_3 = 'EXCELLENT'

    new_formatted_test_html = html_test_template.format(f'{round((test_accuracy*100), 1)}%', new_true_pos, new_false_pos, new_false_neg, new_true_neg, f'{round((precision_1*100), 1)}%', f'{round((precision_2*100), 1)}%', performance_1, performance_2, performance_3)
    test_acc_display.text = new_formatted_test_html

    # Update the ColumnDataSource

    confus_source.data = new_confus_d

    # new_color_mapper = LinearColorMapper(palette = Viridis256, low = min(new_confus_d['count']), high = max(new_confus_d['count']))
    
    # color_bar.color_mapper = new_color_mapper

    # bubble.scatter(fill_color = transform('count', new_color_mapper)

# --------------- NEW DATA TABLE ----------------

indices = []
tested_names = []
tested_smiles = []
predicted = []
actual = []
tfpn = []

test_cols = ['Index', 'Substance Name', 'Smiles', 'Predicted Class', 'Actual Class', 'Prediction Type']
test_tab_columns = [TableColumn(field=col, title=col, width=110) for col in test_cols]

test_table_data = {'Index': indices,
            'Substance Name': tested_names,
            'Smiles': tested_smiles, 
            'Predicted Class': predicted,
            'Actual Class': actual,
            'Prediction Type': tfpn}
new_source = ColumnDataSource(data=test_table_data)
abridg_source = ColumnDataSource(data=test_table_data)
new_table = DataTable(source=abridg_source, columns=test_tab_columns, width = 660, height_policy = 'auto', autosize_mode = "none", index_position=None)


# Testing model, and updating confusion matrix and table
def train_test_model():
    global new_true_pos
    global new_false_pos
    global new_false_neg
    global new_true_neg

    global tested_names
    global tested_smiles
    global indices
    global predicted
    global actual
    global tfpn

    np.random.seed(123)

    global save_index
    save_num = int(test_save_select.value)
    save_index = test_save_select.options.index(str(save_num))
    temp_split = [int(split) for split in save_source.data['train_val_test_split'][save_index].split("/")]
    temp_data_choice = save_source.data['saved_data_choice'][save_index]
    temp_data_index = data_opts.index(temp_data_choice)
    temp_alg = save_source.data['saved_algorithm'][save_index]
    temp_hyperparams = eval(save_source.data['saved_hyperparams'][save_index])

    split_data(temp_split[0], temp_split[1], temp_split[2],temp_data_index)

    global model
    model = model_list[save_index]
    if temp_alg == 'DT':
        # model = model_list[0]
        model.max_depth = temp_hyperparams[0]
        model.splitter = temp_hyperparams[1]
    elif temp_alg == 'KNN':
        # model = model_list[1]
        model.n_neighbors = temp_hyperparams[0]
        model.weights = temp_hyperparams[1]
    elif temp_alg == 'SVC':
        # model = model_list[2]
        model.C = temp_hyperparams[0]
        model.kernel = temp_hyperparams[1]

    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)

    indices = list(X_test.index)
    predicted = list(y_test_pred)

    actual.clear()
    for index in indices:
            actual.append(df1_dict['Class'][index])

    tested_names.clear()
    for index in indices:
        tested_names.append(df1_dict['Substance Name'][index])
    
    tested_smiles.clear()
    for index in indices:
        tested_smiles.append(df1_dict['Smiles'][index])


    tfpn.clear()
    for i in range(len(predicted)):
        if predicted[i] == actual[i] and actual[i] == 1:
            tfpn.append('True Positive')
        elif predicted[i] == actual[i] and actual[i] == 0:
            tfpn.append('True Negative')
        elif predicted[i] == 1:
            tfpn.append('False Positive')
        else:
            tfpn.append('False Negative')

    # print(len(indices))
    # print(len(predicted))
    # print(len(actual))
    # print(len(tested_names))
    # print(len(tested_smiles))
    # print(len(tfpn))

    full_test_table_data = {'Index': indices,
                'Substance Name': tested_names,
                'Smiles': tested_smiles, 
                'Predicted Class': predicted,
                'Actual Class': actual,
                'Prediction Type': tfpn}
    
    abridg_test_table_data = {'Index': indices[:15],
                'Substance Name': tested_names[:15],
                'Smiles': tested_smiles[:15], 
                'Predicted Class': predicted[:15],
                'Actual Class': actual[:15],
                'Prediction Type': tfpn[:15]}

    new_source.data=full_test_table_data
    abridg_source.data=abridg_test_table_data

    confusion_values = confusion_matrix(y_test, y_test_pred)

    new_true_pos = confusion_values[0][0]
    new_false_pos = confusion_values[0][1]
    new_false_neg = confusion_values[1][0]
    new_true_neg = confusion_values[1][1]
    # print(new_true_pos)
    # print(new_false_pos)
    # print(new_false_neg)
    # print(new_true_neg)

    global test_accuracy
    test_accuracy=round(accuracy_score(y_test, y_test_pred), 3)

    update_cmatrix(None, None, None)

def run_test():
    global my_alg, stage
    stage = 'Test'
    global model

    train_test_model()

    # Updating accuracy display

    temp_test_status_message.text = "Testing complete"
    temp_test_status_message.styles = updated

def load_test():
    temp_test_status_message.text = "Testing..."
    temp_test_status_message.styles = loading
    
    curdoc().add_next_tick_callback(run_test)

test_button.on_click(load_test)

# --------------- EXPORTING FULL TABLE TO XLSX OR CSV (80% of this is courtesy of ChatGPT) ---------------------------
def download_xlsx():
    # Convert source into df
    tested_df = pd.DataFrame(new_source.data)

    # Create an Excel buffer
    excel_buffer = BytesIO()
    
    # Write the DataFrame to the buffer using ExcelWriter
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        tested_df.to_excel(writer, index=False, sheet_name='Sheet1')
    
    # Get the binary data from the buffer
    excel_data = excel_buffer.getvalue()
    
    # Encode the binary data to base64
    b64_excel_data = base64.b64encode(excel_data).decode()
    
    # Define the filename
    filename = "tested_data.xlsx"

    js_download_excel = f"""
    var filename = "{filename}";
    var filetext = atob("{b64_excel_data}");

    var buffer = new Uint8Array(filetext.length);
    for (var i = 0; i < filetext.length; i++) {{
        buffer[i] = filetext.charCodeAt(i);
    }}
    
    var blob = new Blob([buffer], {{"type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}});
    
    // Create a link element
    var link = document.createElement("a");
    
    // Set the href to the Blob URL
    link.href = URL.createObjectURL(blob);
    
    // Set the download attribute
    link.download = filename;
    
    // Append the link to the body
    document.body.appendChild(link);
    
    // Click the link to trigger the download
    link.click();
    
    // Remove the link from the document
    document.body.removeChild(link);
    """
    # Create a CustomJS object with the JavaScript code
    xlsx_custom_js = CustomJS(args=dict(), code=js_download_excel)
    
    # Attach the CustomJS to the button click event
    export_excel.js_on_click(xlsx_custom_js)

export_excel.on_click(download_xlsx)


def download_csv():
    # Convert source into df
    tested_df = pd.DataFrame(new_source.data)

    # Create a CSV buffer
    csv_buffer = io.StringIO()
    
    # Write the DataFrame to the buffer
    tested_df.to_csv(csv_buffer, index=False)
    
    # Get the CSV data as a string
    csv_data = csv_buffer.getvalue()
    
    # Define the filename
    filename = "tested_data.csv"

    js_download = f"""
    var filename = "{filename}";
    var filetext = `{csv_data}`;
    
    var blob = new Blob([filetext], {{"type": "text/csv;charset=utf-8;"}});
    
    // Create a link element
    var link = document.createElement("a");
    
    // Set the href to the Blob URL
    link.href = URL.createObjectURL(blob);
    
    // Set the download attribute
    link.download = filename;
    
    // Append the link to the body
    document.body.appendChild(link);
    
    // Click the link to trigger the download
    link.click();
    
    // Remove the link from the document
    document.body.removeChild(link);
    """
    # Create a CustomJS object with the JavaScript code
    csv_custom_js = CustomJS(args=dict(), code=js_download)
    
    # Attach the CustomJS to the button click event
    export_csv.js_on_click(csv_custom_js)

export_csv.on_click(download_csv)


# --------------- PREDICTING ---------------


random_smiles = random.choices(df1_dict['Smiles'], k=3)

smiles_select = Select(title="Select Smiles String", value=random_smiles[0], options=[random_smiles[0], random_smiles[1], random_smiles[2], "Custom"], width=200)

user_smiles_input = TextInput(title = 'Enter a SMILES string:', width=200)

# test in dataset C=C(C)C(=O)O

def predict_biodegrad():
    global model, save_index
#     temp_tvt_list = new_train_val_test_split.split("/")
#     temp_train = int(temp_tvt_list[0])
#     temp_val = int(temp_tvt_list[1])
#     temp_test = int(temp_tvt_list[2])
#     temp_columns = save_source.data['saved_columns'][int(predict_select.value)-1]

#     train_validate_model(temp_train,temp_val,temp_test, temp_columns)

    if smiles_select.value != "Custom":
        user_smiles = smiles_select.value
        user_molec = Chem.MolFromSmiles(user_smiles)
    else:
        user_smiles = user_smiles_input.value
        user_molec = Chem.MolFromSmiles(user_smiles)

        if user_molec == None:
            predict_status_message.text = 'Error: invalid Smiles string'
            predict_status_message.styles = not_updated
            return
        
    user_compound = pubchempy.get_compounds(user_smiles, namespace='smiles')

    user_name = user_compound[0].iupac_name

    def molecule_to_descriptors(mol):
        desc = Descriptors.CalcMolDescriptors(mol)
        desc_df = pd.DataFrame([desc])
        X_pred = desc_df.drop(columns=['MaxPartialCharge', 'MaxAbsPartialCharge', 'Ipc', 'MinPartialCharge', 'MinAbsPartialCharge', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW'])
        y_pred = model.predict(X_pred)
        return y_pred

    # from 2016 rdkit ugm github
    def molecule_to_morgan(mol):
        a = np.zeros(2048)
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, radius=1), a)
        X_pred = pd.DataFrame([a])
        y_pred = model.predict(X_pred)
        return y_pred

    def molecule_to_ecfp(mol):
        a = np.zeros(2048)
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2), a)
        X_pred = pd.DataFrame([a])
        y_pred = model.predict(X_pred)
        return y_pred

    def molecule_to_pathfp(mol):
        a = np.zeros(2048)
        DataStructs.ConvertToNumpyArray(Chem.RDKFingerprint(mol, maxPath=2), a)
        X_pred = pd.DataFrame([a])
        y_pred = model.predict(X_pred)
        return y_pred
    
    if save_source.data['saved_data_choice'][save_index] == data_opts[0]:
        y_pred = molecule_to_descriptors(user_molec)

    elif save_source.data['saved_data_choice'][save_index] == data_opts[1]:
        y_pred = molecule_to_morgan(user_molec)

    elif save_source.data['saved_data_choice'][save_index] == data_opts[2]:
        y_pred = molecule_to_ecfp(user_molec)
    elif save_source.data['saved_data_choice'][save_index] == data_opts[3]:
        y_pred = molecule_to_pathfp(user_molec)
        

    if user_smiles in df1_dict['Smiles']:
        known_index = df1_dict['Smiles'].index(user_smiles)
        actual_class = df1_dict['Class'][known_index]
    else:
        actual_class = 'Unknown'

    predict_status_message.text = 'Complete'
    predict_status_message.styles = updated

    new_formatted_predict_html = html_predict_template.format(user_name, user_smiles, y_pred, actual_class)
    predict_display.text = new_formatted_predict_html

    return

def load_predict():
     predict_status_message.text = 'Predicting...'
     predict_status_message.styles = loading
     curdoc().add_next_tick_callback(predict_biodegrad)

 # callback for predict button
predict_button.on_click(load_predict)

def update_predict_status(attr, old, new):
    predict_status_message.text = 'Not running'
    predict_status_message.styles = not_updated

smiles_select.on_change('value', update_predict_status)


# ---------------- VISIBILITY --------------

# # Data exploration plot
# datavis_help.visible = False
# data_exp.visible = False
# select_x.visible = False
# select_y.visible = False

# # Callback function to toggle visibility
# def toggle_data_exp_visibility():
#     datavis_help.visible = not datavis_help.visible
#     data_exp.visible = not data_exp.visible
#     select_x.visible = not select_x.visible
#     select_y.visible = not select_y.visible
#     data_exp_vis_button.label = "Show Data Exploration*" if not data_exp.visible else "Hide Data Exploration*"
#     data_exp_vis_button.icon = down_arrow if not data_exp.visible else up_arrow

# # Link the button to the callback
# data_exp_vis_button.on_click(toggle_data_exp_visibility)

# Custom Smiles String input

user_smiles_input.visible = False
predict_instr.visible = False

def toggle_smiles_input_vis(attr, old, new):
    if smiles_select.value == 'Custom':
        user_smiles_input.visible = True
        predict_instr.visible = True
    else:
        user_smiles_input.visible = False
        predict_instr.visible = False

smiles_select.on_change('value', toggle_smiles_input_vis)


# --------------- LAYOUTS ---------------

tiny_height_spacer = Spacer(height = 15)
small_height_spacer = Spacer(height = 16)
small_med_height_spacer = Spacer(height = 23)
med_height_spacer = Spacer(height = 30)
large_height_spacer = Spacer(height = 45)
ginormous_height_spacer = Spacer(height = 60)
button_spacer = Spacer(height = 30, width = 54)
top_page_spacer = Spacer(height = 20)
left_page_spacer = Spacer(width = 20)
large_left_page_spacer = Spacer(width = 90)

# creating widget layouts
tab0_layout = row(left_page_spacer, column(top_page_spacer, intro_instr))

data_config_layout = layout(
    [data_select, column(small_height_spacer, datatable_help)],
    [tiny_height_spacer],
    [column(row(tvt_slider, column(small_med_height_spacer, splitter_help)), split_display)],
    [tiny_height_spacer],
    [column(save_config_button, save_config_message)]
)

# interactive_graph = column(data_exp_vis_button, row(datavis_help, column(data_exp, row(select_x, select_y)))) #create data graph visualization 
tab1_layout = row(left_page_spacer, column(top_page_spacer, row(data_config_layout, data_tab_table), tiny_height_spacer)) #interactive_graph

# interactive_graph = column(row(data_exp_vis_button, datavis_help), data_exp, row(select_x, select_y)) #create data graph visualization 
# tab1_layout = row(left_page_spacer, column(top_page_spacer, row(data_tab_table, data_config_layout), tiny_height_spacer, interactive_graph))


hyperparam_layout = layout(
    [hp_slider],
    [hp_toggle],
    [hp_select],
    [tune_button, tune_help],
    [tune_status_message],
    [ginormous_height_spacer]
)

delete_layout = layout(
    [delete_multiselect],
    [delete_button],
    [delete_status_message]
)

tab2_layout = row(left_page_spacer, column(top_page_spacer, alg_select, row(train_button, train_help), train_status_message, ginormous_height_spacer, hyperparam_layout, delete_layout), large_left_page_spacer, column(learning_curve, saved_data_table), column(top_page_spacer, val_acc_display))

# save_layout = row(column(test_save_select, display_save_button), saved_data_table)

test_button_layout = layout(
    [column(test_save_select, row(test_button, test_help), temp_test_status_message, ginormous_height_spacer, export_excel, export_csv)]
)

tab3_layout = row(left_page_spacer, column(top_page_spacer, row(column(row(test_button_layout, large_left_page_spacer, bubble), new_table), column(small_med_height_spacer, test_acc_display))))

tab4_layout = row(left_page_spacer, column(top_page_spacer, smiles_select, user_smiles_input, predict_instr, predict_button, predict_status_message), column(top_page_spacer, predict_display))

tabs = Tabs(tabs = [TabPanel(child = tab0_layout, title = 'Instructions'),
                    TabPanel(child = tab1_layout, title = 'Data'),
                    TabPanel(child = tab2_layout, title = 'Train and Validate'),
                    TabPanel(child = tab3_layout, title = 'Test'),
                    TabPanel(child = tab4_layout, title = 'Predict')
                ])

curdoc().add_root(tabs)