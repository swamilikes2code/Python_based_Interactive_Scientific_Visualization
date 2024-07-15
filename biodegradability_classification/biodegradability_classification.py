import pandas as pd
import numpy as np
from io import BytesIO
import os
import random
import base64
from math import nan

from bokeh.io import curdoc
from bokeh.layouts import column, row, Spacer, layout
from bokeh.models import Div, ColumnDataSource, DataTable, TableColumn, Button, RangeSlider, Select, Slider, Checkbox, Tabs, TabPanel, TextInput, PreText, HelpButton, Tooltip, MultiSelect, HoverTool
from bokeh.models.callbacks import CustomJS
from bokeh.models.dom import HTML
from bokeh.models.ui import SVGIcon
from bokeh.plotting import figure
from bokeh.transform import dodge
import pubchempy
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs, Descriptors, AllChem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.exceptions import ConvergenceWarning
from bokeh.util.warnings import BokehUserWarning, warnings
from datetime import datetime

#entire code timer
start_time = datetime.now()
warnings.simplefilter(action='ignore', category=BokehUserWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)


# --------------- MESSAGE STYLES --------------- #

header = {'color': 'black', 'font-size': '18px'}
not_updated = {'color': 'red', 'font-size': '14px'}
loading = {'color': 'orange', 'font-size': '14px'}
updated = {'color': 'green', 'font-size': '14px'}
completed = {'color': 'black', 'font-size': '14px'}

# --------------- ACCURACY LISTS --------------- #
# declare at the top to use everywhere
val_accuracy = []

# --------------- STATUS MESSAGES --------------- #

step_one = Div(text='<b>1) PREPARE DATA</b>', styles=header)
step_two = Div(text='<b>2) TRAIN</b>', styles=header)
step_three = Div(text='<b>3) VALIDATE</b>', styles=header)
step_four = Div(text='<b>4) TEST</b>', styles=header)
step_five = Div(text='<b>5) PREDICT</b>', styles=header)

save_config_message = Div(text='Configuration not saved', styles=not_updated)
train_status_message = Div(text='Not running', styles=not_updated)
tune_status_message = Div(text='Not running', styles=not_updated)
temp_test_status_message = Div(text='Not running', styles=not_updated)
predict_status_message = Div(text = 'Not running', styles=not_updated)
delete_status_message = Div(text='Changes not saved', styles = not_updated)

# --------------- BUTTONS --------------- #

save_config_button = Button(label="Save Current Configuration", button_type="warning", width = 250)
train_button = Button(label="Run ML algorithm", button_type="success", width=150, height = 31)
tune_button = Button(label="Tune", button_type="success", width=150, height = 31)
delete_button = Button(label = "Delete", button_type = 'danger', width = 200, height = 31)
test_button = Button(label = "Test", button_type = "success", width = 150, height = 31)
predict_button = Button(label = 'Predict', button_type = "success", width = 200, height = 31)

#svg icons for buttons
up_arrow = SVGIcon(svg = '''<svg  xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-chevron-up"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M6 15l6 -6l6 6" /></svg>''')
down_arrow = SVGIcon(svg = '''<svg  xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-chevron-down"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M6 9l6 6l6 -6" /></svg>''')

# still calling it data exploration for now instead of "Show Histogram" as it's less descriptive
hist_visibility_button = Button(label="Show Data Exploration", button_type="primary", icon = down_arrow)

export_excel = Button(label="Download Full Table to Excel (.xlsx)", width=200, height=31)
export_csv = Button(label="Download Full Table to CSV (.csv)", width=200, height=31)

# --------------- HTML TEMPLATES --------------- #
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

        <div class="section">
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
        </div>

        <div class="section">
            <div class="row">
                <h2>Precisions:</h2>
                <div class="column_2">
                    <h3>Ready Biodegradability:</h3>
                    <p>{}</p>
                </div>
                <div class="column_2">
                    <h3>Non-ready Biodegradability:</h3>
                    <p>{}</p>
                </div>
            </div>
        </div>


        <div class="row">
            <h2>Performance:</h2>
            <p>On average, your model was <b>{}</b> at classifying ready biodegradabiliy, and <b>{}</b> at classifying non-ready biodegradability,
            resulting in an overall <b>{}</b> performance.</p>
        </div>

        <div class="row">
            <h2>NOTE:</h2>
            <p><b>Precision for Ready Biodegradability:</b> true positives/(true postives + false positives)--> Percentage of predicted positives
              that were true (actual) positives.</p>
            <p><b>Precision for Non-ready Biodegradability:</b> true negatives/(true negatives + false negatives)--> Percentage of predicted negatives
              that were true (actual) negatives.</p>
            <p><b>Testing accuracy:</b> Average of the two precision values.</p>
            <p><b>INEFFECTIVE</b> performance: accuracy score < 70%</p> 
            <p><b>EFFECTIVE</b> performance: accuracy value > 70%</p> 
            <p>The industry standard for accuracy scores is between 70%-90%.</p>
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
            <h2>SMILES String:</h2>
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

html_warning_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            background-color: #f4f4f4;
        }}
        .container {{
            background-color: #ffffff;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        h2 {{
            color: #444;
            padding-bottom: 5px;
        }}
        p {{
            margin: 15px 0;
        }}
        .section {{
            margin-bottom: 20px;
            padding: 10px;
            background-color: #def2f1;
        }}
        .highlight {{
            background-color: #e7f3fe;
            padding: 2px 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="section">
            <h2><i>Incomplete Step(s):</i></h2>
            <h2>{}</h2>
            <p>Please {} before continuing</p>
        </div>
    </div>
</body>
</html>
"""

# --------------- INSTRUCTIONS --------------- #
intro_instr_template = """
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
            border: 1px solid #ddd;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}

        h1 {{
            color: #333;
            text-align: center;
        }}

        h2 {{
            color: #444;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
        }}

        h3 {{
            color: #555;
            padding-bottom: 5px;
        }}

        .row {{
            display: flex;
            text-align: center;
            padding-bottom: 10px
        }}

        .column_5 {{
            flex: 1;
            padding: 5px;
        }}

        p {{
            margin: 15px 0;
            text-align: left;
        }}
        
        .section {{
            border-radius: 5px;
        }}

        .highlight {{
            background-color: #e7f3fe;
            border-left: 5px solid #2196F3;
            padding: 2px 5px;
            text-align: center;
        }}

        .full-width {{
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
        }}

        .center-text {{
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="section">
        <div class="row">
            <div style="background-color: {bg_color_1}; padding: 5px; border: 1px solid #ddd; border-radius: 7px;">
                <div class="column_5">
                    <h2>1) PREPARE DATA</h2>
                    <h3>Prepare the biodegradability data for training.</h3>
                    <p>On the <span class="highlight"><b>Data</b></span> tab, first choose whether to use <b>Molecular Features</b> or <b>Fingerprints</b> to train the model. Next, split the data into <b>training</b>, <b>validating</b>, and <b>testing</b>.</p> 
                    <p>You can also view a <b>Data Exploration Histogram</b>, showing the distribution of certain molecular features across the dataset.</p>
                </div>
            </div>

            <div style="background-color: #ffffff;">
                <div class="column_5">
                    <p>‎</p>
                </div>
            </div>


            <div style="background-color: {bg_color_2}; padding: 5px; border: 1px solid #ddd; border-radius: 7px;">
                <div class="column_5">
                    <h2>2) TRAIN</h2>
                    <h3>Train a machine learning model on your prepared data.</h3>
                    <p>On the <span class="highlight"><b>Train and Validate</b></span> tab, select the <b>machine learning algorithm</b> of your choice, and run it, displaying the run's <b>validation accuracy</b> in both a datatable, and a <b>Learning Curve</b>.</p>
                </div>
            </div>

            <div style="background-color: #ffffff;">
                <div class="column_5">
                    <p>‎</p>
                </div>
            </div>

            <div style="background-color: {bg_color_3}; padding: 5px; border: 1px solid #ddd; border-radius: 7px;">    
                <div class="column_5">
                    <h2>3) VALIDATE</h2>
                    <h3>Fine-tune the hyperparameters of your model.</h3>
                    <p>On the <span class="highlight"><b>Train and Validate</b></span> tab, fine-tune the algorithm's <b>hyperparameters</b>, and compare different runs' validation accuracies in the table, avoiding <b>overfitting</b> by looking at the Learning Curves. </p>
                </div>
            </div>

            <div style="background-color: #ffffff;">
                <div class="column_5">
                    <p>‎</p>
                </div>
            </div>
            
            <div style="background-color: {bg_color_4}; padding: 5px; border: 1px solid #ddd; border-radius: 7px;">
                <div class="column_5">
                    <h2>4) TEST</h2>
                    <h3>Perform a final test of your model's performance.</h3>
                    <p>On the <span class="highlight"><b>Test</b></span> tab complete your final test of the saved model of your choice, displaying its testing accuracy, and a <b>confusion matrix</b>.</p> 
                    <p>It is recommended to use your run with the highest validation accuracy here.</p>
                </div>
            </div>

            <div style="background-color: #ffffff;">
                <div class="column_5">
                    <p>‎</p>
                </div>
            </div>
            
            <div style="background-color: {bg_color_5}; padding: 5px; border: 1px solid #ddd; border-radius: 7px;">
                <div class="column_5">
                    <h2>5) PREDICT</h2>
                    <h3>Input a SMILES string and predict its class using your model.</h3>
                    <p>On the <span class="highlight"><b>Predict</b></span> tab, test any of the saved models by inputting a <b>SMILES string</b>, displaying the IUPAC name of your chosen molecule, its predicted class, and if the molecule appears in the dataset, its actual class.</p>
                </div>
            </div>
        </div>
    </div>

    <div class="section full-width"> 
    <div style="background-color: {bg_color_6}; padding: 5px; border: 1px solid #ddd; border-radius: 7px; width: 100%;">
        <div class="center-text">    
            <h3>{text}</h3>
            </div>
    </div>
</div>
</body>
</html>
"""

formatted_instr_html = intro_instr_template.format(
    bg_color_1='#fafafa', 
    bg_color_2='#fafafa', 
    bg_color_3='#fafafa', 
    bg_color_4='#fafafa', 
    bg_color_5='#fafafa',
    bg_color_6='#fafafa',
    text='Explore the module to create your own Machine Learning Model.'
)

intro_instr = Div(text=formatted_instr_html, width=1000, height=1000)

formatted_val_html = html_val_template.format('N/A')
val_acc_display = Div(text=formatted_val_html)

formatted_test_html = html_test_template.format('N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A')
test_acc_display = Div(text=formatted_test_html)

formatted_predict_html = html_predict_template.format('N/A', 'N/A', 'N/A', 'N/A')
predict_display = Div(text=formatted_predict_html)

step_two_warning_html = html_warning_template.format('1) Preparing Data', 'return to the <b>Data</b> tab and <b>Save Current Configuration</b>')
step_two_warning = Div(text=step_two_warning_html, width=200, height=200, visible=False)

step_three_warning_html = html_warning_template.format('2) Train', '<b>Run</b> current <b>ML Algorithm</b>')
step_three_warning = Div(text=step_three_warning_html, width=200, height=200, visible=False)

steps_four_five_warning_html = html_warning_template.format('2) Train', 'return to <b>Train and Validate</b> and <b>Run</b> at least one <b>ML Algorithm</b>')
step_four_warning = Div(text=steps_four_five_warning_html, width=200, height=200, visible=False)
step_five_warning = Div(text=steps_four_five_warning_html, width=200, height=200, visible=False)

'''
splitter_help = HelpButton(tooltip=Tooltip(content=Div(text="""
                 <div style='background-color: #DEF2F1; padding: 16px; font-family: Arial, sans-serif;'>
                 <div>Use this <b>slider</b> to split the data into <i>train/validate/test</i> percentages.</div>
                 <div>For more info, see <i>Preparing Data</i> in the above menu.</div>""", width=280), position="right"))

datatable_help = HelpButton(tooltip=Tooltip(content=Div(text="""
                 <div style='background-color: #DEF2F1; padding: 16px; font-family: Arial, sans-serif;'>
                 <div>Select whether to use <b>features</b> or a <b>molecular fingerprint</b> to train model.</div>
                                                        <div>For more info, see <i>Our Dataset</i> and <i>Fingerprints</i> in the above menu.</div>
                 </div>""", width=280), position="right"))

datavis_help = HelpButton(tooltip=Tooltip(content=Div(text="""
                 <div style='background-color: #DEF2F1; padding: 16px; font-family: Arial, sans-serif;'>
                 UPDATE THIS WHEN HISTOGRAMS ARE IN
                 </div>""", width=280), position="right"))

train_help = HelpButton(tooltip=Tooltip(content=Div(text="""
                  <div style='background-color: #DEF2F1; padding: 20px; font-family: Arial, sans-serif;'>
                  <div>Select one of the following <b>Machine Learning algorithms</b>.</div> 
                                                    <div>For more info, see <i>Algorithms</i> in the above menu.</div>
                  </div>""", width=280), position="right"))

tune_help = HelpButton(tooltip=Tooltip(content=Div(text="""
                 <div style='background-color: #DEF2F1; padding: 20px; font-family: Arial, sans-serif;'>
                 <div>Based on the ML algorithm chosen above, fine-tune its <b>hyperparameters</b> to improve the model's validation accuracy.
                                                   Use the <b>Learning Curve</b> to prevent <b>Overfitting.</b></div>
                                                   <div>For more info, see <i>Hyperparameters</i> and <i>Overfitting</i> in the above menu.</div>
                 </div>""", width=280), position="right"))

test_help = HelpButton(tooltip=Tooltip(content=Div(text="""
                <div style='background-color: #DEF2F1; padding: 20px; font-family: Arial, sans-serif;'>
                <div>Select the save from the previous tab to test the model, and view its <b>confusion matrix</b> below.</div>
                <div>‎</div>     
                <div>NOTE: This should be considered the <b>final</b> test of your model, and is NOT intended for additional validation.</div>
                </div>""", width=280), position="right"))
'''
predict_instr = Div(text="""
                 <div style='background-color: #DEF2F1; padding: 1px; font-family: Arial, sans-serif;'>
                    To create a SMILES String, visit
                    <a href="http://pubchem.ncbi.nlm.nih.gov//edit3/index.html" target="_blank">
                    pubchem.ncbi.nlm.nih.gov//edit3/index.html </a>
                    (Instructions on 'Help' button)
                 </div>""",
width=160, height=60)

asterisk = Div(text="""
                 <div style='background-color: #ffffff;'>
                    *Highest validation accuracy
                 </div>""",
width=200, height=20)

# --------------- UPDATE INSTRUCTIONS COLORS --------------- #
def update_color():
    bg_1 = '#fafafa'
    bg_2 = '#fafafa'
    bg_3 = '#fafafa'
    bg_4 = '#fafafa'
    bg_5 = '#fafafa'
    bg_6 = '#fafafa'
    t = 'Continue exploring the module.'


    if save_config_message.styles == updated:
        bg_1='#9fdbad'
    if train_status_message.styles == updated:
        bg_2='#9fdbad'
    if tune_status_message.styles == updated:
        bg_3='#9fdbad'
    if temp_test_status_message.styles == updated:
        bg_4='#9fdbad'
    if predict_status_message.styles == updated:
        bg_5='#9fdbad' 

    if step_two_warning.visible == True:
        bg_1='#ff7f7f'

    if step_three_warning.visible == True or step_four_warning.visible == True or step_five_warning.visible == True:
        bg_2='#ff7f7f'

    if bg_1 == bg_2 == bg_3 == bg_4 == bg_5 == '#9fdbad':
        t = 'Congratulations, you have completed the module! Feel free to continue exploring.'
        bg_6='#9fdbad' 

    if bg_1 == '#ff7f7f' or bg_2 == '#ff7f7f':
        t = 'Warning: You have incomplete step(s).'
        bg_6='#ff7f7f'


    new_formatted_instr_html = intro_instr_template.format(
        bg_color_1=bg_1, 
        bg_color_2=bg_2, 
        bg_color_3=bg_3, 
        bg_color_4=bg_4, 
        bg_color_5=bg_5,
        text = t,
        bg_color_6=bg_6
    )
    
    intro_instr.text = new_formatted_instr_html

###########################################################
# --------------- DATA LOAD AND SELECTION --------------- #
###########################################################

#for ref:
# df is original csv, holds fingerprint list and 167 cols of fingerprint bits
# df_display > df_subset > df_dict are for displaying table

total_data_section_timer_start = datetime.now()                         # ----------- TIMER CODE

read_csv_start = datetime.now()                                         # ----------- TIMER CODE

# toggle whether you are testing here or running from server
master = True
# master = False

####################################################################################################
# Load data from the csv file                        # ---- This section takes 1.5-2.5 to run ---- #
if master:
    df1 = pd.read_csv("biodegradability_classification/data/option_1.csv", low_memory=False, na_filter=False) # -------------------- #
    df2 = pd.read_csv("biodegradability_classification/data/option_2.csv", low_memory=False, na_filter=False) # -------------------- #
    df3 = pd.read_csv("biodegradability_classification/data/option_3.csv", low_memory=False, na_filter=False) # -------------------- #
    df4 = pd.read_csv("biodegradability_classification/data/option_4.csv", low_memory=False, na_filter=False) # -------------------- #
else:
    df1 = pd.read_csv("./data/option_1.csv", low_memory=False, na_filter=False) # -------------------- #
    df2 = pd.read_csv("./data/option_2.csv", low_memory=False, na_filter=False) # -------------------- #
    df3 = pd.read_csv("./data/option_3.csv", low_memory=False, na_filter=False) # -------------------- #
    df4 = pd.read_csv("./data/option_4.csv", low_memory=False, na_filter=False) # -------------------- #

dataset_size = len(df1)                              # ---- This section takes 1.5-2.5 to run ---- #
                                                     # ---- This section takes 1.5-2.5 to run ---- #
all_df = [df1, df2, df3, df4]                        # ---- This section takes 1.5-2.5 to run ---- #
                                                     # ---- This section takes 1.5-2.5 to run ---- #
# just holding mandatory cols                        # ---- This section takes 1.5-2.5 to run ---- #
df = df1.iloc[:, :4]                                 # ---- This section takes 1.5-2.5 to run ---- #
####################################################################################################
read_csv_stop = datetime.now()                                          # ----------- TIMER CODE
elapsed_time = read_csv_stop - read_csv_start                           # ----------- TIMER CODE
print(f"Reading in data: {elapsed_time.total_seconds():.2f} seconds") #lines 565 - 578 take 1.7-2.5 seconds

# Columns that should always be shown
mandatory_columns = ['Substance Name', 'SMILES', 'Class']

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

to_dictionary_time_start = datetime.now()                                   # ----------- TIMER CODE
####################################################################################################

def to_dict(df): 
    return {col: df[col].tolist() for col in df.columns} # sped up to 1 second execution

df1_dict = to_dict(df1)
df2_dict = to_dict(df2)
df3_dict = to_dict(df3)
df4_dict = to_dict(df4)
#df1_dict = df1.to_dict("list")                      # ---- This section takes 5.0-10.0 to run ---- #
#df2_dict = df2.to_dict("list")                      # ---- This section takes 5.5-10.0 to run ---- #
#df3_dict = df3.to_dict("list")                      # ---- This section takes 5.5-10.0 to run ---- #
#df4_dict = df4.to_dict("list")                      # ---- This section takes 5.5-10.0 to run ---- #
####################################################################################################
to_dictionary_time_end = datetime.now()                                     # ----------- TIMER CODE
elapsed_time = to_dictionary_time_end - to_dictionary_time_start            # ----------- TIMER CODE
print(f"Data to dictionary time: {elapsed_time.total_seconds():.2f} seconds") #606 - 615 takes 5-10 seconds

cols1 = [key for key in df1_dict.keys() if key not in mandatory_columns]
cols2 = [key for key in df2_dict.keys() if key not in mandatory_columns]
cols3 = [key for key in df3_dict.keys() if key not in mandatory_columns]
cols4= [key for key in df4_dict.keys() if key not in mandatory_columns]

all_cols = [cols1, cols2, cols3, cols4]

# Create figure
data_tab_columns = [TableColumn(field=col, title=col, width=150) for col in (mandatory_columns+cols1[:7])]
data_tab_table = DataTable(source=df1_tab_source, columns=data_tab_columns, width=1000, height_policy = 'auto', autosize_mode = "none")

data_select = Select(title="Select Features:", value = 'Molecular Properties', options=data_opts, width = 200)

data_initialization_end = datetime.now()                                    # ----------- TIMER CODE
elapsed_time = data_initialization_end - total_data_section_timer_start     # ----------- TIMER CODE
print(f"Entire Data Section Runtime: {elapsed_time.total_seconds():.2f} seconds")       # ----------- TIMER CODE


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

###################################################
# --------------- DATA SPLIT CODE --------------- #
###################################################

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

    update_color()

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

#########################################################
# --------------- SAVE DATA BUTTON CODE --------------- #
#########################################################

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

    update_color()

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

#############################################
# --------------- HISTOGRAM --------------- #
#############################################

# Split data based on the class - int == works!! tested in histogram.py
class_0 = df1[df1['Class'] == 0]
class_1 = df1[df1['Class'] == 1]

# Default histogram column
default_hist_column = 'MolWt'

# Define the bins
bins = np.linspace(df1[default_hist_column].min(), df1[default_hist_column].max(), 20)

# Calculate histogram for each class
hist_0, edges_0 = np.histogram(class_0[default_hist_column], bins=bins)
hist_1, edges_1 = np.histogram(class_1[default_hist_column], bins=bins)

# Prepare data for plotting

# Calculate the center positions of each bin
hist_centers = (edges_0[:-1] + edges_0[1:]) / 2
hist_width = .3*(hist_centers[1]-hist_centers[0])
dodge_val = .625*hist_width

# Create a new ColumnDataSource
source = ColumnDataSource(data=dict(
    hist_centers=hist_centers,
    top_class_0=hist_0,
    top_class_1=hist_1
))

# Create the figure
histogram = figure(title=f"Histogram of {default_hist_column} with Class Color Coding",
           x_axis_label=default_hist_column, y_axis_label='Frequency',
           tools="save",
           width=800, height=400)

# Add class 0 bars
bars_class_0 = histogram.vbar(x=dodge('hist_centers', -dodge_val, range=histogram.x_range), top='top_class_0', width=0.3*(hist_centers[1] - hist_centers[0]),
                      color='blue', alpha=0.6, legend_label='Class 0', source=source)

# Add class 1 bars
bars_class_1 = histogram.vbar(x=dodge('hist_centers', dodge_val, range=histogram.x_range), top='top_class_1', width=0.3*(hist_centers[1] - hist_centers[0]),
                      color='red', alpha=0.6, legend_label='Class 1', source=source)

# Add hover tool for interaction
hover = HoverTool()
hover.tooltips = [("Range", "@hist_centers"),
                  ("Class 0 Frequency", "@top_class_0"),
                  ("Class 1 Frequency", "@top_class_1")]
histogram.add_tools(hover)

# Style the plot
histogram.legend.click_policy = "hide"
histogram.legend.location = "top_right"
histogram.xgrid.grid_line_color = None
histogram.ygrid.grid_line_color = "gray"
histogram.ygrid.grid_line_dash = [6, 4]

# Create a Select widget for choosing histogram column
hist_options = ['Class',
                'MolWt',
                'NumValenceElectrons',
                'NumRadicalElectrons',
                'HeavyAtomCount',
                'NHOHCount',
                'NOCount',
                'NumAliphaticCarbocycles',
                'NumAliphaticHeterocycles',
                'NumAliphaticRings',
                'NumAromaticCarbocycles',
                'NumAromaticHeterocycles',
                'NumAromaticRings',
                'NumHAcceptors',
                'NumHDonors',
                'NumHeteroatoms',
                'NumRotatableBonds',
                'NumSaturatedCarbocycles',
                'NumSaturatedHeterocycles',
                'NumSaturatedRings',
                'RingCount',
                ]
hist_x_select = Select(title="X Axis:", value=default_hist_column, options=hist_options)

# Callback function for Select widget
def update_hist(attrname, old, new):
    selected_column = hist_x_select.value

    bins = np.linspace(df1[selected_column].min(), df1[selected_column].max(), 20)

    # Update histogram data based on selected column
    hist_0, edges_0 = np.histogram(class_0[selected_column], bins=bins)
    hist_1, edges_1 = np.histogram(class_1[selected_column], bins=bins)

    hist_centers = (edges_0[:-1] + edges_0[1:]) / 2
    hist_width = .3*(hist_centers[1]-hist_centers[0])
    dodge_val = .625*hist_width

    source.data = dict(
        hist_centers=hist_centers,
        top_class_0=hist_0,
        top_class_1=hist_1
    )
    bars_class_0.data_source.data['top'] = hist_0
    bars_class_1.data_source.data['top'] = hist_1

    # if selected_column == 'Class':
    #     bars_class_0.glyph.update(
    #         x = 0, width = .9
    #     )
    #     bars_class_1.glyph.update(
    #         x = 1, width = .9
    #     )
    #     histogram.x_range = (-.5, 1.5)
    #     histogram.xaxis.ticker = [0, 1]

    # else:
    bars_class_0.glyph.update(
        x=dodge('hist_centers', -dodge_val, range=histogram.x_range),
        width=hist_width
    )
    bars_class_1.glyph.update(
        x=dodge('hist_centers', dodge_val, range=histogram.x_range),
        width=hist_width
    )
    # histogram.x_range = None

    histogram.title.text = f"Histogram of {selected_column} with Class Color Coding"
    histogram.xaxis.axis_label = selected_column

# Attach callback to Select widget
hist_x_select.on_change('value', update_hist)


##################################################
# --------------- LEARNING CURVE --------------- # 
##################################################

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
    tools='wheel_zoom, box_zoom, reset, save',
    x_range=(0, 1),
    y_range=(0, 1),  # Set y-axis range from 0 to 1
    width=500,
    height=400
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

############################################################
# --------------- ALGORITHM SELECT AND RUN --------------- #
############################################################

# algorithm name holder
my_alg = 'Decision Tree'

# Create select button
alg_select = Select(title="Select ML Algorithm:", value="Decision Tree", options=["Decision Tree", "K-Nearest Neighbor", "Logistic Regression"], width=200)

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
    elif my_alg == 'Logistic Regression':
        print("C", model.C)
        print("solver", model.solver)

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
    elif my_alg == 'Logistic Regression':
        #hyperparameters are 
        # solver (lbfgs, liblinear, newton-cg, newton-cholesky, sag, saga) 
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
            title = "solver",
            value = "liblinear",
            options = ['lbfgs', 'liblinear', 'newton-cholesky', 'sag', 'saga']  #'newton-cg' took a long time to run
        )

        model.C = hp_slider.value
        model.solver = hp_select.value


# list of the models to use
np.random.seed(123)
# model_list = [DecisionTreeClassifier(), KNeighborsClassifier(), SVC()]
# model = model_list[0]
model_list = []
model = DecisionTreeClassifier()
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
        model = LogisticRegression()
    set_hyperparameter_widgets()
    train_status_message.text = 'Not running'
    train_status_message.styles = not_updated

    update_color()

# creating widgets
test_accuracy = 0.0

# Attach callback to Select widget
alg_select.on_change('value', update_algorithm)


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

    update_color()

    train_validate_model()

def split_data(train_percentage, val_percentage, test_percentage, data_index):
    global X_train, X_val, X_test, y_train, y_val, y_test

    temp_df = all_df[data_index]
    temp_cols = all_cols[data_index]    

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

    for opt in test_save_select.options:
        if '*' in opt:
            test_save_select.value = opt
            predict_select.value = opt

    # test_save_select.value = test_save_select.options[-1]
    # predict_select.value = predict_select.options[-1]
    # print(test_save_select.value)
    # print(predict_select.value)

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

##################################################################
# --------------- HYPERPARAMETER TUNING + BUTTON --------------- #
##################################################################

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

    update_color()

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
    elif my_alg == 'Logistic Regression':
        model.C = new
    tune_status_message.text = "Not running"
    tune_status_message.styles = not_updated

    update_color()

def hp_select_callback(attr, old, new):
    global my_alg
    hyperparam_list[1] = new
    if my_alg == 'Decision Tree':
        model.splitter = new
    elif my_alg == 'K-Nearest Neighbor':
        model.weights = new
    elif my_alg == 'Logistic Regression':
        model.solver = new
    tune_status_message.text = "Not running"
    tune_status_message.styles = not_updated

    update_color()

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

tune_button.on_click(load_tuned_config)


########################################
# --------------- SAVE --------------- #
########################################

# making select to choose save num to display/use
delete_multiselect = MultiSelect(title = "Choose saves to delete:", options = [], margin=(5, 40, 5, 5), width = 200)
test_save_select = Select(title = "Choose a save to test:", options = [], margin=(5, 40, 5, 5), width = 200, height = 40)

def update_test_message(attr, old, new):
    temp_test_status_message.text = "Not running"
    temp_test_status_message.styles = not_updated

    update_color()

test_save_select.on_change('value', update_test_message)

predict_select = Select(title = 'Choose a save to predict with:', options = [], width = 200, height = 40)

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

high_score = [-1, -1, 0]
old_high_score = [-1, -1, 0]

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
    delete_multiselect.options.append(str(new_save_number))
    predict_select.options.append(str(new_save_number))

    global high_score
    global old_high_score

    high_score = old_high_score[:]

    for i in test_save_select.options:
        current_index = test_save_select.options.index(str(i))
        test_save_select.options[current_index] = test_save_select.options[current_index].replace('*', '')
        # delete_multiselect.options[current_index] = delete_multiselect.options[current_index].replace('*', '')
        predict_select.options[current_index] = predict_select.options[current_index].replace('*', '')


        if val_accuracy[current_index] > high_score[2]:
            high_score.clear()
            high_score.append(int(test_save_select.options[current_index]))
            high_score.append(current_index)
            high_score.append(val_accuracy[int(test_save_select.options[current_index])-1])

    test_save_select.options[high_score[1]] = str(high_score[0]) + '*'
    # delete_multiselect.options[high_score[1]] = str(high_score[0]) + '*'
    predict_select.options[high_score[1]] = str(high_score[0]) + '*'


    temp_test_status_message.text = 'Not running'
    temp_test_status_message.styles = not_updated

    update_color()

    new_train_val_test_split = str(split_list[0]) + '/' + str(split_list[1]) + '/' + str(split_list[2])

    new_saved_data_choice = user_data

    if my_alg == 'Decision Tree':
        new_saved_algorithm = 'DT'
    elif my_alg == 'K-Nearest Neighbor':
        new_saved_algorithm = 'KNN'
    elif my_alg == 'Logistic Regression':
        new_saved_algorithm = 'LR'
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

    global high_score
    global old_high_score
    old_high_score = [-1, -1, 0]
    high_score = old_high_score[:]

    opt = temp.copy()
    for i in opt:
        current_index = opt.index(str(i))
        opt[current_index] = opt[current_index].replace('*', '')

        if val_accuracy[current_index] > high_score[2]:
            high_score.clear()
            high_score.append(int(opt[current_index]))
            high_score.append(current_index)
            high_score.append(val_accuracy[int(opt[current_index])-1])

    opt[high_score[1]] = str(high_score[0]) + '*'


    test_save_select.update(
        options = opt.copy(),
        value = None
    )

    predict_select.update(
        options = opt.copy(),
        value = None
    )

    for col in save_source.data:
        save_source.data[col] = [val for index, val in enumerate(save_source.data[col]) if (index+1) not in saves_to_del]

    delete_status_message.text = 'Deleted'
    delete_status_message.styles = updated

    update_color()

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

#############################################
# --------------- TESTING --------------- #
#############################################

true_pos = nan
false_pos = nan
false_neg = nan
true_neg = nan


twenty_five_percent = dataset_size * 0.25
def determine_scale():
    # Calculate the number of instances in each split
    if test_save_select.value == '':
        temp_test_status_message.text = 'Error: please select a Save'
        temp_test_status_message.styles = not_updated
        update_color()
        return
        
    save_index = test_save_select.options.index(test_save_select.value)
    temp_split = [int(split) for split in save_source.data['train_val_test_split'][save_index].split("/")]
    
    test_size = dataset_size * (temp_split[2]/100)
    base_scale = 4
    scale = base_scale * (test_size / twenty_five_percent)
    # Ensure the scale is at least 1 to avoid too small circles
    scale = max(scale, 1)
    # print(scale)
    return scale

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
    scale = determine_scale()
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

    if precision_1 <= 0.70:
        performance_1 = 'INEFFECTIVE'
    else:
        performance_1 = 'EFFECTIVE'

    if precision_2 <= 0.70:
        performance_2 = 'INEFFECTIVE'
    else:
        performance_2 = 'EFFECTIVE'
    
    if test_accuracy <= 0.70:
        performance_3 = 'INEFFECTIVE'
    else:
        performance_3 = 'EFFECTIVE'

    new_formatted_test_html = html_test_template.format(f'{round((test_accuracy*100), 1)}%', new_true_pos, new_false_pos, new_false_neg, new_true_neg, f'{round((precision_1*100), 1)}%', f'{round((precision_2*100), 1)}%', performance_1, performance_2, performance_3)
    test_acc_display.text = new_formatted_test_html

    # Update the ColumnDataSource

    confus_source.data = new_confus_d

    # new_color_mapper = LinearColorMapper(palette = Viridis256, low = min(new_confus_d['count']), high = max(new_confus_d['count']))
    
    # color_bar.color_mapper = new_color_mapper

    # bubble.scatter(fill_color = transform('count', new_color_mapper)

###################################################
# --------------- NEW DATA TABLE --------------- #
###################################################

indices = []
tested_names = []
tested_smiles = []
predicted = []
actual = []
tfpn = []

test_cols = ['Index', 'Substance Name', 'SMILES', 'Predicted Class', 'Actual Class', 'Prediction Type']
test_tab_columns = [TableColumn(field=col, title=col, width=110) for col in test_cols]

test_table_data = {'Index': indices,
            'Substance Name': tested_names,
            'SMILES': tested_smiles, 
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

    if test_save_select.value == '':
        temp_test_status_message.text = 'Error: please select a Save'
        temp_test_status_message.styles = not_updated
        return
        
    save_index = test_save_select.options.index(test_save_select.value)
    
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
    elif temp_alg == 'LR':
        # model = model_list[2]
        model.C = temp_hyperparams[0]
        model.solver = temp_hyperparams[1]

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
        tested_smiles.append(df1_dict['SMILES'][index])


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

    # Old testing print statements
    # print(len(indices)) # print(len(predicted)) # print(len(actual)) # print(len(tested_names)) # print(len(tested_smiles)) # print(len(tfpn))

    full_test_table_data = {'Index': indices,
                'Substance Name': tested_names,
                'SMILES': tested_smiles, 
                'Predicted Class': predicted,
                'Actual Class': actual,
                'Prediction Type': tfpn}
    
    abridg_test_table_data = {'Index': indices[:15],
                'Substance Name': tested_names[:15],
                'SMILES': tested_smiles[:15], 
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
    # print(new_true_pos) # print(new_false_pos) # print(new_false_neg) # print(new_true_neg)

    global test_accuracy
    test_accuracy=round(accuracy_score(y_test, y_test_pred), 3)

    update_cmatrix(None, None, None)

    temp_test_status_message.text = "Testing complete"
    temp_test_status_message.styles = updated
    update_color()

# def run_test():
#     global my_alg, stage
#     stage = 'Test'
#     global model

#     train_test_model()

def load_test():
    temp_test_status_message.text = "Testing..."
    temp_test_status_message.styles = loading
    
    curdoc().add_next_tick_callback(train_test_model)

test_button.on_click(load_test)

# --------------- EXPORTING FULL TABLE TO XLSX OR CSV (80% of this is courtesy of ChatGPT) --------------- #
def helper():
    global b64_excel_data
    # Convert source into df
    tested_df = new_source.to_df()
    # print(tested_df.info)
    # print(tested_df)

    # Create an Excel buffer
    excel_buffer = BytesIO()
    
    # Write the DataFrame to the buffer using ExcelWriter
    # with pd.ExcelWriter(excel_buffer, 'openpyxl') as writer:
    tested_df.to_excel(excel_buffer, index=False)
    
    # Get the binary data from the buffer
    excel_data = excel_buffer.getvalue()
    
    # Encode the binary data to base64
    b64_excel_data = base64.b64encode(excel_data).decode('UTF-8')
    # print(type(b64_excel_data))

helper()

def download_xlsx():
    helper()
    js_xlsx = CustomJS(args=dict(b64_excel_data=b64_excel_data), code="""
        console.log('hi');
        var filename = 'data_result.xlsx';
        var filetext = atob(b64_excel_data.toString());
        console.log(typeof b64_excel_data);

        var buffer = new Uint8Array(filetext.length);
        for (var i = 0; i < filetext.length; i++) {
            buffer[i] = filetext.charCodeAt(i);
        }
        
        var blob = new Blob([buffer], {"type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"});
        var link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = filename;
        link.target = '_blank';
        link.style.visibility = 'hidden'
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        // link.dispatchEvent(new MouseEvent('click'))
    """)
    # temp = PreText(text='')
    # temp.js_on_change('text',js_xlsx)
    # print(temp.text)
    # temp.text=''
    # temp.text='x'
    # print(temp.text)
    export_excel.js_on_click(js_xlsx)


export_excel.on_click(download_xlsx)
# export_excel.js_on_click(js_xlsx)

# Create a CustomJS object with the JavaScript code
# b64 = export_excel.on_click(download_xlsx)
# export_excel.js_on_click(js_xlsx)

# export_excel.js_on_click(xlsx_custom_js)


#from the bokeh export csv demo
export_csv.js_on_click(CustomJS(args=dict(source=new_source), code=open(os.path.join(os.path.dirname(__file__),"csv_download.js")).read()))

# export_csv.on_click(download_csv)

##############################################
# --------------- PREDICTING --------------- #
##############################################

random_smiles = random.choices(df1_dict['SMILES'], k=3)

smiles_select = Select(title="Select SMILES String", value=random_smiles[0], options=[random_smiles[0], random_smiles[1], random_smiles[2], "Custom"], width=200)

user_smiles_input = TextInput(title = 'Enter a SMILES string:', width=200)

# test in dataset C=C(C)C(=O)O

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

def predict_biodegrad():
    global model
#     temp_tvt_list = new_train_val_test_split.split("/")
#     temp_train = int(temp_tvt_list[0])
#     temp_val = int(temp_tvt_list[1])
#     temp_test = int(temp_tvt_list[2])
#     temp_columns = save_source.data['saved_columns'][int(predict_select.value)-1]

#     train_validate_model(temp_train,temp_val,temp_test, temp_columns)
    if predict_select.value == '':
        predict_status_message.text = 'Error: please select a Save'
        predict_status_message.styles = not_updated
        return
    save_index = predict_select.options.index(predict_select.value)
    model = model_list[save_index]

    if smiles_select.value != "Custom":
        user_smiles = smiles_select.value
        user_molec = Chem.MolFromSmiles(user_smiles)
    else:
        user_smiles = user_smiles_input.value
        user_molec = Chem.MolFromSmiles(user_smiles)

        if user_molec == None:
            predict_status_message.text = 'Error: invalid SMILES string'
            predict_status_message.styles = not_updated
            return

    if user_smiles in df1_dict['SMILES']:
        known_index = df1_dict['SMILES'].index(user_smiles)
        user_name = df1_dict['Substance Name'][known_index].lower()
    else:
        user_compound = pubchempy.get_compounds(user_smiles, namespace='smiles')
        user_name = user_compound[0].iupac_name

    if save_source.data['saved_data_choice'][save_index] == data_opts[0]:
        y_pred = molecule_to_descriptors(user_molec)
    elif save_source.data['saved_data_choice'][save_index] == data_opts[1]:
        y_pred = molecule_to_morgan(user_molec)
    elif save_source.data['saved_data_choice'][save_index] == data_opts[2]:
        y_pred = molecule_to_ecfp(user_molec)
    elif save_source.data['saved_data_choice'][save_index] == data_opts[3]:
        y_pred = molecule_to_pathfp(user_molec)
        

    if user_smiles in df1_dict['SMILES']:
        known_index = df1_dict['SMILES'].index(user_smiles)
        actual_class = df1_dict['Class'][known_index]
    else:
        actual_class = 'Unknown'

    predict_status_message.text = 'Prediction complete'
    predict_status_message.styles = updated

    update_color()

    new_formatted_predict_html = html_predict_template.format(user_name, user_smiles, y_pred[0], actual_class)
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
    update_color()

predict_select.on_change('value', update_predict_status)
smiles_select.on_change('value', update_predict_status)


# ---------------- VISIBILITY --------------

# Histogram
histogram.visible = False
hist_x_select.visible = False

def toggle_hist_visibility():
    histogram.visible = not histogram.visible
    hist_x_select.visible = not hist_x_select.visible
    hist_visibility_button.label = "Show Data Exploration*" if not histogram.visible else "Hide Data Exploration*"
    hist_visibility_button.icon = down_arrow if not histogram.visible else up_arrow

hist_visibility_button.on_click(toggle_hist_visibility)


# Custom SMILES String input
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


# Visiblility of warning messages vs. status messages

warning_spacer_1 = Spacer(height = 80)
warning_spacer_2 = Spacer(height = 80)
warning_spacer_3 = Spacer(height = 80)

warning_spacer_1.visible = True
warning_spacer_2.visible = True
warning_spacer_3.visible = True

train_status_message.visible = True
tune_status_message.visible = True
temp_test_status_message.visible = True
predict_status_message.visible = True

def toggle_step_two_warn():
    if save_config_message.styles != not_updated:
        train_status_message.visible = True
        step_two_warning.visible = False
        warning_spacer_1.visible = True
        tune_status_message.visible = True
        step_three_warning.visible = False
        warning_spacer_2.visible = True
        temp_test_status_message.text = 'Not running'
        temp_test_status_message.styles = not_updated
        temp_test_status_message.visible  = True
        step_four_warning.visible = False
        warning_spacer_3.visible = True
        predict_status_message.text = 'Not running'
        predict_status_message.styles = not_updated
        predict_status_message.visible = True
        step_five_warning.visible = False

    else:
        train_status_message.visible = False
        step_two_warning.visible = True
        warning_spacer_1.visible = False

    update_color()

train_button.on_click(toggle_step_two_warn)
save_config_button.on_click(toggle_step_two_warn)

def toggle_step_three_warn():
    if train_status_message.styles != not_updated:
        step_three_warning.visible = False
        tune_status_message.visible = True
        warning_spacer_2.visible = True
    else:
        step_three_warning.visible = True
        tune_status_message.visible = False
        warning_spacer_2.visible = False

    update_color()

tune_button.on_click(toggle_step_three_warn)
train_button.on_click(toggle_step_three_warn)

saves_exist = False
def toggle_step_four_warn():
    global saves_exist
    if saves_exist:
        step_four_warning.visible = False
        temp_test_status_message.visible = True
        warning_spacer_3.visible = True

    elif save_source.data['saved_data_choice'] != []:
        saves_exist = True
        step_four_warning.visible = False
        temp_test_status_message.visible = True
        warning_spacer_3.visible = True
    else:
        step_four_warning.visible = True
        temp_test_status_message.visible = False
        warning_spacer_3.visible = False

    update_color()

test_button.on_click(toggle_step_four_warn)

def toggle_step_five_warn():
    global saves_exist
    if saves_exist:
        step_five_warning.visible = False
        predict_status_message.visible = True

    elif save_source.data['saved_data_choice'] != []:
        saves_exist = True
        step_five_warning.visible = False
        predict_status_message.visible = True
    else:
        step_five_warning.visible = True
        predict_status_message.visible = False

    update_color()

predict_button.on_click(toggle_step_five_warn)


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
    [data_select, column(small_height_spacer)],
    [tiny_height_spacer],
    [column(row(tvt_slider, column(small_med_height_spacer)), split_display)],
    [tiny_height_spacer],
    [column(save_config_button, save_config_message)]
)

tab1_layout = row(left_page_spacer, column(top_page_spacer, row(column(step_one, data_config_layout), data_tab_table), tiny_height_spacer, hist_visibility_button, hist_x_select, histogram))

hyperparam_layout = layout(
    [step_three],
    [hp_slider],
    [hp_toggle],
    [hp_select],
    [tune_button],
    [tune_status_message]
)

delete_layout = layout(
    [delete_multiselect],
    [delete_button],
    [delete_status_message]
)

tab2_layout = row(left_page_spacer, column(top_page_spacer, step_two, alg_select, row(train_button), train_status_message, step_two_warning, warning_spacer_1, hyperparam_layout, warning_spacer_2, step_three_warning, delete_layout), large_left_page_spacer, column(top_page_spacer, learning_curve, saved_data_table), column(top_page_spacer, val_acc_display))

test_button_layout = layout(
    [column(step_four, test_save_select, asterisk, row(test_button), temp_test_status_message, step_four_warning, warning_spacer_3, export_excel, export_csv)]
)

tab3_layout = row(left_page_spacer, column(top_page_spacer, row(column(row(test_button_layout, large_left_page_spacer, bubble), new_table), column(small_med_height_spacer, test_acc_display))))

tab4_layout = row(left_page_spacer, column(top_page_spacer, step_five, predict_select, asterisk, smiles_select, user_smiles_input, predict_instr, predict_button, predict_status_message, step_five_warning), column(top_page_spacer, predict_display))

tabs = Tabs(tabs = [TabPanel(child = tab0_layout, title = 'Steps'),
                    TabPanel(child = tab1_layout, title = 'Data'),
                    TabPanel(child = tab2_layout, title = 'Train and Validate'),
                    TabPanel(child = tab3_layout, title = 'Test'),
                    TabPanel(child = tab4_layout, title = 'Predict')
                ])

curdoc().add_root(tabs)

end_time = datetime.now()                                                 # --------- timer code
elapsed_time = end_time - start_time                                      # --------- timer code
print(f"Entire File Runtime: {elapsed_time.total_seconds():.2f} seconds") # --------- timer code