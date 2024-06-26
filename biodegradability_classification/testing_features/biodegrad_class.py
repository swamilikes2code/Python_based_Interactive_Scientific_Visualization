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
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV

#CONTENTS/HEADERS throughout this code
# message styles, accuracy lists, status messages, buttons
# instructions
# data selection, data split, interactive data exploration, save data button
# algorithm select and run, feature selection algorithm
# hyperparameter tuning + button, box plot and save
# testing
# visibility, layouts

# ---------------MESSAGE STYLES-----------------

not_updated = {'color': 'red', 'font-size': '14px'}
loading = {'color': 'orange', 'font-size': '14px'}
updated = {'color': 'green', 'font-size': '14px'}

# ---------------ACCURACY LISTS-----------------
# Create empty lists - declare at the top to use everywhere
val_accuracy = [nan for i in range(10)]
tuned_val_accuracy = [nan for i in range(10)]
tuned_test_accuracy = [nan for i in range(10)]
fs_val_accuracy = [nan for i in range(10)]
fs_test_accuracy = [nan for i in range(10)]
saved_accuracy = [nan for i in range(10)]
combo_list = val_accuracy + tuned_val_accuracy + tuned_test_accuracy + saved_accuracy

saved_test_acc = []


# ---------------STATUS MESSAGES-----------------

save_config_message = Div(text='Configuration not saved', styles=not_updated)
train_status_message = Div(text='Not running', styles=not_updated)
fs_status_message = Div(text='Not running', styles=not_updated)
tune_status_message = Div(text='Not running', styles=not_updated)
plot_status_message = Div(text='Plot not updated', styles=not_updated)
predict_status_message = Div(text = 'Not running', styles=not_updated)

# -------------------BUTTONS--------------------

save_config_button = Button(label="Save Current Configuration", button_type="warning")
train_button = Button(label="Run ML algorithm", button_type="success")
fs_button = Button(label="Run Feature Selection", button_type="success")
tune_button = Button(label = "Tune", button_type = "success")
save_plot_button = Button(label="Save current plot", button_type="warning")
display_save_button = Button(label = "Display save")
predict_button = Button(label = 'Predict')

#svg icons for buttons
up_arrow = SVGIcon(svg = '''<svg  xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-chevron-up"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M6 15l6 -6l6 6" /></svg>''')
down_arrow = SVGIcon(svg = '''<svg  xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-chevron-down"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M6 9l6 6l6 -6" /></svg>''')

data_exp_vis_button = Button(label="Show Data Exploration*", button_type="primary", icon = down_arrow)
fs_vis_button = Button(label="Show Feature Selection*", button_type="primary", icon = down_arrow)

# -----------------INSTRUCTIONS-----------------

intro_instr = Div(text="""
                  <div style='background-color: #DEF2F1; padding: 20px; font-family: Arial, sans-serif;'>
                  <div>Start by opening the <b>data</b> tab. This tab is used for preparing the biodegradability data for training. 
                  You will have the option to split the data into <i>training, validating, and testing</i>, and select which columns 
                  of data (each of which represent a <i>molecular property</i>) should be used to train the mode. Once you are done 
                  preparing your data, save your choices, and continue on to the next tab.*</div>
                  <div>‎</div>
                  <div>The <b>train</b> tab is used to actually train the data, using the <i>Machine Learning algorithm</i> of your choice. 
                  Once you set and run your chosen algorithm, you may continue to the next tab.*</div>
                  <div>‎</div>
                  <div>The <b>fine-tune</b> tab is where you are able to fine-tune the <i>hyperparameters</i> of your chosen model, 
                  and <i>validate</i> the results between two of your tunes on a boxplot. You will also be able to save any model that you have created 
                  with these past three tabs, and display their <i>testing accuracies</i> on the boxplot. Once you have saved at least one model, 
                  you may continue to the final tab.</div>
                  <div>‎</div>
                  <div>The <b>test</b> tab is where you will be able to test any of the saved models. FINISH THIS WHEN TEST TAB IS READY.</div>
                  <div>‎</div>
                  <div>For more information about each of the <i>italicized</i> vocab words, see the above navigation menu.</div>
                  <div>‎</div>
                  <div>*For those interested in more advanced Machine Learning topics, starred tabs contain optional plots and processes 
                  that you may also use to inform your decisions throughout the module. These can be accessed using the <b>blue</b> buttons.</div>
                  </div>""",
width=750, height=500)

splitter_help = HelpButton(tooltip=Tooltip(content=Div(text="""
                 <div style='background-color: #DEF2F1; padding: 16px; font-family: Arial, sans-serif;'>
                 Use this <b>slider</b> to split the data into <i>train/validate/test</i> percentages.
                 </div>""", width=280), position="right"))

datatable_help = HelpButton(tooltip=Tooltip(content=Div(text="""
                 <div style='background-color: #DEF2F1; padding: 16px; font-family: Arial, sans-serif;'>
                 <b>Select/deselect</b> property columns for training the model by dragging your mouse over them.
                 </div>""", width=280), position="right"))

datavis_help = HelpButton(tooltip=Tooltip(content=Div(text="""
                 <div style='background-color: #DEF2F1; padding: 16px; font-family: Arial, sans-serif;'>
                 View the graphical relationship between any two numerical properties.
                 </div>""", width=280), position="right"))

train_help = HelpButton(tooltip=Tooltip(content=Div(text="""
                  <div style='background-color: #DEF2F1; padding: 20px; font-family: Arial, sans-serif;'>
                  Select one of the following <b>Machine Learning algorithms</b>, and click <b>run</b>. This will
                    run the algorithm 10 times, and display a list of these accuracy values.
                  </div>""", width=280), position="right"))

fs_help = HelpButton(tooltip=Tooltip(content=Div(text="""
                  <div style='background-color: #DEF2F1; padding: 20px; font-family: Arial, sans-serif;'>
                  <div><b>Optional:</b> run <b>feature selection</b> for your chosen ML algorithm. This will display
                           which columns are recommended for training this ML algorithm. Afterwards, you can update
                           your chosen columns accordingly on the <b>data</b> tab, and rerun your algorithm.</div>
                           <div><i>Note: Feature selection is <b>not</b> compatible with <b>K-Nearest Neighbors</b></i></div>
                  </div>""", width=280), position="right"))

tune_help = HelpButton(tooltip=Tooltip(content=Div(text="""
                 <div style='background-color: #DEF2F1; padding: 20px; font-family: Arial, sans-serif;'>
                 <div>Change <b>hyperparameters</b> based on your chosen ML algorithm, 
                 and click <b>tune</b> to compare the tuned model's <b>validation accuracies</b> to the untuned model 
                 on the boxplot, as well as your current tune's actual <b>testing accuracies</b>.</div>
                 <div>You can <b>save</b> any model at any time and <b>display</b> any saved model's <b>testing accuracy</b> on the plot.</div>
                 </div>""", width=280), position="right"))

test_instr = Div(text="""
                 <div style='background-color: #DEF2F1; padding: 20px; font-family: Arial, sans-serif;'>
                 TEST INSTRUCTIONS GO HERE:
                 </div>""",
width=300, height=75)


# --------------- DATA SELECTION ---------------

#for ref:
# df is original csv, holds fingerprint list and 167 cols of fingerprint bits
# df_display > df_subset > df_dict are for displaying table

# Load data from the csv file
file_path = r'biodegrad.csv'
df = pd.read_csv(file_path, low_memory=False)
df_display = df.iloc[:,:22]  #don't need to display the other 167 rows of fingerprint bits
df = df.drop(columns=['Fingerprint List'])  #removing the display column, won't be useful in training

# Columns that should always be shown
mandatory_columns = ['Substance Name', 'Smiles', 'Class']

# Ensure mandatory columns exist in the dataframe (if not, create dummy columns) (hopefully shouldn't have to apply)
for col in mandatory_columns:
    if col not in df_display.columns:
        df_display[col] = "N/A"

# saved list to append to
user_columns = []

# Limit the dataframe to the first 10 rows
df_subset = df_display.head(15)

df_dict = df_subset.to_dict("list")
cols = list(df_dict.keys())

# Separate mandatory and optional columns
optional_columns = [col for col in cols if col not in mandatory_columns]

# Create column datasource
data_tab_source = ColumnDataSource(data=df_subset)

# Create figure
data_tab_columns = [TableColumn(field=col, title=col, width = 100) for col in cols]
data_tab_table = DataTable(source=data_tab_source, columns=data_tab_columns, width=800, height_policy = 'auto', autosize_mode = "none")

# Create widget excluding mandatory columns
# checkbox_button_group = CheckboxButtonGroup(labels=optional_columns, active=list(range(len(optional_columns))), orientation = 'vertical')
data_multiselect = MultiSelect(options = optional_columns, value = optional_columns, size = 12)

# Update columns to display
def update_cols(display_columns):
    # Always include mandatory columns
    all_columns = mandatory_columns + display_columns
    data_tab_table.columns = [col for col in data_tab_columns if col.title in all_columns]

def update_table(attr, old, new):
    # cols_to_display = [checkbox_button_group.labels[i] for i in checkbox_button_group.active]
    cols_to_display = data_multiselect.value
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
    #("name", "@names"),
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
        #'names' : df['Substance Name'],
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
data_multiselect.on_change('value', update_table)

# range slider on change
tvt_slider.js_on_change('value', callback)
tvt_slider.on_change('value', update_values)

# Save columns to saved list (split already saved)
def save_config():
    # temp_columns = [checkbox_button_group.labels[i] for i in checkbox_button_group.active]
    temp_columns = data_multiselect.value
    if len(temp_columns) == 0:
        save_config_message.text = 'Error: must select at least one feature'
        save_config_message.styles = not_updated
        return

    global user_columns
    user_columns.clear()
    user_columns = temp_columns.copy()

    global combo_list
    combo_list = [nan for i in range(40)]
    update_boxplot()

    #split_list isn't located here as the split values update in the list upon the change of the range slider
    #the collective save button is to make the design more cohesive

    save_config_message.text = 'Configuration saved'
    save_config_message.styles = updated

def load_config():
    save_config_message.text = "Loading config..."
    save_config_message.styles = loading

    train_status_message.text='Not running'
    train_status_message.styles=not_updated


    tune_status_message.text='Not running'
    tune_status_message.styles=not_updated

    fs_status_message.text = 'Not running'
    fs_status_message.styles=not_updated

    curdoc().add_next_tick_callback(save_config)

# Attach callback to the save button
save_config_button.on_click(load_config)

# --------------- ALGORITHM SELECT AND RUN ---------------

# algorithm name holder
my_alg = 'Decision Tree'

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
        model = DecisionTreeClassifier()
    elif my_alg == "K-Nearest Neighbor":
        model = KNeighborsClassifier()
    else:
        model = LinearSVC()
    
    set_hyperparameter_widgets()

    split_and_train_model(split_list[0],split_list[1],split_list[2], user_columns)

    # Changing the list used to create boxplot
    global combo_list
    combo_list.clear()
    combo_list = val_accuracy + [nan for i in range(20)] + saved_accuracy

    update_boxplot()

    # Updating accuracy display
    accuracy_display.text = f"""<div style='background-color: #FBE9D0; padding: 20px; font-family: Arial, sans-serif;'>
    <div><b>Your Data Split:</b> {split_list} </div><div><b>Your Selected columns:</b> {user_columns}<div><b>Validation Accuracy:</b> {val_accuracy}</div><div><b>Test Accuracy:</b> {test_accuracy}</div>
    </div>"""

def split_and_train_model(train_percentage, val_percentage, test_percentage, columns):

    # run and shuffle ten times, and save result in list
    global val_accuracy, test_accuracy
    global tuned_val_accuracy, tuned_test_accuracy

    if stage == 'Train':
        val_accuracy.clear()
        test_accuracy.clear()
    elif stage == 'Tune':
        tuned_val_accuracy.clear()
        tuned_test_accuracy.clear()
    elif stage == 'FS':
        fs_val_accuracy.clear()
        fs_test_accuracy.clear()

    
    train_columns = []
    train_columns += columns
    if 'Fingerprint List' in columns:
        train_columns.remove("Fingerprint List")
        train_columns += [str(i) for i in range(167)]

    np.random.seed(123)

    for i in range(10):
        X = df[train_columns]
        y = df['Class']

        # splitting
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(100-train_percentage)/100)
        test_split = test_percentage / (100-train_percentage)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_split)
        
        # train model
        model.fit(X_train, y_train)
        
        # calculating accuracy
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        if stage == 'Train':
            val_accuracy.append(round(accuracy_score(y_val, y_val_pred), 3))
            test_accuracy.append(round(accuracy_score(y_test, y_test_pred), 3))
        elif stage == 'Tune':
            tuned_val_accuracy.append(round(accuracy_score(y_val, y_val_pred), 3))
            tuned_test_accuracy.append(round(accuracy_score(y_test, y_test_pred), 3))
        elif stage == 'FS':
            fs_val_accuracy.append(round(accuracy_score(y_val, y_val_pred), 3))
            fs_test_accuracy.append(round(accuracy_score(y_test, y_test_pred), 3))
            

def load_ML():
    train_status_message.text = f'Running {my_alg}...'
    train_status_message.styles = loading

    tune_status_message.text='Not running'
    tune_status_message.styles=not_updated

    curdoc().add_next_tick_callback(run_ML)

# Attach callback to the run button
train_button.on_click(load_ML)


# --------------- FEATURE SELECTION ALGORITHM ---------------

# result_text = PreText(text="", width=500, height=200)
# selected_features_text = Div(text="")
fs_accuracy_display = Div(text="""<div style='background-color: #FBE9D0; padding: 20px; font-family: Arial, sans-serif;'>
                          <div><b>Your Data Split:</b> N/A </div><div><b>Feature Selected columns:</b> N/A</div><div><b>Validation Accuracy:</b> N/A</div><div><b>Test Accuracy:</b> N/A</div>
                          </div>""", width=600)

def run_FS():
    if save_config_message.styles == not_updated:
        fs_status_message.text = 'Error: must save data configuration before feature selection'
        fs_status_message.styles = not_updated
        return

    global my_alg, stage, model
    stage = 'FS'
    
    # Assigning model based on selected ML algorithm, using default hyperparameters
    if my_alg == "Decision Tree":
        model = DecisionTreeClassifier()
    elif my_alg == "K-Nearest Neighbor":
        model = None
        fs_status_message.text = "Error, please select a different ML algorithm"
        fs_status_message.styles = not_updated
        return
    else:
        model = LinearSVC()
    
    # Prepare data (excluding 'Class' as it is the target variable)
    X = df.drop(columns=['Class', 'Substance Name', 'Smiles'])  # Features
    y = df['Class']  # Target
    #ensure columns are numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)  # Convert to numeric and fill NaNs with 0

    np.random.seed(123)
    #create training,validation,and test set
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(100-split_list[0])/100)
    test_split = split_list[2] / (100-split_list[0])
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_split)

    #perform feature selection
    rfecv = RFECV(estimator=model, step=1, cv=5, scoring='accuracy')  # 5-fold cross-validation
    rfecv.fit(X_train, y_train)

    #get optimal number of features
    # optimal_features = rfecv.n_features_
    
    # Get the selected features
    selected_features = X.columns[rfecv.support_]

    # Clean up fingerprint info
    ideal_cols = []
    contains_fingerprint = False
    for feat in selected_features:
        if 'pka' in feat or'α'in feat:
            ideal_cols.append(feat)
        else:
            contains_fingerprint = True
    if contains_fingerprint:
        ideal_cols.append('Fingerprint List')
    # selected_features_text.text = f"Selected Columns: {ideal_cols}"

    # Transform the training and testing data to keep only the selected features
    X_train_rfecv = rfecv.transform(X_train)
    X_test_rfecv = rfecv.transform(X_test)

    # Fit the model using the selected features
    model.fit(X_train_rfecv, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test_rfecv)

    # accuracy = round(accuracy_score(y_test, y_pred), 2)
    

    # Update the result text
    # report = classification_report(y_test, y_pred)
    # result_text.text = f"Accuracy: {accuracy}\n"#\nClassification Report:\n{report}"
    fs_status_message.text = f"Feature selection completed using {my_alg}."
    fs_status_message.styles = updated


    split_and_train_model(split_list[0],split_list[1],split_list[2], selected_features.tolist())
    # Updating accuracy display


    fs_accuracy_display.text = f"""<div style='background-color: #FBE9D0; padding: 20px; font-family: Arial, sans-serif;'>
    <div><b>Your Data Split:</b> {split_list} </div><div><b>Feature Selected Columns:</b> {ideal_cols}</div><div><b>Validation Accuracy:</b> {fs_val_accuracy}</div><div><b>Test Accuracy:</b> {fs_test_accuracy}</div>
    </div>"""

#first update the text that the feature selection algorithm is running
def load_FS():
    fs_status_message.text = f'Running Feature Selection with {my_alg}. This may take more than 30 seconds...'
    fs_status_message.styles = loading
    curdoc().add_next_tick_callback(run_FS)#then run the feature selection algorithm

#when the feature selection button gets clicks
fs_button.on_click(load_FS)




# --------------- HYPERPARAMETER TUNING + BUTTON ---------------

# a list of an int an string
## decision tree - int/nan, string
## KNN - int, string
## SVC - int, ""
hyperparam_list = [nan,"best"]

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

    split_and_train_model(split_list[0],split_list[1],split_list[2], user_columns)


    # Changing the list used to create boxplot
    global combo_list
    combo_list[10:20] = tuned_val_accuracy.copy()
    combo_list[20:30] = tuned_test_accuracy.copy()

    update_boxplot()

    mean_tuned_val_accuracy = np.mean(tuned_val_accuracy)
    std_tuned_val_accuracy = np.std(tuned_val_accuracy)
    mean_tuned_test_accuracy = np.mean(tuned_test_accuracy)
    std_tuned_test_accuracy = np.std(tuned_test_accuracy)

    # Updating accuracy display
    tuned_accuracy_display.text = f"""<div style='background-color: #FBE9D0; padding: 20px; font-family: Arial, sans-serif;'>
    <div><b>Tuned Validation Accuracy:</b></div> 
    <div>Mean: {round(mean_tuned_val_accuracy, 3)}, Standard Dev: {round(std_tuned_val_accuracy, 3)}</div>
    <div>‎</div>
    <div><b>Tuned Test Accuracy:</b></div> 
    <div>Mean: {round(mean_tuned_test_accuracy, 3)}, Standard Dev: {round(std_tuned_test_accuracy, 3)}</div>
    </div>"""

# hyperparameter tuning widgets, default to decision tree
hp_slider = Slider(
    title = "Max Depth of Tree",
    start= 0,
    end = 15,
    value = 2,
    step = 1
)
hp_select = Select(
    title = "Splitter strategy",
    value = "best",
    options = ["best", "random"]
)
hp_toggle = Checkbox(
    label = "None",
    visible = True,
    active = False
)
hp_toggle.margin = (24, 10, 5, 10)

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
        print("max iter", model.max_iter)
        print("loss func", model.loss)

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
        model.max_iter = new
        hyperparam_list[1] = ""

def hp_select_callback(attr, old, new):
    global my_alg
    hyperparam_list[1] = new
    if my_alg == 'Decision Tree':
        model.splitter = new
    elif my_alg == 'K-Nearest Neighbor':
        model.weights = new
    else:
        hyperparam_list[1] = ""

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
    global my_alg
    if my_alg == 'Decision Tree':
        #hyperparameters are 
        # splitter strategy (splitter, best vs. random, select)
        # max_depth of tree (max_depth, int slider)

        hp_slider.update(
            title = "Max Depth of Tree",
            disabled = True,
            show_value = False,
            start= 1,
            end = 15,
            value = 2,
            step = 1
        )
        hp_toggle.update(
            label = "None",
            visible = True,
            active = True
        )
        hp_select.update(
            title = "Splitter strategy",
            value = "best",
            options = ["best", "random"]
        )
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
            value = 5,
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
        # loss (loss, hinge vs. squared_hinge, select) 
        # the max iterations to be run (max_iter, int slider)
        # model = LinearSVC()

        hp_slider.update(
            title = "Maximum iterations", #default is 1000
            disabled = False,
            show_value = True,
            start = 1,
            end = 20,
            value = 10,
            step = 1
        )

        hp_toggle.visible = False

        hp_select.visible = False

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

plot_counter = 0 #the amount of times button has been pressed

# Create empty plot
boxplot = figure(x_range=['pretune val', 'posttune val', 'test', 'saved'],
            y_range = (0.4, 1.0),
            width = 450,
            height = 450,
            tools="",
            toolbar_location=None,
            background_fill_color="#eaefef",
            title="Model Accuracies",
            y_axis_label="prediction accuracy")

df_box = pd.DataFrame()
source = ColumnDataSource(df_box)
df_hover_points = pd.DataFrame()
hover_points_source = ColumnDataSource(df_hover_points)


# Create status message Div
# saved_split_message = Div(text = 'Saved split: N/A', styles = not_updated)
# saved_col_message = Div(text='Saved columns: N/A', styles=not_updated)
# saved_alg_message = Div(text='Saved alg: N/A', styles=not_updated)
# saved_data_message = Div(text='Saved val acc: N/A', styles=not_updated)

def update_df_box():
    # making the initial boxplot
    global combo_list
    d = {'kind': ['pretune val' for i in range(10)] + ['posttune val' for j in range(10)] + ['test' for k in range(10)] + ['saved' for l in range(10)],
        'accuracy': combo_list
        }
    
    dp = {'kind' : ['pretune val', 'posttune val', 'test', 'saved'],
          'accuracy' : [.7, .7, .7, .7]}
    
    global df_box
    df_box = pd.DataFrame(data=d)

    global df_hover_points
    df_hover_points = pd.DataFrame(data=dp)

    # compute quantiles
    qs = df_box.groupby("kind").accuracy.quantile([0.25, 0.5, 0.75])
    qs = qs.unstack().reset_index()
    qs.columns = ["kind", "q1", "q2", "q3"]
    df_box = pd.merge(df_box, qs, on="kind", how="left")

    df_hover_points = pd.merge(df_hover_points, qs, on="kind", how="left")
    df_hover_points['accuracy'] = df_hover_points['q2']

    # compute IQR outlier bounds
    iqr = df_box.q3 - df_box.q1
    df_box["upper"] = df_box.q3 + 1.5*iqr
    df_box["lower"] = df_box.q1 - 1.5*iqr

    # set max and mins for whiskers
    df_box['min'] = ''
    df_box['max'] = ''
    
    for index, entry in enumerate(df_box['kind']):
        minmax = get_minmax(entry)
        df_box.iloc[index, -2] = minmax[0]
        df_box.iloc[index, -1] = minmax[1]

    df_hover_points['min'] = [df_box['min'][i] for i in [0, 10, 20, 30]]
    df_hover_points['max'] = [df_box['max'][i] for i in [0, 10, 20, 30]]

    # update outliers
    global outliers
    outliers = ColumnDataSource(df_box[~df_box.accuracy.between(df_box.lower, df_box.upper)])
    
    global source
    source.data = dict(df_box)

    global hover_points_source
    hover_points_source.data = dict(df_hover_points)

    plot_status_message.text = 'Plot updated'
    plot_status_message.styles = updated

def get_minmax(kind):
    temp_list = []
    temp_index = 0

    if kind == 'pretune val':
        temp_list = combo_list[:10]
    elif kind == 'posttune val':
        temp_list = combo_list[10:20]
        temp_index = 10
    elif kind == 'test':
        temp_list = combo_list[20:30]
        temp_index = 20
    elif kind == 'saved':
        temp_list = combo_list[30:]
        temp_index = 30

    if combo_list[temp_index] == nan:  # when module is first loaded
        return 0,0
    else:
        abs_min = min(temp_list)
        while abs_min < df_box["lower"][temp_index]:
            temp_list.remove(abs_min)
            abs_min = min(temp_list)

        abs_max = max(temp_list)
        while abs_max > df_box["upper"][temp_index]:
            temp_list.remove(abs_max)
            abs_max = max(temp_list)

        return abs_min, abs_max

def make_glyphs():
    # make all of the glyphs

    global hover_points
    hover_points = boxplot.scatter("kind", "accuracy", source = hover_points_source, size = 50, alpha=0)

    # outlier range
    global whisker
    global outlier_points
    whisker = Whisker(base="kind", upper="max", lower="min", source=source)
    whisker.upper_head.size = whisker.lower_head.size = 20
    boxplot.add_layout(whisker)

    outlier_points = boxplot.scatter("kind", "accuracy", source=outliers, size=6, color="black", alpha=0.3)

    # quantile boxes
    global kinds, cmap, top_box, bottom_box
    kinds = df_box.kind.unique()
    cmap = factor_cmap("kind", "Paired3", kinds)
    top_box = boxplot.vbar(x = "kind", width = 0.7, bottom = "q2", top = "q3", color=cmap, line_color="black", source = source)
    bottom_box = boxplot.vbar("kind", 0.7, "q1", "q2", color=cmap, line_color="black", source = source)

    box_hover = HoverTool(tooltips=[
                            ('max', '@max'),
                            ('q3', '@q3'),
                            ('median', '@q2'),
                            ('q1', '@q1'),
                            ('min', '@min')
                        ], mode='vline', renderers = [hover_points])
    boxplot.add_tools(box_hover)

    # constant plot features
    boxplot.xgrid.grid_line_color = None
    boxplot.axis.major_label_text_font_size="14px"
    boxplot.axis.axis_label_text_font_size="12px"

def update_boxplot():
    global df_box
    global source
    global plot_counter
    global outliers
    update_df_box()

    if plot_counter == 0:
        make_glyphs()


    whisker.source = source
    top_box.data_source = source
    bottom_box.data_source = source
    outlier_points.data_source = outliers
    plot_counter += 1

def load_boxplot():
    plot_status_message.text = 'Updating plot...'
    plot_status_message.styles = loading
    curdoc().add_next_tick_callback(update_boxplot)

# making select to choose save num to display/use
display_save_select = Select(title = "Choose a save to display", options = [], margin=(5, 40, 5, 5))
predict_select = Select(title = 'Choose a save to predict with', options = [])

new_save_number = 0

# Define an empty data source
saved_data = dict(
    save_number = [],
    train_val_test_split = [],
    saved_columns = [],
    saved_algorithm = [],
    saved_hyperparams = [],
    mean_saved_test_acc = [],
    std_saved_test_acc = []
)
save_source = ColumnDataSource(saved_data)

# Define table columns
saved_columns = [
    TableColumn(field="save_number", title="#", width = 25),
    TableColumn(field="train_val_test_split", title="Train/Val/Test split", width = 260),
    TableColumn(field="saved_columns", title="Saved col."),
    TableColumn(field="saved_algorithm", title="Saved alg.", width = 140),
    TableColumn(field="saved_hyperparams", title="Saved hp.", width = 220),
    TableColumn(field="mean_saved_test_acc", title="Mean Test acc.", width = 210),
    TableColumn(field="std_saved_test_acc", title="Std Test acc.", width = 180)
]

# Create a DataTable
saved_data_table = DataTable(source=save_source, columns=saved_columns, width=600, height=280, index_position=None)


def save_plot():
    if tune_status_message.styles == not_updated:
        plot_status_message.text = '<div>Error: must tune model</div><div>before saving</div>'
        plot_status_message.styles = not_updated
        return

    global combo_list
    global hyperparam_list
    global new_save_number
    global new_train_val_test_split
    global new_saved_columns
    global new_saved_algorithm
    global new_saved_hyperparams
    global saved_test_acc
    global new_mean_saved_test_acc
    global new_std_saved_test_acc

    new_save_number += 1
    display_save_select.options.append(str(new_save_number))
    predict_select.options.append(str(new_save_number))

    new_train_val_test_split = str(split_list[0]) + '/' + str(split_list[1]) + '/' + str(split_list[2])


    new_saved_columns = user_columns
    if my_alg == 'Decision Tree':
        new_saved_algorithm = 'DT'
    elif my_alg == 'K-Nearest Neighbor':
        new_saved_algorithm = 'KNN'
    elif my_alg == 'Support Vector Classification':
        new_saved_algorithm = 'SVC'
    else:
        new_saved_algorithm = my_alg
    new_saved_hyperparams = str(hyperparam_list) # convert back to list for usage when loading a saved profile
    new_saved_test_acc = combo_list[20:30]

    saved_test_acc.append(new_saved_test_acc)

    new_mean_saved_test_acc = round(np.mean(new_saved_test_acc), 3)
    new_std_saved_test_acc = round(np.std(new_saved_test_acc), 3)

    add_row()

    plot_status_message.text = 'Plot saved'
    plot_status_message.styles = updated

# Add new row to datatable every time a plot is saved
def add_row():
    new_saved_data = {
        'save_number': [new_save_number],
        'train_val_test_split': [new_train_val_test_split],
        'saved_columns': [new_saved_columns],
        'saved_algorithm': [new_saved_algorithm],
        'saved_hyperparams': [new_saved_hyperparams],
        'mean_saved_test_acc': [new_mean_saved_test_acc],
        'std_saved_test_acc' : [new_std_saved_test_acc]
    }
    save_source.stream(new_saved_data)

def load_save():
    plot_status_message.text = 'Updating saved data...'
    plot_status_message.styles = loading
    curdoc().add_next_tick_callback(save_plot)

# Attach callback to the save_plot button
save_plot_button.on_click(load_save)


def display_save():
    # print(display_save_select.value)
    if len(display_save_select.options) == 0:
        plot_status_message.text = '<div>Error: must save plot</div><div>before displaying</div>'
        plot_status_message.styles = not_updated
        return
    elif display_save_select.value == '':
        plot_status_message.text = '<div>Error: must choose a save</div><div>before displaying</div>'
        plot_status_message.styles = not_updated
        return

    global saved_accuracy, combo_list, saved_test_acc
    index = int(display_save_select.value) - 1
    saved_accuracy = saved_test_acc[index]

    combo_list[30:] = saved_accuracy.copy()
    update_boxplot()

    plot_status_message.text = 'Plot updated'
    plot_status_message.styles = updated

def load_display_save():
    plot_status_message.text = 'Updating plot...'
    plot_status_message.styles = loading
    curdoc().add_next_tick_callback(display_save)

# callback to display_save button
display_save_button.on_click(load_display_save)

# --------------- TESTING ---------------

user_smiles_input = TextInput(title = 'Enter a SMILES string:')
# test in dataset C=C(C)C(=O)O

def predict_biodegrad():
    temp_tvt_list = new_train_val_test_split.split("/")
    temp_train = int(temp_tvt_list[0])
    temp_val = int(temp_tvt_list[1])
    temp_test = int(temp_tvt_list[2])
    temp_columns = save_source.data['saved_columns'][int(predict_select.value)-1]

    split_and_train_model(temp_train,temp_val,temp_test, temp_columns)

    user_molec = Chem.MolFromSmiles(user_smiles_input.value)

    user_fp = np.array(MACCSkeys.GenMACCSKeys(user_molec))

    user_df = pd.DataFrame(user_fp)

    user_df = user_df.transpose() #each bit has its own column

    # --------------TEST TAB WORKS UP UNTIL HERE-----------------------
    # The model is not receiving the actual saved columns it needs, except for fingerprint. 
    # For example, has 167 features, but is expecting 185 features as input (there are 18 columns, excluding fingerprint)
    # If I get to it I'll try to fix this this weekend

    user_biodegrad = model.predict(user_df)

    print(user_biodegrad)

    predict_status_message.styles = updated
    # if user_biodegrad == 0:
    #     predict_status_message.text = 'Molecule is not readily biodegradable (class 0)'
    # elif user_biodegrad == 1:
    #     predict_status_message.text = 'Molecule is readily biodegradable (class 1)'
    # else:
    #     predict_status_message.text = 'error'

    return

def load_predict():
    predict_status_message.text = 'Predicting...'
    predict_status_message.styles = loading
    curdoc().add_next_tick_callback(predict_biodegrad)

# callback for predict button
predict_button.on_click(load_predict)


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

# Feature selection
fs_help.visible = False
fs_button.visible = False
fs_status_message.visible = False
fs_accuracy_display.visible = False
# selected_features_text.visible = not selected_features_text.visible
# result_text.visible = not result_text.visible

#Callback function to toggle visibility
def toggle_feature_select_visibility():
    fs_help.visible = not fs_help.visible
    fs_button.visible = not fs_button.visible
    fs_status_message.visible = not fs_status_message.visible
    fs_accuracy_display.visible = not fs_accuracy_display.visible
    fs_vis_button.label = "Show Feature Selection*" if not fs_help.visible else "Hide Feature Selection*"
    fs_vis_button.icon = down_arrow if not fs_vis_button.visible else up_arrow

# Link the button to the callback
fs_vis_button.on_click(toggle_feature_select_visibility)



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
    [datatable_help, data_multiselect],
    [small_height_spacer],
    [splitter_help, column(tvt_slider, split_display)],
    [small_height_spacer],
    [button_spacer, column(save_config_button, save_config_message)]
)
interactive_graph = column(data_exp_vis_button, row(datavis_help, column(data_exp, row(select_x, select_y)))) #create data graph visualization 
tab1_layout = row(left_page_spacer, column(top_page_spacer, row(data_tab_table, data_config_layout), small_height_spacer, interactive_graph))

fs_layout = column(fs_vis_button, fs_help, fs_button, fs_status_message, fs_accuracy_display)
tab2_layout = row(left_page_spacer, column(top_page_spacer, train_help, alg_select, train_button, train_status_message, accuracy_display, height_spacer, fs_layout))

hyperparam_layout = layout(
    [tune_help],
    [hp_slider, hp_toggle],
    [hp_select],
    [tune_button, save_plot_button],
    [tune_status_message],
    [tuned_accuracy_display],
    [large_height_spacer]
)
save_layout = row(column(display_save_select, display_save_button, plot_status_message), saved_data_table)
tab3_layout = row(left_page_spacer, column(top_page_spacer, hyperparam_layout, save_layout), boxplot)

tab4_layout = row(left_page_spacer, column(top_page_spacer, test_instr, user_smiles_input, predict_select, predict_button, predict_status_message))

tabs = Tabs(tabs = [TabPanel(child = tab0_layout, title = 'Instructions'),
                    TabPanel(child = tab1_layout, title = 'Data'),
                    TabPanel(child = tab2_layout, title = 'Train'),
                    TabPanel(child = tab3_layout, title = 'Fine-Tune'),
                    TabPanel(child = tab4_layout, title = 'Test')
                ])

curdoc().add_root(tabs)