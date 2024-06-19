import pandas as pd
import numpy as np
from bokeh.models import ColumnDataSource, DataTable, TableColumn, CheckboxButtonGroup, Button, Div, RangeSlider, Select, Whisker, Slider, Checkbox, Tabs, TabPanel, TextInput
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models.callbacks import CustomJS
from bokeh.plotting import figure
from bokeh.palettes import Category10
from bokeh.transform import factor_cmap
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
'''
This file is a draft for combining the module's features
'''

#for ref:
# df is original csv, holds fingerprint list and 167 cols of fingerprint bits
# df_display > df_subset > df_dict are for displaying table

# ---------------MESSAGE STYLES-----------------

not_updated = {'color': 'red', 'font-size': '16px'}
loading = {'color': 'orange', 'font-size': '16px'}
updated = {'color': 'green', 'font-size': '16px'}

# ---------------STATUS MESSAGES-----------------

# data_table_title = Div(text='Select columns for training', styles=updated)
save_config_message = Div(text='Configuration not saved', styles=not_updated)
train_status_message = Div(text='Not running', styles=not_updated)
# tuning_title = Div(text='Tune hyperparameters', styles=updated)
tune_status_message = Div(text='Not running', styles=not_updated)
plot_status_message = Div(text='Plot not updated', styles=not_updated)
predict_status_message = Div(text = 'Not running', styles=not_updated)

# -------------------BUTTONS--------------------

save_config_button = Button(label="Save Current Configuration", button_type="success")
train_button = Button(label="Run ML algorithm", button_type="success")
tune_button = Button(label = "Tune", button_type = "success")
save_plot_button = Button(label="Save current plot", button_type="warning")
display_save_button = Button(label = "Display save")
predict_button = Button(label = 'Predict')

# -----------------INSTRUCTIONS-----------------

data_instr = Div(text="""Use the <b>slider</b> to split the data into <i>train/validate/test</i> percentages,
                       and <b>select/deselect</b> property columns for training the model. 
                       You can see the graphical relationship between any two properties in the plot below.
                       Finally, <b>save</b> the configuration.""",
width=200, height=140)

train_instr = Div(text="""Select and run one of the following <b>Machine Learning algorithms</b>""",
width=200, height=50)

# This will likely be changed when validation and testing are two different buttons
tune_instr = Div(text="""Change <b>hyperparameters</b> based on your chosen ML algorithm, 
                        and click <b>tune</b> to compare the tuned model's <b>validation accuracies</b> to the untuned model 
                        on the boxplot. You can <b>save</b> any model at any time and <b>display</b> any saved model on the plot""",
width=200, height=120)

test_instr = Div(text="""TEST INSTRUCTIONS GO HERE""",
width=200, height=100)


# --------------- DATA SELECTION ---------------

# Load data from the csv file
file_path = r'biodegrad.csv'
df = pd.read_csv(file_path, low_memory=False)
df_display = df.iloc[:,:22] #don't need to display the other 167 rows of fingerprint bits
df = df.drop(columns=['Fingerprint List']) #removing the display column, won't be useful in training

# Columns that should always be shown
mandatory_columns = ['Substance Name', 'Smiles', 'Class']

# Ensure mandatory columns exist in the dataframe (if not, create dummy columns) (hopefully shouldn't have to apply)
for col in mandatory_columns:
    if col not in df_display.columns:
        df_display[col] = "N/A"

# saved list to append to
user_columns = []

# Limit the dataframe to the first 10 rows
df_subset = df_display.head(10)

df_dict = df_subset.to_dict("list")
cols = list(df_dict.keys())

# Separate mandatory and optional columns
optional_columns = [col for col in cols if col not in mandatory_columns]

# Create column datasource
data_tab_source = ColumnDataSource(data=df_subset)

# Create figure
data_tab_columns = [TableColumn(field=col, title=col, width = 100) for col in cols]
data_tab_table = DataTable(source=data_tab_source, columns=data_tab_columns, width=900, autosize_mode = "none")

# Create widget excluding mandatory columns
checkbox_button_group = CheckboxButtonGroup(labels=optional_columns, active=list(range(len(optional_columns))), orientation = 'vertical')

# Update columns to display
def update_cols(display_columns):
    # Always include mandatory columns
    all_columns = mandatory_columns + display_columns
    data_tab_table.columns = [col for col in data_tab_columns if col.title in all_columns]

def update_table(attr, old, new):
    cols_to_display = [checkbox_button_group.labels[i] for i in checkbox_button_group.active]
    update_cols(display_columns=cols_to_display)
    save_config_message.text = 'Configuration not saved'
    save_config_message.styles = not_updated


# --------------- DATA SPLIT ---------------

# saved split list to write to
split_list = [50,25,25] #0-train, 1-val, 2-test

# helper function to produce string
def update_text(train_percentage, val_percentage, test_percentage):
    split_display.text = f"<div>Training split: {train_percentage}</div><div>Validation split: {val_percentage}</div><div>Testing split: {test_percentage}</div>"

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
tvt_slider = RangeSlider(title="Train-Validate-Test (%)", value=(50, 75), start=0, end=100, step=5, tooltips = False, show_value = False)
tvt_slider.bar_color = '#FAFAFA' # may change later, just so that the segments of the bar look the same
split_display = Div(text="<div>Training split: 50</div><div>Validation split: 25</div><div>Testing split: 25</div>")

# --------------- INTERACTIVE DATA VISUALIZATION GRAPH --------------- 

# get columns
data_vis_columns = df.columns.tolist()[:21]

#columns to exclude
data_vis_columns = [col for col in data_vis_columns if col not in ["Class", "Smiles", "Substance Name", "Fingerprint"]]

#convert the class columns to a categorical column if it's not
df['Class'] = df['Class'].astype('category')
# print(df.iloc[312])

# Create a ColumnDataSource
data_vis_source = ColumnDataSource(data=dict(x=[], y=[], class_color=[]))

# Create a figure
data_vis = figure(title="Data Exploration Scatter Plot - search for correlations between numeric variables", x_axis_label='X', y_axis_label='Y', 
           tools="pan,wheel_zoom,box_zoom,reset,hover,save")

# Create an initial scatter plot
data_vis_scatter = data_vis.scatter(x='x', y='y', color='class_color', source=data_vis_source, legend_field='class_color')

# Create dropdown menus for X and Y axis
select_x = Select(title="X Axis", value=data_vis_columns[0], options=data_vis_columns)
select_y = Select(title="Y Axis", value=data_vis_columns[1], options=data_vis_columns)

# Update the data based on the selections
def update_data_vis(attrname, old, new):
    x = select_x.value
    y = select_y.value
    new_vis_data = {
        'x': df[x],
        'y': df[y],
        'class_color': [Category10[3][0] if cls == df['Class'].cat.categories[0] else Category10[3][1] for cls in df['Class']]
    }
        
    # Update the ColumnDataSource with a plain Python dict
    data_vis_source.data = new_vis_data
    
    # Update existing scatter plot glyph if needed
    data_vis_scatter.data_source.data = new_vis_data
    
    data_vis.xaxis.axis_label = x
    data_vis.yaxis.axis_label = y

# Attach the update_data function to the dropdowns
select_x.on_change('value', update_data_vis)
select_y.on_change('value', update_data_vis)

update_data_vis(None, None, None)


# --------------- SAVE DATA BUTTON ---------------

# table on change
checkbox_button_group.on_change('active', update_table)

# range slider on change
tvt_slider.js_on_change('value', callback)
tvt_slider.on_change('value', update_values)

# Save columns to saved list (split already saved)
def save_config():
    temp_columns = [checkbox_button_group.labels[i] for i in checkbox_button_group.active]
    global user_columns
    user_columns.clear()
    user_columns = temp_columns

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

    plot_status_message.text = 'Plot not updated'
    plot_status_message.styles=not_updated
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
accuracy_display = Div(text="<div><b>Validation Accuracy:</b> N/A</div><div><b>Test Accuracy:</b> N/A</div>")
test_accuracy = []

# Create empty lists
val_accuracy = [None for i in range(10)]
tuned_val_accuracy = [None for i in range(10)]
saved_accuracy = [None for i in range(10)]
combo_list = val_accuracy + tuned_val_accuracy + saved_accuracy

def run_ML():
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
    combo_list = val_accuracy + [None for i in range(10)] + saved_accuracy

    update_boxplot()

    # Updating accuracy display
    accuracy_display.text = f"<div><b>Validation Accuracy:</b> {val_accuracy}</div><div><b>Test Accuracy:</b> {test_accuracy}</div>"

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
            val_accuracy.append(round(accuracy_score(y_val, y_val_pred), 2))
            test_accuracy.append(round(accuracy_score(y_test, y_test_pred), 2))
        elif stage == 'Tune':
            tuned_val_accuracy.append(round(accuracy_score(y_val, y_val_pred), 2))
            tuned_test_accuracy.append(round(accuracy_score(y_test, y_test_pred), 2))

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
## decision tree - int/None, string
## KNN - int, string
## SVC - int, ""
hyperparam_list = [None,"best"]

# Create empty lists
tuned_test_accuracy = []

# create displays
tuned_accuracy_display = Div(text = "<div><b>Tuned Validation Accuracy:</b> N/A</div><div><b>Tuned Test Accuracy:</b> N/A</div>")

def run_tuned_config():
    global my_alg, stage
    stage = 'Tune'
    tune_status_message.text = f'Algorithm: {my_alg}'
    tune_status_message.styles = updated
    global model

    split_and_train_model(split_list[0],split_list[1],split_list[2], user_columns)


    # Changing the list used to create boxplot
    global combo_list
    combo_list[10:20] = tuned_val_accuracy.copy()

    update_boxplot()

    # Updating accuracy display
    tuned_accuracy_display.text = f"<div><b>Validation Accuracy:</b> {tuned_val_accuracy}</div><div><b>Test Accuracy:</b> {tuned_test_accuracy}</div>"

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
hp_toggle.margin = (24, 10, 24, 10)

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
            start = 500,
            end = 1500,
            value = 1000,
            step = 100
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
p = figure(x_range=['pretune', 'posttune', 'saved'],
            y_range = (0.0, 1.0),
            width = 500,
            height = 500,
            tools="",
            toolbar_location=None,
            background_fill_color="#eaefef",
            title="Validation Accuracies",
            y_axis_label="accuracy")

df_box = pd.DataFrame()
source = ColumnDataSource()

# Create status message Div
# saved_split_message = Div(text = 'Saved split: N/A', styles = not_updated)
# saved_col_message = Div(text='Saved columns: N/A', styles=not_updated)
# saved_alg_message = Div(text='Saved alg: N/A', styles=not_updated)
# saved_data_message = Div(text='Saved val acc: N/A', styles=not_updated)

def update_df_box():
    # making the initial boxplot
    global combo_list
    d = {'kind': ['pretune' for i in range(10)] + ['posttune' for j in range(10)] + ['saved' for l in range(10)],
        'accuracy': combo_list
        }
    
    # print(d)
    global df_box
    df_box = pd.DataFrame(data=d)

    # compute quantiles
    qs = df_box.groupby("kind").accuracy.quantile([0.25, 0.5, 0.75])
    qs = qs.unstack().reset_index()
    qs.columns = ["kind", "q1", "q2", "q3"]
    df_box = pd.merge(df_box, qs, on="kind", how="left")

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

    # update outliers
    global outliers
    outliers = ColumnDataSource(df_box[~df_box.accuracy.between(df_box.lower, df_box.upper)])
    
    global source
    source.data = dict(df_box)

    plot_status_message.text = 'Plot updated'
    plot_status_message.styles = updated

def get_minmax(kind):
    if kind == 'pretune':
        if combo_list[0] == None: # when module is first loaded
            return 0,0
        else:
            combo_list_pretune = combo_list[:10]
            abs_min = min(combo_list_pretune)
            while abs_min < df_box["lower"][0]:
                combo_list_pretune.remove(abs_min)
                abs_min = min(combo_list_pretune)

            abs_max = max(combo_list_pretune)
            while abs_max > df_box["upper"][0]:
                combo_list_pretune.remove(abs_max)
                abs_max = max(combo_list_pretune)

            return abs_min, abs_max
    elif kind == 'posttune':
        if combo_list[10] == None: # when module is first loaded
            return 0,0
        else:
            combo_list_posttune = combo_list[10:20]
            abs_min = min(combo_list_posttune)
            while abs_min < df_box["lower"][10]:
                combo_list_posttune.remove(abs_min)
                abs_min = min(combo_list_posttune)

            abs_max = max(combo_list_posttune)
            while abs_max > df_box["upper"][10]:
                combo_list_posttune.remove(abs_max)
                abs_max = max(combo_list_posttune)

            return abs_min, abs_max
    elif kind == 'saved':
        if combo_list[-1] == None:
            return 0,0
        else:
            combo_list_saved = combo_list[20:]
            abs_min = min(combo_list_saved)
            while abs_min < df_box["lower"][20]:
                combo_list_saved.remove(abs_min)
                abs_min = min(combo_list_saved)

            abs_max = max(combo_list_saved)
            while abs_max > df_box["upper"][20]:
                combo_list_saved.remove(abs_max)
                abs_max = max(combo_list_saved)

            return abs_min, abs_max

def make_glyphs():
    # make all of the glyphs
    # outlier range
    global whisker
    global outlier_points
    whisker = Whisker(base="kind", upper="max", lower="min", source=source)
    whisker.upper_head.size = whisker.lower_head.size = 20
    p.add_layout(whisker)

    outlier_points = p.scatter("kind", "accuracy", source=outliers, size=6, color="black", alpha=0.3)

    # quantile boxes
    global kinds, cmap, top_box, bottom_box
    kinds = df_box.kind.unique()
    cmap = factor_cmap("kind", "Paired3", kinds)
    top_box = p.vbar(x = "kind", width = 0.7, bottom = "q2", top = "q3", color=cmap, line_color="black", source = source)
    bottom_box = p.vbar("kind", 0.7, "q1", "q2", color=cmap, line_color="black", source = source)

    # constant plot features
    p.xgrid.grid_line_color = None
    p.axis.major_label_text_font_size="14px"
    p.axis.axis_label_text_font_size="12px"

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
display_save_select = Select(title = "Choose a save to display", options = [])
predict_select = Select(title = 'Choose a save to predict with', options = [])

new_save_number = 0

# Define an empty data source
saved_data = dict(
    save_number = [],
    train_val_test_split = [],
    saved_columns = [],
    saved_algorithm = [],
    saved_hyperparams = [],
    saved_val_acc = []
)
save_source = ColumnDataSource(saved_data)

# Define table columns
saved_columns = [
    TableColumn(field="save_number", title="#", width = 25),
    TableColumn(field="train_val_test_split", title="Train/Val/Test split", width = 220),
    TableColumn(field="saved_columns", title="Saved col."),
    TableColumn(field="saved_algorithm", title="Saved alg.", width = 140),
    TableColumn(field="saved_hyperparams", title="Saved hp.", width = 220),
    TableColumn(field="saved_val_acc", title="Val. accuracies")
]

# Create a DataTable
saved_data_table = DataTable(source=save_source, columns=saved_columns, width=600, height=280, index_position=None)


def save_plot():
    global combo_list
    # global saved_accuracy
    global hyperparam_list
    global new_save_number
    global new_train_val_test_split
    global new_saved_columns
    global new_saved_algorithm
    global new_saved_hyperparams
    global new_saved_val_acc

    # saved_accuracy.clear()
    # saved_accuracy = combo_list[10:20]

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
    new_saved_val_acc = combo_list[10:20]

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
        'saved_val_acc' : [new_saved_val_acc]
    }
    save_source.stream(new_saved_data)

def load_save():
    plot_status_message.text = 'Updating saved data...'
    plot_status_message.styles = loading
    curdoc().add_next_tick_callback(save_plot)

# Attach callback to the save_plot button
save_plot_button.on_click(load_save)


def display_save():
    global saved_accuracy, saved_data_table, combo_list
    saved_accuracy = save_source.data['saved_val_acc'][int(display_save_select.value)-1]

    combo_list[20:] = saved_accuracy
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

def predict_biodegrad():
    temp_tvt_list = new_train_val_test_split.split("/")
    temp_train = int(temp_tvt_list[0])
    temp_val = int(temp_tvt_list[1])
    temp_test = int(temp_tvt_list[2])

    temp_cols = save_source.data['saved_columns'][int(predict_select.value)-1]
    split_and_train_model(temp_train,temp_val,temp_test, temp_cols)

    user_molec = Chem.MolFromSmiles(user_smiles_input.value)
    user_fp = np.array(MACCSkeys.GenMACCSKeys(user_molec))
    user_df = pd.DataFrame(user_fp)
    user_df = user_df.transpose() #each bit has its own column

    user_biodegrad = model.predict(user_df)

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

# --------------- LAYOUTS ---------------

# creating widget layouts
table_layout = column(row(checkbox_button_group, data_tab_table))
slider_layout = column(tvt_slider, split_display, save_config_button, save_config_message)
interactive_graph = column(row(select_x, select_y), data_vis) #create data graph visualization 
tab1_layout = column(row(column(data_instr, slider_layout), table_layout), interactive_graph)
tab2_layout = column(train_instr, alg_select, train_button, train_status_message, accuracy_display)
hyperparam_layout = column(row(hp_slider, hp_toggle), hp_select, tune_button, tune_status_message, tuned_accuracy_display, save_plot_button)
plot_layout = column(p, plot_status_message, display_save_select, display_save_button)
tab3_layout = row(column(tune_instr, hyperparam_layout), plot_layout, saved_data_table)
tab4_layout = column(test_instr, user_smiles_input, predict_select, predict_button, predict_status_message)

tabs = Tabs(tabs = [TabPanel(child = tab1_layout, title = 'Data'),
                    TabPanel(child = tab2_layout, title = 'Train'),
                    TabPanel(child = tab3_layout, title = 'Fine-Tune'),
                    TabPanel(child = tab4_layout, title = 'Test')
                ])


# just to see the elements
# test_layout = column(tab1_layout, tab2_layout, hyperparam_layout, plot_layout)
curdoc().add_root(tabs)