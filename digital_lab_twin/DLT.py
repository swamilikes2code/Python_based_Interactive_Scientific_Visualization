
#needed for this to work: working directory should have a models folder with mmscalerX.pkl and mmscalerY.pkl and model.pt, and an outputs folder
# imports, put these at the very top of everything
import matplotlib.pyplot as plt
import matplotlib
import tqdm
import copy
import modularNN as mnn
import numpy as np
import pandas as pd
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import joblib
import pathlib as path
import os
from bokeh.plotting import figure, show, curdoc
from bokeh.layouts import row, column
from bokeh.io import curdoc
from bokeh.layouts import layout, Spacer
from bokeh.io import export_svgs
from bokeh.models.dom import HTML
import datetime
from bokeh.models import ColumnDataSource, HoverTool, Slider, CustomJS, TabPanel, Tabs, Div, Paragraph, Button, Select, RadioButtonGroup, NumericInput, DataTable, StringFormatter, TableColumn, TextInput, HelpButton, Tooltip, NumberFormatter
import warnings
import random
import time
from os.path import dirname, join
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names, but StandardScaler was fitted with feature names")
#Instrutions Tab Section_____________________________________________________________________________________________________________________
#Intro Text sectionSection ---------------------------------------------------------------------------------------------------------------------
#This is the intro text section, it is a div, which is a bokeh object that allows you to write html and css in python
#headers are made with <h1> </h1> tags, and paragraphs are made with <p> </p> tags, headers are automatically bolded
#prefix holder so any file calls are relative to the current directory
prefix = 'digital_lab_twin/'
#boolean for if on master or not--false will keep things presuming you are running from dlt folder, otherwise append prefix
master = True
# master = False
#when running, ask user if they are running from master or not
"""master = input("Are you running from master? (y/n): ")
if master == 'y':z
    master = True
else:
    master = False"""

# ------- STATUS MESSAGES --------
not_updated = {'color': 'red', 'font-size': '14px'}
loading = {'color': 'orange', 'font-size': '14px'}
updated = {'color': 'green', 'font-size': '14px'}
run_button_status_message = Div(text='Not running', styles=not_updated)
run_button_edit_tab_status_message = Div(text='Not running', styles=not_updated)

def train_status_message_callback(attr, old, new):
    run_button_edit_tab_status_message.text = 'Not running'
    run_button_edit_tab_status_message.styles = not_updated

def optimize_status_message_callback(attr, old, new):
    run_button_status_message.text = 'Not running'
    run_button_status_message.styles = not_updated

intro = Div(text="""
       
        <h3>Simple Photobioreactor Summary</h3>
        <p>A photobioreactor is a container, like a fish tank, filled with water and special microscopic plants called algae. 
        It provides the algae with light, nutrients, and carbon dioxide to help them grow. 
        The algae use sunlight to make their own food through a process called photosynthesis. 
        The photobioreactor allows us to grow algae and use their biomass to produce clean and renewable energy,
        like biofuels. It also helps clean up the environment by absorbing harmful pollutants, such as carbon dioxide.
        In simple terms, a photobioreactor is a special container that helps tiny plants called algae grow using
        light, nutrients, and carbon dioxide to make clean energy and help the planet.(section for the paragraph).</p>
        
        <h3>Sliders</h3>
        <p> <b>To generate data, you can change the following values with the coresponding slider</b>:<br>
            Initial Nitrate Concentration: Set the initial nitrate concentration in g/L (0.2 - 2)<br>
            Initial Biomass Concentration: Set the initial biomass concentration in g/L (0.2 - 2)<br>
            Inlet Flow: Control the inlet flow of nitrate into the reactor (0.001 - 0.015 L/h)<br>
            Inlet Concentration: Inlet concentration of nitrate feed to the reactor (5 - 15 g/L)<br>
            Light Intensity: Control the light intensity in the range of 100 - 200 umol/m2-s </p>

         <h3>Hover Tool</h3>
        <p> Hover the cursor over the line and you will be able to the element name, time, and  element concentration <br>
            <b><u>Note</u></b>: the Lutien concentration is 1000 times greater than what the actual concentration is, so that you are able to see the Lutine curve</p>
        
         <h3>Reset Button</h3>
        <p> This Button will reset the graph to the original position based on the initial conditions before the sliders were changed</p>
        
         <h3>Run Button</h3>
        <p> This Button will take the slider conditions that you have and will create a new plot based on those new conditions<br>
        <b><u>Note</u></b>: There are two run buttons the run button on the Train Tab changes the graph in the Train tab and the Optimize tab, <br>
        the run button on the Evaluate Tab on will only change the graph of the evaluate tab</p>
        
         <h3>Export Button</h3>
        <p> This Button will take the data points of the Time, Nitrate Concentration, Biomass concentration, and Lutine concentration<br>
        and put them in a csv file and this csv file will be located in your downloads folder the file will be named "exported_data_{timestamp}.csv"<br>
        the timestamp is the current time and will be formated as year-month-day-hour-minuete-second</p>
        
         <h3>Help Button</h3>
        <p> In the Train tab you can see little question mark buttons next to the interactive elements, these buttons will give you <br>
        information on what this tool is used for and how it will can change your graph</p>
        
        
        
    """)


#Predecition Tab Section_____________________________________________________________________________________________________________________
# Defining Values ---------------------------------------------------------------------------------------------------------------------
#C_X is Biomass
#C_N is Nitrate
#C_L is Lutine
#Time is in hours
#To generate data, just change these values in this block (perhaps in a loop), In my opinion, 
#a good range for C_x0 (which is the initial concnentration of biomass in the reactor C_X) is 0.2 - 2 g/L
# a good range for C_N0 (which is the initial concnetraiton of nitrate in the reactor C_N) is 0.2 - 2 g/L
# a good range for F_in (the inlet flow rate of nitrate into the reactor) is 1e-3 1.5e-2 L/h
# a good range for C_N_in (the inlet concentration of nitrate feed to the reactor) is 5 - 15 g/L
# a good range for intensity of light is 100 - 200 umol/m2-s

U_O = 0.152 # 1/h  Cell specific growth rate
U_D = 5.95*1e-3 # 1/h  Cell specific decay rate
K_N = 30.0*1e-3 # g/L  nitrate half-velocity constant

Y_NX = 0.305 # g/g  nitrate yield coefficient

K_O = 0.350*1e-3*2 #g/g-h Lutine synthesis rate

K_D = 3.71*0.05/90 # L/g-h  lutein consumption rate constant

K_NL = 10.0*1e-3 # g/L  nitrate half- velocity constant for lutein synthesis



# curdoc().theme = "dark_minimal"# this makes the graph in dark mode

#Creating Sliders ---------------------------------------------------------------------------------------------------------------------
light_intensity = Slider(start=100, end=200, value=150, step= 1, title="Light Intensity (umol/m2-s):(100 - 200)")
inlet_flow = Slider(start=0.001, end=0.015, value= 0.008, step=.0001, format = "0.000", title="Inlet Flow of Nitrates(L/h):(0.001 - 0.015)")
#pH = Slider(start=0.1, end=9, value=0.5, step=.1, title="PH")
inlet_concentration = Slider(start=5, end=15, value=10, step=.1, title="Inlet Concentration of Nitrates(g/L):(5 - 15)")
nitrate_con = Slider(start=0.2, end=2, value=1, step=.05, title="Initial Nitrate Concentration(g/L):(0.2 - 2)")
biomass_con = Slider(start=0.2, end=2, value=0.5, step=.05, title="Initial Biomass Concentration(g/L):(0.2 - 2)")
light_intensity.on_change('value', optimize_status_message_callback)
inlet_flow.on_change('value', optimize_status_message_callback)
inlet_concentration.on_change('value', optimize_status_message_callback)
nitrate_con.on_change('value', optimize_status_message_callback)
biomass_con.on_change('value', optimize_status_message_callback)


#pytorch Preloop  section ---------------------------------------------------------------------------------------------------------------------
#initial run so bokeh plot is not empty
modelLocation = 'models/model.pt'
if master:
    modelLocation = prefix + modelLocation
model = torch.load(modelLocation)
model.eval()
#scalers
stScalerXLocation = 'models/stScalerX.pkl'
stScalerYLocation = 'models/stScalerY.pkl'
if master:
    stScalerXLocation = prefix + stScalerXLocation
    stScalerYLocation = prefix + stScalerYLocation
stScalerX = joblib.load(stScalerXLocation)
stScalerY = joblib.load(stScalerYLocation)


#function takes in initial conditions and runs the model
#overwrites XDF with the predicted values
#updates bokeh plot with new values
#call when run button is hit
def predLoop(C_X, C_N, C_L, F_in, C_N_in, I0):
    # initialize everything 
    # note that these load funcs will need you to change to your current directory here!
    #print(os.getcwd() )
    #os.chdir('C:\\Users\\kenda\\Documents\\GitHub\\Python_based_Interactive_Scientific_Visualization\\digital_lab_twin') #Windows version
    #os.chdir('/Users/tyreesedavidson/Documents/GitHub/Python_based_Interactive_Scientific_Visualization/digital_lab_twin') #Mac version
    XTimes = np.linspace(0, 150, 200)
    XDF = pd.DataFrame(XTimes, columns=['Time'])
    #generate empty columns
    XDF['C_X'] = np.zeros(200)
    XDF['C_N'] = np.zeros(200)
    XDF['C_L'] = np.zeros(200)
    XDF['F_in'] = np.zeros(200)
    XDF['C_N_in'] = np.zeros(200)
    XDF['I0'] = np.zeros(200)
    XTimes = XDF.pop('Time') #popped for plotting
    #write init conditions to df
    #Only write to the first row for these 3, they'll be filled in thru the loop
    XDF['C_X'][0] = C_X
    XDF['C_N'][0] = C_N
    XDF['C_L'][0] = C_L
    #write to all rows for these 3, they won't be filled in thru the loop
    XDF['F_in'] = F_in
    XDF['C_N_in'] = C_N_in
    XDF['I0'] = I0

    #loop through the experiment and predict each timestep
    for i in range(0, 199):
        #get the current timestep
        X_current = XDF.iloc[i]
        #scale the current timestep
        X_current_scaled = stScalerX.transform([X_current])
        #predict the next timestep
        Y_current_scaled = model(torch.tensor(X_current_scaled, dtype=torch.float32))
        #unscale the prediction
        Y_current_scaled = Y_current_scaled.detach().numpy()
        Y_current = stScalerY.inverse_transform(Y_current_scaled)
        #store the prediction
        nextTimeStep = i+1
        XDF.iloc[nextTimeStep, 0] = Y_current[0,0]
        XDF.iloc[nextTimeStep, 1] = Y_current[0,1]
        XDF.iloc[nextTimeStep, 2] = Y_current[0,2]
    #after this loop, XDF should be filled with the predicted values
    #export XDF to csv
    #add times back in
    XDF['Time'] = XTimes
    return XDF
    #XDF.to_csv('outputs/prediction.csv', index=False)
    #TODO: re-call the plotting function to show results to user

# predLoop(biomass_con.value, nitrate_con.value, 0, inlet_flow.value, inlet_concentration.value, light_intensity.value)
#testing with default values
#predLoop(C_X_init, C_N_init, C_L_init, F_in_init, C_N_in_init, I0_init)




#Data Generation Section ---------------------------------------------------------------------------------------------------------------------

data = "outputs/prediction.csv"
if master:
    data = prefix + data
datas = pandas.read_csv(data)
source = ColumnDataSource(datas)
#initial Data  for reset section ---------------------------------------------------------------------------------------------------------------------
initial_csv1 = "outputs/prediction.csv"
if master:
    initial_csv1 = prefix + initial_csv1
initial_data = pandas.read_csv(initial_csv1)
initial_source = ColumnDataSource(initial_data)

#Plotting Function Section ---------------------------------------------------------------------------------------------------------------------
p = figure(title = "Change in concentrations over time in a photobioreactor", x_axis_label = "Time(hours)", y_axis_label = "concentration", width=400, height=370)

def plot_graph(sources):
    #Removes previous lines and hover tools
    p.renderers = [] #removes previous lines
    p.tools = [] #removes previous hover tools
    
    
    # Example of updating CL value

    line_a = p.line('Time', 'C_X', source = sources, line_width = 4 ,  line_color = "aqua", legend_label = "Biomass (g/L)")
    p.add_tools(HoverTool(renderers = [line_a], tooltips=[  ('Name', 'Biomass (g/L)'),
                                    ('Hour', '@Time'),
                                    ('Concentration', '@C_X'),# adds the hover tool to the graph for the specifed line
    ],))

    line_b = p.line('Time', 'C_N', source = sources, line_width = 4 , line_color = "orange", legend_label = "Nitrate (g/L)")
    p.add_tools(HoverTool( renderers = [line_b],tooltips=[('Name', 'Nitrate (g/L)'),
                                    ('Hour', '@Time'), 
                                    ('Concentration', '@C_N'), 
    ],))
    sources.data['modified_C_L'] = sources.data['C_L'] * 1000# CL is multiplied by 1000 to make it visible on the graph and this is done wih the column data source
    line_c = p.line('Time', 'modified_C_L', source = sources , line_width = 4, line_color = "lime",  legend_label = "Lutein (g/L x 1000)")# CL is multiplied by 1000 to make it visible on the graph
    p.add_tools(HoverTool( renderers = [line_c],tooltips=[('Name', 'Lutein (g/L)'),
                                    ('Hour', '@Time'), 
                                    ('Concentration', '@modified_C_L'), 
    ],)) 
   


    return p

p = plot_graph(source)

# display legend in top left corner (default is top right corner)
p.legend.location = "top_left"

# add a title to your legend
p.legend.title = "Elements"

# change appearance of legend text
p.legend.label_text_font = "times"
p.legend.label_text_font_style = "italic"
p.legend.label_text_color = "navy"

# change border and background of legend
p.legend.border_line_width = 3
p.legend.border_line_color = "black"
p.legend.border_line_alpha = 0.8
p.legend.background_fill_color = "white"
p.legend.background_fill_alpha = 0.5




p.toolbar.autohide = True



# Add the Slider to the figure ---------------------------------------------------------------------------------------------------------------------


slides = column(light_intensity, inlet_flow,  inlet_concentration, nitrate_con, biomass_con) #pH,


# Creating the Button---------------------------------------------------------------------------------------------------------------------

#Reset Button******************************************************************************************************************************
reset_button = Button(label = "Reset", button_type = "danger", height = 60, width = 300)

def rest_button_callback():
    light_intensity.value = 150
    nitrate_con.value = 1
    biomass_con.value = 0.5
    inlet_concentration.value = 10
    inlet_flow.value = 0.008
    run_button_status_message.text = 'Configuration Reset'
    run_button_status_message.styles = updated



def load_rest_button_callback():
    run_button_status_message.text = 'Resetting...'
    run_button_status_message.styles = loading
    rest_button_callback()

reset_button.on_click(load_rest_button_callback)




 
#Export Button******************************************************************************************************************************
# File Export Data Area



def export_data():
    # Get the data from the ColumnDataSource
    data = source.data
    day = data['Time']#data for time
    biomass = data['C_X']#data for biomass
    nitrate = data['C_N']#data for nitrate
    lutine = data['C_L']#data for lutine


    # Create a dictionary to hold the data
    export_data = {'Time': day, 'Biomass': biomass, 'Nitrate': nitrate, 'Lutein': lutine}

     # Generate a unique filename using current date and time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"exported_data_{timestamp}.csv"

        # Get the path to the user's downloads folder
    downloads_folder = os.path.expanduser("~/Downloads")

    # Create the file path
    file_path = os.path.join(downloads_folder, filename)

    # Export the data as a CSV file to the downloads folder
    with open(file_path, 'w') as f:
        f.write('Time,Biomass,Nitrate,Lutein\n')
        for i in range(len(day)):
            f.write(f'{day[i]},{biomass[i]},{nitrate[i]},{lutine[i]}\n')





export_button = Button(label="Export Data", button_type="success",  height = 60, width = 300)


#Run Button******************************************************************************************************************************
run_button = Button(label = "Run", button_type = "primary", height = 60, width = 300)

def runbutton_function(li = light_intensity, inf = inlet_flow,  inc = inlet_concentration, nit = nitrate_con, bio = biomass_con, ):  #pH = pH,
    
    #set initial conditions by changing these vals
    C_X_init = bio.value
    C_N_init = nit.value
    C_L_init = 0.0
    F_in_init = inf.value
    C_N_in_init = inc.value
    I0_init = li.value
    # p = figure() PREDLOOP IDEA EVERYTIME THE FUNCTION IS CALLED IT ALSO PASSES IN A NEW FIGURE THAT IS CREATED


    # print((C_X_init, C_N_init, C_L_init, F_in_init, C_N_in_init, I0_init))
    datas = predLoop(C_X_init, C_N_init, C_L_init, F_in_init, C_N_in_init, I0_init)
    #creates the source for the graph that the new plot will be based on
    #data = "outputs/prediction.csv"
    #datas = pandas.read_csv(data)
    sourceS = ColumnDataSource(datas)
    #attempt to reset the graph IDEA: ADD     p.renderers = [] AT THE BEGINNING OF THE PLOTTING FUNCTION
    plot_graph(sourceS) ######this is the new plot that will be shown YOU NEED TO FIX THIS SO THAT THE FIGURE IS UPDATED
    export_button.js_on_event("button_click", CustomJS(args=dict(source=sourceS),
                            code=open(join(dirname(__file__), "download.js")).read()))
    run_button_status_message.text = 'Configuration Ran'
    run_button_status_message.styles = updated
    

def load_runbutton_function():
    run_button_status_message.text = 'Running...'
    run_button_status_message.styles = loading
    runbutton_function(li = light_intensity, inf = inlet_flow,  inc = inlet_concentration, nit = nitrate_con, bio = biomass_con,)

run_button.on_click(load_runbutton_function)


#Edit Tab Section______________________________________________________________________________________________________________________________
#Model Inputs Section-----------------------------------------------------------------------------------------------
optimizer_options = ["ADAM", "SGD"]
loss_options = ["MSE", "MAE"]

#test = NumericInput(value=0.2, high = 100, low = 0, mode = "float", title="Test Split:(0 - 1)")# 
# changed direction of helpbuttons from right to left (better when viewed in portrait mode)

train = NumericInput(value=0.6, high = 0.7, low = 0.1, mode = "float", title="Train Split:(0.1-0.7)", )# 
train.on_change('value', train_status_message_callback)
# train_tooltip = Tooltip(content=("""Determine as a percentage how much of the data will be used to teach the model. What is left out of training will be used to validate and test."""), position = "left")
# train_help_button = HelpButton(tooltip=train_tooltip, button_type = "light", )
train_help_button = HelpButton(tooltip=Tooltip(content=HTML("""
                 <div style='padding: 16px; font-family: Arial, sans-serif; width: 180px;'>
                 <div>Determine as a percentage how much of the data will be used to teach the model. What is left out of training will be used to validate and test.</div>
                 """), position="left"))

#val_split = NumericInput(value=0.2, high = 100, low = 0, mode = "float", title="Val Split:(0 - 1)")# 

neurons = Slider (start = 7, end = 50, value = 18, step = 1, title = "Number of Neurons")# 
neurons.on_change('value', train_status_message_callback)
# neurons_tooltip = Tooltip(content=("""Determine how dense each neural network layer is. The network contains 3 layers, with an activator function in between each. Denser networks are resource intensive, but thinner networks may compromise accuracy."""), position = "left")
# neurons_help_button = HelpButton(tooltip=neurons_tooltip, button_type = "light")
neurons_help_button = HelpButton(tooltip=Tooltip(content=HTML("""
                 <div style='padding: 16px; font-family: Arial, sans-serif; width: 180px;'>
                 <div>Determine how dense each neural network layer is. The network contains 3 layers, with an activator function in between each. Denser networks are resource intensive, but thinner networks may compromise accuracy.</div>
                 """), position="left"))

epochs = Slider (start = 5, end = 30, value = 25, step = 5, title = "Epochs")# 
epochs.on_change('value', train_status_message_callback)
# epochs_tooltip = Tooltip(content=("""Determine how many times the network will read over the training data. This heavily impacts the model’s processing time."""), position = "left")
# epochs_help_button = HelpButton(tooltip=epochs_tooltip, button_type = "light")
epochs_help_button = HelpButton(tooltip=Tooltip(content=HTML("""
                 <div style='padding: 16px; font-family: Arial, sans-serif; width: 180px;'>
                 <div>Determine how many times the network will read over the training data. This heavily impacts the model’s processing time.</div>
                 """), position="left"))
    
batch_Size = Slider (start = 25, end = 200, value = 25, step = 25, title = "Batch Size")# 
batch_Size.on_change('value', train_status_message_callback)
# batch_Size_tooltip = Tooltip(content=("""Determine how many datapoints to feed the network at one time. An ideal batch size will help optimize runtime and model accuracy."""), position = "left")
# batch_Size_help_button = HelpButton(tooltip=batch_Size_tooltip, button_type = "light")
batch_Size_help_button = HelpButton(tooltip=Tooltip(content=HTML("""
                 <div style='padding: 16px; font-family: Arial, sans-serif; width: 180px;'>
                 <div>Determine how many datapoints to feed the network at one time. An ideal batch size will help optimize runtime and model accuracy.</div>
                 """), position="left"))

learning_rate = NumericInput(value=0.001, high = 0.01, low = 0.0001, mode = "float", title="Learning Rate:(0.0001-0.01)")# Student chooses the learning rate
learning_rate.on_change('value', train_status_message_callback)
# learning_rate_tooltip = Tooltip(content=("""Choose a maximum value by which the optimizer may adjust neuron weights. The lower this is, the smaller the changes any given epoch will have on the model."""), position = "left")
# learning_rate_help_button = HelpButton(tooltip=learning_rate_tooltip, button_type = "light")
learning_rate_help_button = HelpButton(tooltip=Tooltip(content=HTML("""
                 <div style='padding: 16px; font-family: Arial, sans-serif; width: 180px;'>
                 <div>Choose a maximum value by which the optimizer may adjust neuron weights. The lower this is, the smaller the changes any given epoch will have on the model.</div>
                 """), position="left"))

loss_Fn = Select(title="Loss Function:", value="MAE", options= loss_options, height = 60, width = 300)# Student chooses the loss function
loss_Fn.on_change('value', train_status_message_callback)
# loss_Fn_tooltip = Tooltip(content=f"Choose an algorithm to measure the accuracy of your predictions. MSE judges by square loss, whereas MAE judges by absolute loss. ", position = "left") #{', '.join(loss_options)}
# loss_Fn_help_button = HelpButton(tooltip=loss_Fn_tooltip, button_type = "light")
loss_Fn_help_button = HelpButton(tooltip=Tooltip(content=HTML("""
                 <div style='padding: 16px; font-family: Arial, sans-serif; width: 180px;'>
                 <div>Choose an algorithm to measure the accuracy of your predictions. MSE judges by square loss, whereas MAE judges by absolute loss.</div>
                 """), position="left"))

optimizer = Select(title="Optimizer:", value="ADAM", options= optimizer_options, height = 60, width = 300)# Student chooses the optimizer 
optimizer.on_change('value', train_status_message_callback)
# O_tooltip = Tooltip(content=f"Choose an algorithm by which the neural network will adjust it’s inner neurons. Both choices can be efficient, but may require further tuning of other parameters.", position="left") # {', '.join(optimizer_options)}
# optimizer_help_button = HelpButton(tooltip=O_tooltip, button_type = "light")
optimizer_help_button = HelpButton(tooltip=Tooltip(content=HTML("""
                 <div style='padding: 16px; font-family: Arial, sans-serif; width: 180px;'>
                 <div>Choose an algorithm by which the neural network will adjust it’s inner neurons. Both choices can be efficient, but may require further tuning of other parameters.</div>
                 """), position="left"))

#Mean Square Error / Root Mean Square Error section--------------------------------------------------------------------------------------------------------------------
    
mean_squared_error = TextInput(value = str(0.0206), title = "MSE (Test)", width = 300, disabled = True)
root_mean_squared_error = TextInput(value = str(0.1437), title = "RMSE (Test)", width = 300, disabled = True)
lowest_mse_validation = TextInput(value = str(100), title = "Lowest Loss (Validation)", width = 300, disabled = True)
epoch_of_lowest_loss = TextInput(value = str('N/A'), title = "Epoch of Lowest Loss", width = 300, disabled = True)
#TODO: get the final MSE from valLoss, RMSE is sqrt of that, plot those out same as above





"""
# Define the callback function for the sliders
def validate_sum(attr, old, new):
    # Calculate the sum of the slider values
    total = test.value + train.value + val_split.value
    
    # Check if the sum is not equal to 1
    if total != 1:
        # Calculate the difference between the current sum and 1
        diff = total - 1
        
        # Find the slider with the maximum value
        max_slider = max(test, train, val_split, key=lambda s: s.value)
        
        # Adjust the value of the max_slider by the difference
        max_slider.value -= diff

# Attach the callback to the 'value' change event of each slider
test.on_change('value', validate_sum)
train.on_change('value', validate_sum)
val_split.on_change('value', validate_sum)"""


#Reset Buttton For Edit Tab Section -----------------------------------------------------------------------------------------------
reset_button_edit_tab = Button(label = "Reset", button_type = "danger", height = 60, width = 300)

def reset_button_edit_tab_function():
    learning_rate.value = 0.001
    loss_Fn.value = "MAE"
    optimizer.value = "ADAM"
    train.value = 0.6
    neurons.value = 18
    #val_split.value = 0.2
    #test.value = 0.2
    epochs.value = 25
    batch_Size.value = 25
    lowest_mse_validation.value = str(100)
    epoch_of_lowest_loss.value = str('N/A')
    p2.renderers = []
    run_button_edit_tab_status_message.text = 'Configuration Reset'
    run_button_edit_tab_status_message.styles = updated

def load_reset_button_edit_tab_function():
    run_button_edit_tab_status_message.text = 'Resetting...'
    run_button_edit_tab_status_message.styles = loading
    reset_button_edit_tab_function()

reset_button_edit_tab.on_click(load_reset_button_edit_tab_function)
    
#Model Loop section for edit tab_____________________________________________________________________________________________________________________

#the below code is designed to drag and drop into the bokeh visualization
#static section should run once on launch, dynamic section should run on each change
### Static (run once)
csvPath = 'STEMVisualsSynthData.csv'
if master:
    csvPath = prefix + csvPath
rawData = pd.read_csv(csvPath, header=0)
#remove unneeded column
rawData.drop('Index_within_Experiment', axis = 1, inplace = True)
#X is inputs--the three Concentrations, F_in, I0 (light intensity), and c_N_in (6)
X = rawData[['Time', 'C_X', 'C_N', 'C_L', 'F_in', 'C_N_in', 'I0']]
Y = X.copy(deep=True)
#drop unnecessary rows in Y
Y.drop('F_in', axis = 1, inplace = True)
Y.drop('C_N_in', axis = 1, inplace = True)
Y.drop('I0', axis = 1, inplace = True)
Y.drop('Time', axis = 1, inplace = True)
#Y vals should be X concentrations one timestep ahead, so remove the first index
Y.drop(index=0, inplace=True)
#To keep the two consistent, remove the last index of X
X.drop(index=19999, inplace=True)
#set device
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
lossCSVPath = 'models/losses.csv'
if master:
    lossCSVPath = prefix + lossCSVPath
lossCSV = pd.read_csv(lossCSVPath, header=0)

def model_loop(lR = learning_rate,  lFn = loss_Fn, opt = optimizer, tr = train,  n = neurons, e = epochs, b = batch_Size, X = X, Y = Y, device = device, optimizer_options = optimizer_options, loss_options = loss_options): #ts = test, vs = val_split,
  #create a timer for whole model loop
  modelStart = time.perf_counter()  
  #user defined parameters: current values can serve as a default
  #splits - expects 3 floats that add to 1
  trainSplit = tr.value
  #valSplit = vs.value
  #testSplit = ts.value
  #model params
  initNeuronNum = n.value #number of neurons in the first layer, 7 < int < 100
  loss = loss_options.index(lFn.value) #0 = MSE, 1 = MAE
  optimizer = optimizer_options.index(opt.value ) #0 = Adam, 1 = SGD
  learnRate = lR.value #0.0001 < float < 0.01
  #training params
  epochs = e.value #0 < int < 200
  batchSize = b.value #0 < int < 200
    
  ### Dynamic (run on each change)
  #test the all-in-one function
  ## model, Y_test_tensor, testPreds, XTestTime, lossDF, stScalerX, stScalerY, testPreds, mse, rmse= mnn.trainAndSaveModel(X, Y, trainSplit,  initNeuronNum, loss, optimizer, learnRate, epochs, batchSize, device) #valSplit, testSplit,
  #split the data
  X_train, X_val, X_test, Y_train, Y_val, Y_test, XTrainTime, XValTime, XTestTime = mnn.dataSplitter(X, Y, trainSplit)
  #scale the data
  stScalerX, stScalerY, X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled= mnn.scaleData(X_train, X_val, X_test, Y_train, Y_val, Y_test)
  #tensorize the data
  X_train_tensor, X_val_tensor, X_test_tensor, Y_train_tensor, Y_val_tensor, Y_test_tensor = mnn.tensors(X_train_scaled, X_val_scaled, X_test_scaled, Y_train_scaled, Y_val_scaled, Y_test_scaled)
  #if possible, move the tensors to the GPU
  X_train_tensor = X_train_tensor.to(device)
  X_val_tensor = X_val_tensor.to(device)
  X_test_tensor = X_test_tensor.to(device)
  Y_train_tensor = Y_train_tensor.to(device)
  Y_val_tensor = Y_val_tensor.to(device)
  Y_test_tensor = Y_test_tensor.to(device)

  #create the model
  model, lossFunction, optimizer = mnn.modelCreator(initNeuronNum, loss, optimizer, learnRate)
  #if possible, move the model to the GPU
  model = model.to(device)
  #train the model
  #model, trainLoss, valLoss = trainModel(model, lossFunction, optimizer, epochs, batchSize, X_train_tensor, X_val_tensor, Y_train_tensor, Y_val_tensor)
  batch_start = torch.arange(0, len(X_train_tensor), batchSize)
    
# Hold the best model
  best_mse = np.inf   # init to infinity
  best_weights = None
  trainLoss = []
  valLoss = []
  # starting the training, want to add a timer to see how long it takes (print only to console)
  timeStart = time.perf_counter()
  # training loop
  for epoch in range(epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train_tensor[start:start+batchSize]
                y_batch = Y_train_tensor[start:start+batchSize]
                # forward pass
                y_pred = model(X_batch)
                loss = lossFunction(y_pred, y_batch)
                # backward pass
                for param in model.parameters():
                    param.grad = None
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_train_tensor)
        mse = lossFunction(y_pred, Y_train_tensor)
        mse = float(mse)
        trainLoss.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())
        
        #validation loss
        y_pred = model(X_val_tensor)
        mse = lossFunction(y_pred, Y_val_tensor)
        mse = float(mse)
        valLoss.append(mse)
    
    # restore model and return best accuracy
  model.load_state_dict(best_weights)
  #time end
  timeEnd = time.perf_counter()
  #total time for training
  totalTime = timeEnd - timeStart
  print(f"Total time for training: {totalTime:0.4f} seconds")
  lossDF = mnn.saveLosses(trainLoss, valLoss)
  #saveModel(model, stScalerX, stScalerY)
  testPreds, mse, rmse = mnn.testPredictions(model, X_test_tensor, lossFunction, Y_test_tensor) #3 columns, 1 for each output (biomass/nitrate/lutein)
  #read in the losses
  lossCSV = lossDF
  #save testPreds to a csv
  #testPreds.to_csv('models/testPreds.csv', index=False) 
  #pick an experiment number (between zero and 100)
  experimentNum = random.randint(0, 100)
  #experimentnum times 200 will give the starting index, then add 199 to get the ending index
  start = experimentNum * 200
  end = start + 199
  #get the experiment data
  experimentDataX = X.iloc[start:end]
  experimentDataY = Y.iloc[start:end]
  #the first row of experimentDataX is the initial conditions, isolate them for use in predLoop
  initialConditions = experimentDataX.iloc[0]
  XTimes = np.linspace(0, 150, 200)
  XDF = pd.DataFrame(XTimes, columns=['Time'])
  #generate empty columns
  XDF['C_X'] = np.zeros(200)
  XDF['C_N'] = np.zeros(200)
  XDF['C_L'] = np.zeros(200)
  XDF['F_in'] = np.zeros(200)
  XDF['C_N_in'] = np.zeros(200)
  XDF['I0'] = np.zeros(200)
  XTimes = XDF.pop('Time') #popped for plotting
  #set initial conditions by referencing initialConditions
  C_X_init = initialConditions[1]
  C_N_init = initialConditions[2]
  C_L_init = initialConditions[3]
  F_in_init = initialConditions[4]
  C_N_in_init = initialConditions[5]
  I0_init = initialConditions[6]
  #call predLoop
  XDF = predLoop(C_X_init, C_N_init, C_L_init, F_in_init, C_N_in_init, I0_init)
  #add the experiment data to the XDF
  #remove the last row of XDF, it goes into the next experiment
  XDF.drop(index=199, inplace=True)
  XDF['C_X_actual'] = experimentDataY['C_X'].to_numpy()
  XDF['C_N_actual'] = experimentDataY['C_N'].to_numpy()
  XDF['C_L_actual'] = experimentDataY['C_L'].to_numpy()

  #export XDF to csv (for initial run)
  #XDF.to_csv('outputs/expPredVsDataset.csv', index=False)
  #model end timer
  modelEnd = time.perf_counter()
  #total time for model
  modelTime = modelEnd - modelStart
  print(f"Total time for model: {modelTime:0.4f} seconds")
  #show model time without training
  print(f"Total time for model without training: {modelTime - totalTime:0.4f} seconds")
  return lossDF, testPreds, mse, rmse, XDF

    
# #Loss Graph Data section ---------------------------------------------------------------------------------------------------------------------
# loss_data = "models/losses.csv"
# loss_datas = pandas.read_csv(loss_data)
# loss_source = ColumnDataSource(loss_datas)
#Loss Graph section ---------------------------------------------------------------------------------------------------------------------
p2 = figure(title = "Loss Graph (Training)", x_axis_label = "Epochs", y_axis_label = "Loss (percentage)", width=400, height=370)

def loss_graph(loss_data, p2): # function to plot the loss graph
    #Removes previous lines and hover tools
    p2.renderers = [] #removes previous lines
    p2.tools = [] #removes previous hover tools    
    #if loss data is not a string, then it is a dataframe
    if type(loss_data) != str:
        loss_datas = loss_data
    else:
        loss_datas = pandas.read_csv(loss_data)
        loss_datas['index'] = loss_datas.index
    loss_source = ColumnDataSource(loss_datas)
    # Example of updating CL value

    train_loss = p2.line('index', 'trainLoss', source = loss_source, line_width = 4 ,  line_color = "violet", legend_label = "Train Loss")
    p2.add_tools(HoverTool(renderers = [train_loss], tooltips=[  ('Name', 'Train Loss'),  ('Epochs', '@index'), ('Loss', '@trainLoss')],))
                                    # adds the hover tool to the graph for the specifed line
"""   
    value_loss = p2.line('index', 'valLoss', source = loss_source, line_width = 4 , line_color = "navy", legend_label = "Validation Loss")
    p2.add_tools(HoverTool(renderers = [value_loss], tooltips=[  ('Name', 'Validation Loss'), ('Epochs', '@index'), ('Loss', '@valLoss') ],))
"""
    
    # Add the lines to the plot
    
    
lossInitPath = 'models/losses.csv'
if master:
    lossInitPath = prefix + lossInitPath
loss_graph(lossInitPath, p2)

#Parity Plot section ---------------------------------------------------------------------------------------------------------------------
p3 = figure(title = "Parity Plot", x_axis_label = "Actual Concentration", y_axis_label = "Predicted Concentration", width=400, height=370)
def parity_plot(parity_data, p3): # function to plot the parity graph
    #Removes previous lines and hover tools
    p3.renderers = [] #removes previous lines
    p3.tools = [] #removes previous hover tools    
    #if parity data is not a string, then it is a dataframe
    if type(parity_data) != str:
        parity_datas = parity_data
    else:
        parity_datas = pandas.read_csv(parity_data)
        parity_datas['index'] = parity_datas.index
    parity_source = ColumnDataSource(parity_datas)
    # Example of updating CL value

    biomass = p3.scatter('Biomass_Actual', 'Biomass_Predicted', source = parity_source, size = 4 ,  fill_color = "aqua", legend_label = "Biomass", fill_alpha = 0.8, muted_color = 'aqua', muted_alpha=0.1,)
    p3.add_tools(HoverTool(renderers = [biomass], tooltips=[  ('Name', 'Biomass'),  ('Biomass Actual:', '@Biomass_Actual'), ('Biomass Predicted:', '@Biomass_Predicted')],) )
                                    # adds the hover tool to the graph for the specifed line
   
    nitrate = p3.scatter('Nitrate_Actual', 'Nitrate_Predicted', source = parity_source, size = 4 , fill_color = "orange", legend_label = "Nitrate", fill_alpha = 0.8, muted_color = 'orange', muted_alpha=0.1,)
    p3.add_tools(HoverTool(renderers = [nitrate], tooltips=[  ('Name', 'Nitrate'), ('Nitrate Actual:', '@Nitrate_Actual'), ('Nitrate Predicted:', '@Nitrate_Predicted') ],) )
    
    lutien = p3.scatter('Lutein_Actual', 'Lutein_Predicted', source = parity_source, size = 4 , fill_color = "lime", legend_label = "Lutein", fill_alpha = 0.8, muted_color = 'lime', muted_alpha=0.1,)
    p3.add_tools(HoverTool(renderers = [lutien], tooltips=[  ('Name', 'Lutein'), ('Lutein Actual:', '@Lutein_Actual'), ('Lutein Predicted:', '@Lutein_Predicted') ],) )
    
    # Add the lines to the plot
parityDataPath = 'models/testPreds.csv'   
if master:
    parityDataPath = prefix + parityDataPath
parity_plot(parityDataPath, p3)
p3.legend.click_policy="hide"
# display legend in top left corner (default is top right corner)
p3.legend.location = "top_left"

# change appearance of legend text
p3.legend.label_text_font = "times"
p3.legend.label_text_font_style = "italic"
p3.legend.label_text_color = "navy"

# change border and background of legend
p3.legend.border_line_width = 3
p3.legend.border_line_color = "black"
p3.legend.border_line_alpha = 0.8
p3.legend.background_fill_color = "white"
p3.legend.background_fill_alpha = 0.5

#Predictions vs Actual Plot section ---------------------------------------------------------------------------------------------------------------------
p4 = figure(title = "Prediction vs Actual Plot", x_axis_label = "Time", y_axis_label = "Concentration", width=400, height=370)
def versus_plot(vs_data, p4): # function to plot the parity graph
    #Removes previous lines and hover tools
    p4.renderers = [] #removes previous lines
    p4.tools = [] #removes previous hover tools    
    #if parity data is not a string, then it is a dataframe
    if type(vs_data) != str:
        vs_datas = vs_data
    else:
        vs_datas = pandas.read_csv(vs_data)
        vs_datas['index'] = vs_datas.index
    vs_source = ColumnDataSource(vs_datas)
    # Example of updating CL value

    biomass = p4.line('Time', 'C_X_actual', source = vs_source, line_width = 4 ,  line_color = "aqua", legend_label = "Biomass(Actual)", muted_color = 'aqua',)
    p4.add_tools(HoverTool(renderers = [biomass], tooltips=[  ('Name', 'Biomass'),  ('Time:', '@Time'), ('Concentration:', '@C_X_actual')],) )
                                    # adds the hover tool to the graph for the specifed line
   
    nitrate = p4.line('Time', 'C_N_actual', source = vs_source, line_width = 4 , line_color = "orange", legend_label = "Nitrate(Actual)", muted_color = 'orange',)
    p4.add_tools(HoverTool(renderers = [nitrate], tooltips=[  ('Name', 'Nitrate'), ('Time:', '@Time'), ('Concentration:', '@C_N_actual') ],) )
    
    vs_source.data['modified_C_L_actual'] = vs_source.data['C_L_actual'] * 1000# CL is multiplied by 1000 to make it visible on the graph and this is done wih the column data source

    lutien = p4.line('Time', 'modified_C_L_actual', source = vs_source, line_width = 4 , line_color = "lime", legend_label = "Lutein(Actual)(x1000)", muted_color = 'lime',)
    p4.add_tools(HoverTool(renderers = [lutien], tooltips=[  ('Name', 'Lutein'), ('Time:', '@Time'), ('Concentration:', '@C_L_actual') ],) )
    
    
    biomass_predicted = p4.line('Time', 'C_X', source = vs_source, line_dash = 'dashed', line_width = 4 ,  line_color = "aqua", legend_label = "Biomass(Predicted)", muted_color = 'aqua',)
    p4.add_tools(HoverTool(renderers = [biomass_predicted], tooltips=[  ('Name', 'Biomass'),  ('Time:', '@Time'), ('Concentration:', '@C_X')],) )
                                    # adds the hover tool to the graph for the specifed line
   
    nitrate_predicted = p4.line('Time', 'C_N', source = vs_source, line_dash = 'dashed', line_width = 4 , line_color = "orange", legend_label = "Nitrate(Predicted)", muted_color = 'orange',)
    p4.add_tools(HoverTool(renderers = [nitrate_predicted], tooltips=[  ('Name', 'Nitrate'), ('Time:', '@Time'), ('Concentration:', '@C_N') ],) )
       
    vs_source.data['modified_C_L'] = vs_source.data['C_L'] * 1000# CL is multiplied by 1000 to make it visible on the graph and this is done wih the column data source

    lutien_predicted = p4.line('Time', 'modified_C_L', source = vs_source, line_dash = 'dashed', line_width = 4 , line_color = "lime", legend_label = "Lutein(Predicted)(x1000)", muted_color = 'lime',)
    p4.add_tools(HoverTool(renderers = [lutien_predicted], tooltips=[  ('Name', 'Lutein'), ('Time:', '@Time'), ('Concentration:', '@C_L') ],) )
    
    # Add the lines to the plot
expPredsCSVPath = 'outputs/expPredVsDataset.csv'
if master:
    expPredsCSVPath = prefix + expPredsCSVPath    
versus_plot(expPredsCSVPath, p4)
p4.legend.click_policy="hide"
# display legend in top left corner (default is top right corner)
p4.legend.location = "top_left"

# change appearance of legend text
p4.legend.label_text_font = "times"
p4.legend.label_text_font_style = "italic"
p4.legend.label_text_color = "navy"

# change border and background of legend
p4.legend.border_line_width = 3
p4.legend.border_line_color = "black"
p4.legend.border_line_alpha = 0.8
p4.legend.background_fill_color = "white"
p4.legend.background_fill_alpha = 0.3



#Run Button******************************************************************************************************************************


run_button_edit_tab = Button(label = "Run", button_type = "primary", height = 60, width = 300)

def first_clicked(p2 = p2):
    p2.renderers = []
    # run_button_edit_tab.label = "Running..."
    # run_button_edit_tab.button_type = "danger"
    run_button_edit_tab_status_message.text = "Configuration Ran"
    run_button_edit_tab_status_message.styles = updated

def load_first_clicked():
    run_button_edit_tab_status_message.text = "Running..."
    run_button_edit_tab_status_message.styles = loading
    first_clicked(p2 = p2)
    
run_button_edit_tab.on_click(load_first_clicked)


#create dataframe to hold past runs, columns are LearnRate, lossFn, Optimizer, TrainSplit, Neurons, Epochs, BatchSize, LowestLoss, EpochofLowestLoss
pastRuns =  pd.DataFrame(columns=['LearnRate', 'lossFn', 'Optimizer', 'TrainSplit', 'Neurons', 'Epochs', 'BatchSize', 'LowestLoss', 'EpochofLowestLoss'])
charts = ColumnDataSource(pastRuns)

def edit_run_button_function(lR = learning_rate,  lFn = loss_Fn, opt = optimizer, tr = train, n = neurons, e = epochs, b = batch_Size, X = X, Y = Y, device = device, optimizer_options = optimizer_options, loss_options = loss_options, p2 = p2, p3 = p3, mean = mean_squared_error, root_mean = root_mean_squared_error, p4 = p4, minValLoss = lowest_mse_validation, minValLossIndex = epoch_of_lowest_loss, charts = charts): #ts = test, vs = val_split,
    #Idea: have two functions and the inputs can be( lossDF, testPreds, mse, rmse, XDF ) insted of (lR = learning_rate,  lFn = loss_Fn, opt = optimizer, tr = train, n = neurons, e = epochs, b = batch_Size, X = X, Y = Y, device = device, optimizer_options = optimizer_options, loss_options = loss_options, p2 = p2, p3 = p3, mean = mean_squared_error, root_mean = root_mean_squared_error, p4 = p4)
    #could also clear the graphs here before the actual graph functions
    #p2.renderers = []
    #run_button_edit_tab.disabled = True
    #function start timer
    functionStart = time.perf_counter()
    run_button_edit_tab.label = "Run"
    run_button_edit_tab.button_type = "primary"
    learning_rate = lR
    loss = lFn
    optimizer = opt
    train = tr
    #test = ts
    #val_split = vs
    neurons = n
    epochs = e
    batch_Size = b
    lossDF, testPreds, mse, rmse, XDF = model_loop(learning_rate, loss, optimizer, train,  neurons, epochs, batch_Size, X, Y, device, optimizer_options, loss_options) #test, val_split,
    #generating data from model loop
    loss_graph(lossDF, p2)
    #get min valLoss from lossDF, index is it's epoch num
    minValLoss.value = str(lossDF['valLoss'].min())
    #get the index of the min valLoss
    minValLossIndex.value = str(lossDF['valLoss'].idxmin())
    #TODO: show these things on 1st tab
    #parity graph
    parity_plot(testPreds, p3)
    mean.value = str(mse)
    root_mean.value = str(rmse)
    versus_plot(XDF, p4)
    #create array for this run
    thisRun = np.array([learning_rate.value, loss.value, optimizer.value, train.value, neurons.value, epochs.value, batch_Size.value, minValLoss.value, minValLossIndex.value])
    #append this run to pastRuns
    pastRuns.loc[len(pastRuns)] = thisRun
    print(pastRuns)
    #access the most recent row of pastRuns and stream it to charts
    charts.stream(pastRuns.iloc[-1:])
    #run_button_edit_tab.disabled = False
    #function end timer
    functionEnd = time.perf_counter()
    #total time for function
    functionTime = functionEnd - functionStart
    print(f"Total time for function: {functionTime:0.4f} seconds")
    run_button_edit_tab_status_message.text = "Configuration Ran"
    run_button_edit_tab_status_message.styles = updated
    #TODO: use XDF to plot the actual vs predicted values
    #'Time' on X axis, then two lines per output (actual and predicted) on Y axis
    #C_X, C_N, C_L are model outputs, C_X_actual, C_N_actual, C_L_actual are actual values


    
def load_edit_run_button_function():
    run_button_edit_tab_status_message.text = "Running..."
    run_button_edit_tab_status_message.styles = loading
    edit_run_button_function(lR = learning_rate,  lFn = loss_Fn, opt = optimizer, tr = train, n = neurons, e = epochs, b = batch_Size, X = X, Y = Y, device = device, optimizer_options = optimizer_options, loss_options = loss_options, p2 = p2, p3 = p3, mean = mean_squared_error, root_mean = root_mean_squared_error, p4 = p4, minValLoss = lowest_mse_validation, minValLossIndex = epoch_of_lowest_loss, charts = charts)

    
run_button_edit_tab.on_click(load_edit_run_button_function)

#Fonts Slection******************************************************************************************************************************
font_options = ["Arial", "San Serif", "Times New Roman", ]
fontAccess = Select(title="Font Options:", value="Arial", options= font_options, height = 60, width = 300)# Student chooses the loss function
def font_Callback(attr, old, new):
    if(new == "Arial"):
        intro.styles = {"font-family": "Arial"}
    elif(new == "San Serif"):
        intro.styles = {"font-family": "San Serif"}
    elif(new == "Times New Roman "):
        intro.styles = {"font-family": "Times New Roman"}
    
# fontAccess.on_change('value', font_Callback)


font_size_options = ["100%", "110%", "120%", "130%", "140%", "150%", "160%", "170%", "180%", "190%", "200%"]
sizeAccess = Select(title="Text Size Options:", value="100", options= font_size_options, height = 60, width = 300)# Student chooses the loss function

def size_Callback(attr, old, new, font):
    intro.styles = {'font-size': new, "font-family": font.value}

# sizeAccess.on_change('value', size_Callback, fontAccess)

# Define the callback function for font and size changes
def font_size_callback(attr, old, new):
    selected_font = fontAccess.value
    selected_size = sizeAccess.value
    
    intro.styles = {'font-size': selected_size, "font-family": selected_font}

# Attach the callback functions to the value change events of the font and size widgets
fontAccess.on_change('value', font_size_callback)
sizeAccess.on_change('value', font_size_callback)



#make pandas data frame for the chart

# Define the columns for the DataTable
columns = [TableColumn(field=column_name, title=column_name, width=80) for column_name in pastRuns.columns]

# Create the DataTable
chart_table = DataTable(source=charts, columns=columns, width=390, height=200, autosize_mode = "none")



#Putting the Model together______________________________________________________________________________________________________________________________

top_page_spacer = Spacer(height = 20)
left_page_spacer = Spacer(width = 20)
#Making Tabs and showing the Modles ---------------------------------------------------------------------------------------------------------------------
trains = row(train,train_help_button)
neuron = row(neurons, neurons_help_button)
epoch = row(epochs, epochs_help_button)
batch = row(batch_Size, batch_Size_help_button)
learning = row(learning_rate, learning_rate_help_button)
optimizers = row(optimizer, optimizer_help_button)
losses_help = row(loss_Fn, loss_Fn_help_button)
ls = column(trains, neuron, epoch, batch, learning, optimizers, losses_help,run_button_edit_tab, reset_button_edit_tab, run_button_edit_tab_status_message) #test,val_split,
rs = column(p2, )#Note that the p is just a place holder for the graph that will be shown,and the way i did the 2 p's didnt work
means = column(mean_squared_error, root_mean_squared_error,)
lowest_val_info = column(lowest_mse_validation, epoch_of_lowest_loss, chart_table)
bs = row(left_page_spacer, column(top_page_spacer, row(ls, rs, lowest_val_info)))
evaluate = row(left_page_spacer, column(top_page_spacer, row(p4,p3, means)))
choose = row (fontAccess, sizeAccess)
test = column(choose, intro)
tab1 = TabPanel(child=bs, title="Train")
tab3 = TabPanel(child= row(left_page_spacer, column(top_page_spacer, row(p,column(reset_button, slides, export_button, run_button, run_button_edit_tab_status_message)))), title="Optimize")
tab2 = TabPanel(child = evaluate, title = "Evaluate")
# tab4 = TabPanel(child = test, title = "Instruction")

#TODO: add evaluate tab, parity plot, synth line versus model line and mse/rmse displayed here


  
all_tabs = Tabs(tabs=[tab1,tab2,tab3])
# show(all_tabs)

#Making the layout to show all of the information and the code ---------------------------------------------------------------------------------------------------------------------

# l = layout(
#    [
#        
#        [all_tabs],
#    ] 
#    sizing_mode='scale_width' # ---- this might be a good idea to implement into the other modules, having the width scale
#)


# curdoc().add_root(l) #use "bokeh serve --show bokeh_Module.py" to run the code on a bokeh server
curdoc().add_root(all_tabs)


#code report: is able to plot, but run button is not working, also you also still need to do the validation to ensure that the test, train, and validation split add up to 1 or 100% check notes for more info
# look into imports and how to import the needed imports properly, also look into how to make the run button work, and that the losses.csv is actualy working