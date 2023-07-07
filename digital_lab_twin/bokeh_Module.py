
#needed for this to work: working directory should have a models folder with mmscalerX.pkl and mmscalerY.pkl and model.pt, and an outputs folder
# imports, put these at the very top of everything
import matplotlib.pyplot as plt
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
from bokeh.layouts import layout
from bokeh.io import export_svgs
import datetime
from bokeh.models import ColumnDataSource, HoverTool, Slider, CustomJS, TabPanel, Tabs, Div, Paragraph, Button, Select, RadioButtonGroup, NumericInput
#Instrutions Tab Section_____________________________________________________________________________________________________________________
#Intro Text sectionSection ---------------------------------------------------------------------------------------------------------------------
#This is the intro text section, it is a div, which is a bokeh object that allows you to write html and css in python
#headers are made with <h1> </h1> tags, and paragraphs are made with <p> </p> tags, headers are automatically bolded


intro = Div(text="""
        <h3>Work In Progress!</h3>
        <h2>Header</h2><p>This is a <em>formatted</em> paragraph.</p>
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
        <p> This Button will take the slider conditions that you have and will create a new plot based on those new conditions</p>
        
         <h3>Export Button</h3>
        <p> This Button will take the data points of the Time, Nitrate Concentration, Biomass concentration, and Lutine concentration<br>
        and put them in a csv file and this csv file will be located in your downloads folder the file will be named "exported_data_{timestamp}.csv"<br>
        the timestamp is the current time and will be formated as year-month-day-hour-minuete-second</p>
        
        
        <h4> Section for bold text</h4>
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
light_intensity = Slider(start=100, end=200, value=150, step= 1, title="Light Intesity (umol/m2-s):(100 - 200)")
inlet_flow = Slider(start=0.001, end=0.015, value= 0.008, step=.0001, format = "0.000", title="Inlet Flow(g/L):(0.001 - 0.015)")
pH = Slider(start=0.1, end=9, value=0.5, step=.1, title="PH")
inlet_concentration = Slider(start=5, end=15, value=10, step=.1, title="Inlet Concentration(g/L):(5 - 15)")
nitrate_con = Slider(start=0.2, end=2, value=1, step=.1, title="Initial Nitrate Concentration(g/L):(0.2 - 2)")
biomass_con = Slider(start=0.2, end=2, value=0.5, step=.1, title="Initial Biomass Concentration(g/L):(0.2 - 2)")


#pytorch Preloop  section ---------------------------------------------------------------------------------------------------------------------



#function takes in initial conditions and runs the model
#overwrites XDF with the predicted values
#updates bokeh plot with new values
#call when run button is hit
def predLoop(C_X, C_N, C_L, F_in, C_N_in, I0):
    # initialize everything 
    # note that these load funcs will need you to change to your current directory here!
    print(os.getcwd() )
    # os.chdir('C:\\Users\\[computer_name]\\Documents\\GitHub\\Python_based_Interactive_Scientific_Visualization\\digital_lab_twin') #Windows version
    os.chdir('/Users/tyreesedavidson/Documents/GitHub/Python_based_Interactive_Scientific_Visualization/digital_lab_twin') #Mac version

    model = torch.load('models/model.pt')
    model.eval()
    #scalers
    stScalerX = joblib.load('models/stScalerX.pkl')
    stScalerY = joblib.load('models/stScalerY.pkl')

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
    XDF.to_csv('outputs/prediction.csv', index=False)
    #TODO: re-call the plotting function to show results to user

# predLoop(biomass_con.value, nitrate_con.value, 0, inlet_flow.value, inlet_concentration.value, light_intensity.value)
#testing with default values
#predLoop(C_X_init, C_N_init, C_L_init, F_in_init, C_N_in_init, I0_init)




#Data Generation Section ---------------------------------------------------------------------------------------------------------------------

data = "outputs/prediction.csv"
datas = pandas.read_csv(data)
source = ColumnDataSource(datas)

#initial Data section ---------------------------------------------------------------------------------------------------------------------
initial_csv = "outputs/initial_predictions.csv"
initial_datas = pandas.read_csv(initial_csv)
initial_sources= ColumnDataSource(initial_datas)
#initial Data  for reset section ---------------------------------------------------------------------------------------------------------------------
initial_csv1 = "outputs/prediction.csv"
initial_data = pandas.read_csv(initial_csv1)
initial_source = ColumnDataSource(initial_data)

#Plotting Function Section ---------------------------------------------------------------------------------------------------------------------
p = figure(title = "Change in  concentration over time in a photobioreactor", x_axis_label = "Time(hours)", y_axis_label = "concentration", )

def plot_graph(sources):
    #Removes previous lines and hover tools
    p.renderers = [] #removes previous lines
    p.tools = [] #removes previous hover tools
    
    
    # Example of updating CL value

    line_a = p.line('Time', 'C_X', source = sources, line_width = 4 ,  line_color = "aqua", legend_label = "Biomass")
    p.add_tools(HoverTool(renderers = [line_a], tooltips=[  ('Name', 'Biomass'),
                                    ('Hour', '@Time'),
                                    ('Concentration', '@C_X'),# adds the hover tool to the graph for the specifed line
    ],))

    line_b = p.line('Time', 'C_N', source = sources, line_width = 4 , line_color = "orange", legend_label = "Nitrate")
    p.add_tools(HoverTool( renderers = [line_b],tooltips=[('Name', 'Nitrate'),
                                    ('Hour', '@Time'), 
                                    ('Concentration', '@C_N'), 
    ],))
    sources.data['modified_C_L'] = sources.data['C_L'] * 1000# CL is multiplied by 1000 to make it visible on the graph and this is done wih the column data source
    line_c = p.line('Time', 'modified_C_L', source = sources , line_width = 4, line_color = "lime", legend_label = "Lutien x 1000")# CL is multiplied by 1000 to make it visible on the graph
    p.add_tools(HoverTool( renderers = [line_c],tooltips=[('Name', 'Lutien'),
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


#Define the callback function for the sliders
def update_data(attr, old, new):
    # Get the current values of the sliders
    light_i = light_intensity.value * 0.000001 #converts from micro to grams
    F_in = inlet_flow.value
    ph = pH.value
    Cn_in = inlet_concentration.value
    cn = nitrate_con.value
    cx = biomass_con.value
    
    # Retrieve the data from the source
    data = source.data
    x = data['Time']
    y1 = data['C_X']
    y2 = data['C_N']
    
    # Calculate the updated y-values using a loop
    updated_y1 = []
    updated_y2 = []
    for i in range(len(x)):
        updated_y1.append(U_O * (cn / (cn + K_N)) * cx - U_D * cx  )#Biomass
        updated_y2.append(-Y_NX - U_O * (cn / (cn + K_N))* cx + F_in * Cn_in )#still need to fix the equations  to make it accurat to the number in the ode

    
    # Update the data source
    source.data = {'Time': x, 'C_X': updated_y1, 'C_N': updated_y2}

# Add the callback function to the sliders
updates=[light_intensity, inlet_flow, pH, inlet_concentration, nitrate_con, biomass_con]
# for u in updates:
#     u.on_change('value', update_data)



# light_intensity.on_change('value', update_data)
# inlet_flow.on_change('value', update_data)
# pH.on_change('value', update_data)
# inlet_concentration.on_change('value', update_data)

slides = column(light_intensity, inlet_flow, pH, inlet_concentration, nitrate_con, biomass_con)

# callback = CustomJS(args=dict( source = source , li = light_intensity, inf = inlet_flow, pH = pH, inc = inlet_concentration),
#                     code="""

#     const a = li.value;
#     const b = inf.value;
#     const c = pH.value;
#     const d = inc.value;

#     const data = source.data;
#     const x = data['Time'];
#     const y1 = data['C_X'];
#     const y2 = data['C_N'];

#     const updated_y1 = [];
#     const updated_y2 = [];

#     for (let i = 0; i < x.length; i++) {
#         updated_y1.push(b + a * (c * y1[i] + d));
#         updated_y2.push(b + a * (c * y2[i] + d));
#     }


#     source.data = { 'Time': x, 'C_X': updated_y1, 'C_N': updated_y2 };
#     source.change.emit();
# """)


# light_intensity.js_on_change('value', callback)
# inlet_flow.js_on_change('value', callback)
# pH.js_on_change('value', callback)
# inlet_concentration.js_on_change('value', callback)

#dkllhdfhdlk
#kdjdklfjfdlkfkl
# Creating the Button---------------------------------------------------------------------------------------------------------------------

#Reset Button******************************************************************************************************************************
reset_button = Button(label = "Reset", button_type = "danger", height = 60, width = 300)
reset_button.js_on_click(CustomJS(args=dict( source = source,initial_source = initial_source, p = p, li = light_intensity, inf = inlet_flow, pH = pH, inc = inlet_concentration, nit = nitrate_con, bio = biomass_con),
                                  code="""
   li.value = 150
   inf.value = 0.008
   pH.value = 0.5
   inc.value = 10
   nit.value = 1
   bio.value = 0.5
    // Reset the plot data
    //source.data = initial_source.data; // This is the initial data stored in the ColumnDataSource
    // Reset the axis ranges // this will help reset the axis ranges of the graph and the graph in genral
    // p.x_range.start = Math.min.apply(null, source.data['Time']);
    // p.x_range.end = Math.max.apply(null, source.data['Time']);
    // p.y_range.start = Math.min.apply(null, source.data['C_X'].concat(source.data['C_N'], source.data['C_L']));
    // p.y_range.end = Math.max.apply(null, source.data['C_X'].concat(source.data['C_N'], source.data['C_L']));

    source.change.emit();


""" ))

  # Clear the renderers (lines) from the plot
# p.renderers = []
# for u in updates:
#     u.on_change('value', update_data)


# initial_data = pandas.read_csv("ActualExperiment.csv")
# af = ColumnDataSource(initial_data)

# source = af



 
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
    export_data = {'Time': day, 'Biomass': biomass, 'Nitrate': nitrate, 'Lutine': lutine}

     # Generate a unique filename using current date and time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"exported_data_{timestamp}.csv"

        # Get the path to the user's downloads folder
    downloads_folder = os.path.expanduser("~/Downloads")

    # Create the file path
    file_path = os.path.join(downloads_folder, filename)

    # Export the data as a CSV file to the downloads folder
    with open(file_path, 'w') as f:
        f.write('Time,Biomass,Nitrate,Lutine\n')
        for i in range(len(day)):
            f.write(f'{day[i]},{biomass[i]},{nitrate[i]},{lutine[i]}\n')





export_button = Button(label="Export Data", button_type="success",  height = 60, width = 300)
export_button.on_click(export_data)

#Run Button******************************************************************************************************************************
run_button = Button(label = "Run", button_type = "primary", height = 60, width = 300)

def runbutton_function(li = light_intensity, inf = inlet_flow, pH = pH, inc = inlet_concentration, nit = nitrate_con, bio = biomass_con, ): 
    
    #set initial conditions by changing these vals
    C_X_init = bio.value
    C_N_init = nit.value
    C_L_init = 0.0
    F_in_init = inf.value
    C_N_in_init = inc.value
    I0_init = li.value
    # p = figure() PREDLOOP IDEA EVERYTIME THE FUNCTION IS CALLED IT ALSO PASSES IN A NEW FIGURE THAT IS CREATED


    # print((C_X_init, C_N_init, C_L_init, F_in_init, C_N_in_init, I0_init))
    predLoop(C_X_init, C_N_init, C_L_init, F_in_init, C_N_in_init, I0_init)
    #creates the source for the graph that the new plot will be based on
    data = "outputs/prediction.csv"
    datas = pandas.read_csv(data)
    sourceS = ColumnDataSource(datas)
    #attempt to reset the graph IDEA: ADD     p.renderers = [] AT THE BEGINNING OF THE PLOTTING FUNCTION
    plot_graph(sourceS) ######this is the new plot that will be shown YOU NEED TO FIX THIS SO THAT THE FIGURE IS UPDATED

run_button.on_click(runbutton_function)


#Edit Tab Section______________________________________________________________________________________________________________________________
#Model Inputs Section-----------------------------------------------------------------------------------------------
TYPES = ["NN", "GP", "Forest"]
optimizer_options = ["ADAM", "SGD"]
loss_options = ["MSE", "MAE"]

radio_button_group = RadioButtonGroup(name = "Types", labels=TYPES, active=0)# Student chooses the ML model type

test = NumericInput(value=0.2, high = 100, low = 0, mode = "float", title="Test:(0% - 100%)")# 

train = NumericInput(value=0.6, high = 100, low = 0, mode = "float", title="Train:(0% - 100%)")# 

val_split = NumericInput(value=0.2, high = 100, low = 0, mode = "float", title="Val Split:(0% - 100%)")# 

neurons = Slider (start = 7, end = 100, value = 18, step = 1, title = "Number of Neurons")# 
epochs = Slider (start = 0, end = 200, value = 100, step = 25, title = "Epochs")# 
batch_Size = Slider (start = 0, end = 200, value = 25, step = 25, title = "Batch Size")# 


learning_rate = NumericInput(value=0.001, high = 0.01, low = 0.0001, mode = "float", title="Learning Rate:(0.0001-0.01)")# Student chooses the learning rate

loss_Fn = Select(title="Optimizer:", value="MAE", options= loss_options, height = 60, width = 300)# Student chooses the loss function

optimizer = Select(title="Loss Fn:", value="ADAM", options= optimizer_options, height = 60, width = 300)# Student chooses the optimizer 


#Rest Buttton For Edit Tab Section -----------------------------------------------------------------------------------------------
reset_button_edit_tab = Button(label = "Reset", button_type = "danger", height = 60, width = 300)
reset_button_edit_tab.js_on_click(CustomJS(args=dict( lR = learning_rate,  lFn = loss_Fn, opt = optimizer, tr = train, ts = test, vs = val_split, n = neurons, e = epochs, b = batch_Size),
                                  code="""
   lR.value = 0.0001;
   lFn.value = "ADAM";
   opt.value = "LI";
   n.value = 32;
   vs.value = 0;
   tr.value = 0;
   ts.value = 0;
   e.value = 100;
   b.value = 25;
    
    



""" ))
#Model Loop section for edit tab_____________________________________________________________________________________________________________________

#the below code is designed to drag and drop into the bokeh visualization
#static section should run once on launch, dynamic section should run on each change
### Static (run once)
rawData = pd.read_csv('STEMVisualsSynthData.csv', header=0)
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


def model_loop(lR = learning_rate,  lFn = loss_Fn, opt = optimizer, tr = train, ts = test, vs = val_split, n = neurons, e = epochs, b = batch_Size, X = X, Y = Y, device = device, optimizer_options = optimizer_options, loss_options = loss_options):
    
  #user defined parameters: current values can serve as a default
  #splits - expects 3 floats that add to 1
  trainSplit = tr.value
  valSplit = vs.value
  testSplit = ts.value
  #model params
  initNeuronNum = n.value #number of neurons in the first layer, 7 < int < 100
  loss = loss_options.index(lFn.value) #0 = MSE, 1 = MAE
  optimizer = optimizer_options.index(opt.value ) #0 = Adam, 1 = SGD
  learnRate = lR.value #0.0001 < float < 0.01
  #training params
  epochs = e.value #0 < int < 200
  batchSize = b.value #0 < int < 200
    
  ### Dynamic (run on each change)
  #TODO: upon running, check params are valid then update these values
  #test the all-in-one function
  model, Y_test_tensor, testPreds, XTestTime = mnn.trainAndSaveModel(X, Y, trainSplit, valSplit, testSplit, initNeuronNum, loss, optimizer, learnRate, epochs, batchSize, device)
  #read in the loss CSV
  lossCSV = pd.read_csv('models/losses.csv', header=0)
  #TODO:plot the losses against epochs (stored as indexes)
  #TODO:update the prediction side of the bokeh visualization
    
# #Loss Graph Data section ---------------------------------------------------------------------------------------------------------------------
# loss_data = "models/losses.csv"
# loss_datas = pandas.read_csv(loss_data)
# loss_source = ColumnDataSource(loss_datas)
#Loss Graph section ---------------------------------------------------------------------------------------------------------------------
p2 = figure(title = "Change in  concentration over time in a photobioreactor", x_axis_label = "Time(hours)", y_axis_label = "concentration", )

def loss_graph(loss_data, p2): # function to plot the loss graph
    #Removes previous lines and hover tools
    
    loss_datas = pandas.read_csv(loss_data)
    loss_datas['index'] = loss_datas.index
    loss_source = ColumnDataSource(loss_datas)
    # Example of updating CL value

    train_loss = p2.line('index', 'trainLoss', source = loss_source, line_width = 4 ,  line_color = "aqua", legend_label = "Train Loss")
    p2.add_tools(HoverTool(renderers = [train_loss], tooltips=[  ('Name', 'Train Loss'),
                                    # adds the hover tool to the graph for the specifed line
    ],))
    value_loss = p2.line('index', 'valLoss', source = loss_source, line_width = 4 ,  line_color = "lime", legend_label = "Value Loss")
    p2.add_tools(HoverTool(renderers = [value_loss], tooltips=[  ('Name', 'Value Loss'), ],))

    
    # Add the lines to the plot
    
    
loss_graph("models/losses.csv", p2)


    

#Run Button******************************************************************************************************************************
run_button_edit_tab = Button(label = "Run", button_type = "primary", height = 60, width = 300)

def edit_run_button_function(lR = learning_rate,  lFn = loss_Fn, opt = optimizer, tr = train, ts = test, vs = val_split, n = neurons, e = epochs, b = batch_Size, X = X, Y = Y, device = device, optimizer_options = optimizer_options, loss_options = loss_options, p2 = p2): 
    
    learning_rate = lR.value
    loss = lFn.value
    optimizer = opt.value
    train = tr.value
    test = ts.value
    val_split = vs.value
    neurons = n.value
    epochs = e.value
    batch_Size = b.value
    model_loop(learning_rate, loss, optimizer, train, test, val_split, neurons, epochs, batch_Size, X, Y, device, optimizer_options, loss_options)
    #generating data from model loop
    loss_graph("models/losses.csv", p2)
    
    
    
    
run_button_edit_tab.on_click(edit_run_button_function)

    





#Putting the Model together______________________________________________________________________________________________________________________________
#Making Tabs and showing the Modles ---------------------------------------------------------------------------------------------------------------------
ls = column( radio_button_group, test, train, val_split, neurons, epochs, batch_Size, learning_rate, optimizer, loss_Fn, )
rs = column(p2, run_button_edit_tab, reset_button_edit_tab)#Note that the p is just a place holder for the graph that will be shown,and the way i did the 2 p's didnt work
bs = row(ls, rs)
tab1 = TabPanel(child=bs, title="Edit")
tab2 = TabPanel(child= row(  p,column(reset_button, slides, export_button, run_button) ), title="Predictions")
tab3 = TabPanel(child = column(intro), title = "Instruction")

  
all_tabs = Tabs(tabs=[tab1,tab2,tab3])
# show(all_tabs)

#Making the layout to show all of the information and the code ---------------------------------------------------------------------------------------------------------------------

l = layout(
    [
        
        [all_tabs],
    ],
    sizing_mode='scale_width'
)


curdoc().add_root(l) #use "bokeh serve --show bokeh_Module.py" to run the code on a bokeh server



#code report: 
