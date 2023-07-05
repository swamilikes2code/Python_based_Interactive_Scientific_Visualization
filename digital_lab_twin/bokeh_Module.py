
#needed for this to work: working directory should have a models folder with mmscalerX.pkl and mmscalerY.pkl and model.pt, and an outputs folder
# imports, put these at the very top of everything
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
#Intro Text sectionSection ---------------------------------------------------------------------------------------------------------------------
# NOTE THIS WAS ONLY MADE WITH PYTHON NOT HTML AT ALL

intro = Div(text="""
        <h3>Work In Progress!</h3>
        <p>A photobioreactor is a container, like a fish tank, filled with water and special microscopic plants called algae. 
        It provides the algae with light, nutrients, and carbon dioxide to help them grow. 
        The algae use sunlight to make their own food through a process called photosynthesis. 
        The photobioreactor allows us to grow algae and use their biomass to produce clean and renewable energy, like biofuels. It also helps clean up the environment by absorbing harmful pollutants, such as carbon dioxide. In simple terms, a photobioreactor is a special container that helps tiny plants called algae grow using light, nutrients, and carbon dioxide to make clean energy and help the planet.(section for the paragraph).</p>
        <h4> Section for bold text</h4>
    """)

#Info Text sectionSection ---------------------------------------------------------------------------------------------------------------------
# NOTE THIS WAS ONLY MADE WITH PYTHON NOT HTML AT ALL

info_text =  Paragraph(text="""
        This is a Bokeh demo showing the interactive examples that
       could be used for the photobioreactor example 
        """)

 #Help Text sectionSection ---------------------------------------------------------------------------------------------------------------------
# NOTE THIS WAS ONLY MADE WITH PYTHON NOT HTML AT ALL

help_text= Paragraph(text = """
            In photography, bokeh is the aesthetic quality of the blur produced in out-of-focus parts of an image,
              caused by Circles of Confusion. Bokeh has also been defined as "the way the lens renders out-of-focus
                points of light". Differences in lens aberrations and aperture shape cause very different bokeh effects.
    """
                     )

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
    p.renderers = [] #removes previous lines
    p.tools = [] #removes previous hover tools

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
    line_c = p.line('Time', 'C_L' * 1000, source = sources , line_width = 4, line_color = "lime", legend_label = "Lutine")# CL is multiplied by 1000 to make it visible on the graph
    p.add_tools(HoverTool( renderers = [line_c],tooltips=[('Name', 'Lutine'),
                                    ('Hour', '@Time'), 
                                    ('Concentration', '@C_L'), 
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
for u in updates:
    u.on_change('value', update_data)



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
    source.data = initial_source.data; // This is the initial data stored in the ColumnDataSource
    // Reset the axis ranges // this will help reset the axis ranges of the graph and the graph in genral
    p.x_range.start = Math.min.apply(null, source.data['Time']);
    p.x_range.end = Math.max.apply(null, source.data['Time']);
    p.y_range.start = Math.min.apply(null, source.data['C_X'].concat(source.data['C_N'], source.data['C_L']));
    p.y_range.end = Math.max.apply(null, source.data['C_X'].concat(source.data['C_N'], source.data['C_L']));

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

#Making Tabs and showing the Modles ---------------------------------------------------------------------------------------------------------------------
tab1 = TabPanel(child= row(  p,column(reset_button, slides, export_button, run_button) ), title="Model")
tab2 = TabPanel(child = column(intro, info_text, help_text), title = "Instruction")

  
all_tabs = Tabs(tabs=[tab1,tab2,])
# show(all_tabs)

#Making the layout to show all of the information and the code ---------------------------------------------------------------------------------------------------------------------

l = layout(
    [
        
        [all_tabs],
    ],
    sizing_mode='scale_width'
)


curdoc().add_root(l) #use "bokeh serve --show bokeh_Module.py" to run the code on a bokeh server



#code report: run works with reset, but after running the sliders dont function anymore
