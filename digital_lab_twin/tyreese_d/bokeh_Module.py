
import numpy as np
import os
import pandas
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

# UO = 





    




# curdoc().theme = "dark_minimal"# this makes the graph in dark mode
p = figure(title = "Change in  concentration over time in a photobioreactor", x_axis_label = "Time(hours)", y_axis_label = "concentration", )

#Data Generation Section ---------------------------------------------------------------------------------------------------------------------

data = "ActualExperiment.csv"
datas = pandas.read_csv(data)
source = ColumnDataSource(datas)
#Plotting Function Section ---------------------------------------------------------------------------------------------------------------------

def plot_graph(source):
   
    line_a = p.line('Time', 'C_X', source = source, line_width = 4 ,  line_color = "aqua", legend_label = "Biomass")
    p.add_tools(HoverTool(renderers = [line_a], tooltips=[  ('Name', 'Biomass'),
                                    ('Hour', '@Time'),
                                    ('Concentration', '@C_X'),# adds the hover tool to the graph for the specifed line
    ],))

    line_b = p.line('Time', 'C_N', source = source, line_width = 4 , line_color = "orange", legend_label = "Nitrate")
    p.add_tools(HoverTool( renderers = [line_b],tooltips=[('Name', 'Nitrate'),
                                    ('Hour', '@Time'), 
                                    ('Concentration', '@C_N'), 
    ],))
    line_c = p.line('Time', 'C_L', source = source , line_width = 4, line_color = "lime", legend_label = "Lutine")
    p.add_tools(HoverTool( renderers = [line_c],tooltips=[('Name', 'Lutine'),
                                    ('Hour', '@Time'), 
                                    ('Concentration', '@C_L'), 
    ],))

plot_graph(source)

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

# Add the Slider to the figure

light_intensity = Slider(start=100, end=200, value=150, step= 1, title="Light Intesity (umol/m2-s)")
inlet_flow = NumericInput(value=0.008, high = 0.015, low = 0.001, mode = "float", title="Inlet Flow(g/L):(0.001 - 0.015)")# Student chooses the learning rate

pH = Slider(start=0.1, end=9, value=0.5, step=.1, title="PH")
inlet_concentration = Slider(start=5, end=15, value=10, step=.1, title="Inlet Concentration(g/L)")
nitrate_con = Slider(start=0.2, end=2, value=1, step=.1, title="Nitrate Concentration(g/L)")
biomass_con = Slider(start=0.2, end=2, value=0.5, step=.1, title="Biomass Concentration(g/L)")

#Define the callback function for the sliders
def update_data(attr, old, new):
    # Get the current values of the sliders
    a = light_intensity.value
    b = inlet_flow.value
    c = pH.value
    d = inlet_concentration.value
    e = nitrate_con.value
    f = biomass_con.value
    
    # Retrieve the data from the source
    data = source.data
    x = data['Time']
    y1 = data['C_X']
    y2 = data['C_N']
    
    # Calculate the updated y-values using a loop
    updated_y1 = []
    updated_y2 = []
    for i in range(len(x)):
        updated_y1.append(b * a * (c + y1[i]) + d)
        updated_y2.append(b * a * (c + y2[i]) + d)

    
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
reset_button.js_on_click(CustomJS(args=dict( source = source , li = light_intensity, inf = inlet_flow, pH = pH, inc = inlet_concentration, nit = nitrate_con, bio = biomass_con),
                                  code="""
   li.value = 150
   inf.value = 0.008
   pH.value = 0.5
   inc.value = 10
   nit.value = 1
   bio.value = 0.5

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
# run_button.js_on_click(CustomJS(args=dict( source = source , li = light_intensity, inf = inlet_flow, pH = pH, inc = inlet_concentration),
#                                   code="""
#    li.value = 0.2
#    inf.value = 2
#    pH.value = 0.5
#    inc.value = 4

#     source.change.emit();


# """ ))


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


curdoc().add_root(l)




