from bokeh.plotting import show, figure;
from bokeh.models import ColumnDataSource, HoverTool, Select, CustomJS
import pandas
from bokeh.io import curdoc
from bokeh.layouts import row, column

color_options = ["Light Mode", "Dark Mode"]

colorAccess = Select(title="Color Options:", value="default", options= color_options, height = 60, width = 300)# Student chooses the loss function

def color_Callback(attr, old, new):
    colorAccess.value = new
    plot_graph(source, colorAccess)
    
colorAccess.on_change('value', color_Callback)

#Data Generation Section ---------------------------------------------------------------------------------------------------------------------

data = "outputs/prediction.csv"
datas = pandas.read_csv(data)
source = ColumnDataSource(datas)
#initial Data  for reset section ---------------------------------------------------------------------------------------------------------------------
initial_csv1 = "outputs/prediction.csv"
initial_data = pandas.read_csv(initial_csv1)
initial_source = ColumnDataSource(initial_data)

#Plotting Function Section ---------------------------------------------------------------------------------------------------------------------
p = figure(title = "Change in concentrations over time in a photobioreactor", x_axis_label = "Time(hours)", y_axis_label = "concentration", )

def plot_graph(sources, color_option):
    #Removes previous lines and hover tools
    
    
    if(color_option.value == "Dark Mode"):
        p.renderers = [] #removes previous lines
        p.tools = [] #removes previous hover tools
        curdoc().theme = "dark_minimal"# this makes the graph in dark mode

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
        line_c = p.line('Time', 'modified_C_L', source = sources , line_width = 4, line_color = "lime",  legend_label = "Lutein (x1000)")# CL is multiplied by 1000 to make it visible on the graph
        p.add_tools(HoverTool( renderers = [line_c],tooltips=[('Name', 'Lutien'),
                                        ('Hour', '@Time'), 
                                        ('Concentration', '@modified_C_L'), 
        ],)) 
   
    else:
        curdoc().theme = "caliber"# this makes the graph in dark mode
        p.renderers = [] #removes previous lines
        p.tools = [] #removes previous hover tools
        line_a = p.line('Time', 'C_X', source = sources, line_width = 4 ,  line_color = "navy", legend_label = "Biomass")
        p.add_tools(HoverTool(renderers = [line_a], tooltips=[  ('Name', 'Biomass'),
                                        ('Hour', '@Time'),
                                        ('Concentration', '@C_X'),# adds the hover tool to the graph for the specifed line
        ],))

        line_b = p.line('Time', 'C_N', source = sources, line_width = 4 , line_color = "violet", legend_label = "Nitrate")
        p.add_tools(HoverTool( renderers = [line_b],tooltips=[('Name', 'Nitrate'),
                                        ('Hour', '@Time'), 
                                        ('Concentration', '@C_N'), 
        ],))
        sources.data['modified_C_L'] = sources.data['C_L'] * 1000# CL is multiplied by 1000 to make it visible on the graph and this is done wih the column data source
        line_c = p.line('Time', 'modified_C_L', source = sources , line_width = 4, line_color = "green",  legend_label = "Lutein (x1000)")# CL is multiplied by 1000 to make it visible on the graph
        p.add_tools(HoverTool( renderers = [line_c],tooltips=[('Name', 'Lutien'),
                                        ('Hour', '@Time'), 
                                        ('Concentration', '@modified_C_L'), 
        ],)) 


    return p

p = plot_graph(source, colorAccess)

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

test = row(p, colorAccess)

curdoc().add_root(test)