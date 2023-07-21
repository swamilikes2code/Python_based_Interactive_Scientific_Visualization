
from bokeh.plotting import show, figure
from bokeh.models import CustomJS, Select, NumericInput, RadioButtonGroup, Button, ColumnDataSource, HoverTool, Paragraph, Div
from bokeh.layouts import column, row, layout
from bokeh.models.widgets import NumericInput
import pandas
import numpy as np


intro = Div(text="""
        <h3>Work In Progress!</h3>
    
        <h4> Section for bold text</h4>
    """)


# Prediction Data Section ---------------------------------------------------------------------------------------------------------------------
df = pandas.read_csv("PredExperiment.csv")
source = ColumnDataSource(df)

#Actual Data Section ---------------------------------------------------------------------------------------------------------------------
af = pandas.read_csv("ActualExperiment.csv")
source1 = ColumnDataSource(af)

name = ["Biomass", "Nitrate", "Lutine"]
p = figure(title = "Change in  concentration over time in a photobioreactor", x_axis_label = "Time(hours)", y_axis_label = "concentration", )

#Actual Lines ******************************************************************************************************************************
line_a = p.line('Time', 'C_X', source = af, line_width = 4 ,  line_color = "aqua", legend_label = "Biomass")
p.add_tools(HoverTool(renderers = [line_a], tooltips=[  ('Name', 'Biomass'),
                                  ('Hour', '@Time'),
                                  ('Concentration', '@C_X'),# adds the hover tool to the graph for the specifed line
],))

line_1 = p.line('Time', 'C_X', source = df, line_dash = 'dashdot', line_width = 4, line_color = "blue", legend_label = "Biomass Prediction")
p.add_tools(HoverTool(renderers = [line_1], tooltips=[  ('Name', 'Biomass Prediction'),
                                  ('Hour', '@Time'),
                                  ('Concentration', '@C_X'),# adds the hover tool to the graph for the specifed line
],))
#Model Inputs Section-----------------------------------------------------------------------------------------------
TYPES = ["NN", "GP", "Forest"]

radio_button_group = RadioButtonGroup(name = "Types", labels=TYPES, active=0)# Student chooses the ML model type

learning_rate = NumericInput(value=0.00001, high = 0.1, low = 0.00001, mode = "float", title="Learning Rate:(0.00001-0.1)")# Student chooses the learning rate

optimizer = Select(title="Optimizer:", value="LI", options=["LI", "MSE", "KL Div"], height = 60, width = 300)# Student chooses the optimizer

loss_Fn = Select(title="Loss Fn:", value="ADAM", options=["ADAM", "SGD"], height = 60, width = 300)# Student chooses the loss function


#Rest Buttton Section-----------------------------------------------------------------------------------------------
reset_button = Button(label = "Reset", button_type = "danger", height = 60, width = 300)
reset_button.js_on_click(CustomJS(args=dict( lR = learning_rate,  lFn = loss_Fn, opt = optimizer),
                                  code="""
   lR.value = 0.00001;
   lFn.value = "NN";
    opt.value = "NN";



""" ))
ls = column( radio_button_group,learning_rate, optimizer, loss_Fn)
rs = column(p, reset_button)
bs = row(ls, rs)
l = layout(
    [
        [intro],
        [bs]
    ],
    sizing_mode='scale_width'
)
show(l)

