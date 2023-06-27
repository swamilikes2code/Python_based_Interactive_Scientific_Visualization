
from bokeh.io import show
from bokeh.models import CustomJS, Select, NumericInput, RadioButtonGroup, Button, ColumnDataSource
from bokeh.layouts import column, row
from bokeh.models.widgets import NumericInput
import pandas



# Prediction Data Section ---------------------------------------------------------------------------------------------------------------------
df = pandas.read_csv("PredExperiment.csv")
source = ColumnDataSource(df)

#Actual Data Section ---------------------------------------------------------------------------------------------------------------------
af = pandas.read_csv("ActualExperiment.csv")
source1 = ColumnDataSource(af)
#Model Inputs Section-----------------------------------------------------------------------------------------------
TYPES = ["NN", "GP", "Forest"]

radio_button_group = RadioButtonGroup(name = "Types", labels=TYPES, active=0)# Student chooses the ML model type

learning_rate = NumericInput(value=0.00001, high = 0.1, low = 0.00001, mode = "float", title="Learning Rate:(0.00001-0.1)")# Student chooses the learning rate

optimizer = Select(title="Optimizer:", value="LI", options=["LI", "MSE", "KL Div"])# Student chooses the optimizer

loss_Fn = Select(title="Loss Fn:", value="ADAM", options=["ADAM", "SGD"])# Student chooses the loss function


#Rest Buttton Section-----------------------------------------------------------------------------------------------
reset_button = Button(label = "Reset", button_type = "danger", height = 60, width = 300)
reset_button.js_on_click(CustomJS(args=dict( lR = learning_rate,  lFn = loss_Fn, opt = optimizer),
                                  code="""
   lR.value = 0.00001;
   lFn.value = "NN";
    opt.value = "NN";



""" ))




show(column(radio_button_group,learning_rate, optimizer, loss_Fn, reset_button))