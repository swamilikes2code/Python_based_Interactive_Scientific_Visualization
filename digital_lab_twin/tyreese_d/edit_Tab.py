
from bokeh.io import show
from bokeh.models import CustomJS, Select, NumericInput, RadioButtonGroup
from bokeh.layouts import column, row
from bokeh.models.widgets import NumericInput

#Model Inputs Section-----------------------------------------------------------------------------------------------
TYPES = ["NN", "GP", "Forest"]

radio_button_group = RadioButtonGroup(name = "Types", labels=TYPES, active=0)

learning_rate = NumericInput(value=0.00001, high = 0.1, low = 0.00001, mode = "float", title="Learning Rate:(0.00001-0.1)")



optimizer = Select(title="Optimizer:", value="NN", options=["LI", "MSE", "KL Div"])
loss_Fn = Select(title="Loss Fn:", value="NN", options=["ADAM", "SGD"])




show(column(radio_button_group,learning_rate, optimizer, loss_Fn))