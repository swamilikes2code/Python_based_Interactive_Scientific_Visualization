
from bokeh.io import show
from bokeh.models import CustomJS, Select, Slider, RadioButtonGroup
from bokeh.layouts import column, row

select = Select(title="Type:", value="NN", options=["NN", "GP", "Forest"])


slider = Slider(start=0.00001, end=10, value=.1, step=.00001, title="Learning Rate")

LABELS = ["LI", "MSE", "KL Div"]

radio_button_group = RadioButtonGroup(name = "Loss Fn", labels=LABELS, active=0)



show(column(select, slider, radio_button_group))