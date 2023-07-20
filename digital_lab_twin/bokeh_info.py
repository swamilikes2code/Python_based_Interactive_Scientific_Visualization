from bokeh.models import Slider, Div
from bokeh.layouts import column
from bokeh.plotting import curdoc

# Create the slider widget
slider = Slider(start=0, end=10, value=5, step=1, title="Slider")

# Create a Div widget for the tooltip
tooltip = Div(text="This is a slider that allows you to choose a value between 0 and 10.")

# Callback function to update the tooltip text based on the slider value
def update_tooltip(attr, old, new):
    tooltip.text = f"You chose the value {new} on the slider."

# Attach the callback function to the slider's value change event
slider.on_change('value', update_tooltip)

# Create a layout to display the slider and the tooltip
layout = column(slider, tooltip)

# Add the layout to the current document
curdoc().add_root(layout)