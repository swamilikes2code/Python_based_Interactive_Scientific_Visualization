import pandas as pd
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Slider
from bokeh.layouts import column
from bokeh.io import curdoc

# Read data from CSV file
df = pd.read_csv("PredExperiment.csv")

# Create a ColumnDataSource from the data
source = ColumnDataSource(df)

# Create a figure
p = figure(title="Line Graph", x_axis_label="X", y_axis_label="Y", width=800, height=400)

# Plot a line based on the data
line = p.line('Time', 'C_X', source=source, line_width=2)

# Create a slider
slider = Slider(start=0, end=10, value=5, step=0.1, title="Slider")

# Define the callback function for the slider
def update_data(attrname, old, new):
    # Get the current slider value
    slider_value = slider.value

    # Update the data source with the new values
    source.data['C_X'] = source.data['Time'] * slider_value
    # p.y_range.start = min(source.data['C_X'])
    # p.y_range.end = max(source.data['C_X'])

# Add the callback to the slider
slider.on_change('value', update_data)

# Create a layout for the plot and slider
layout = column(p, slider)

# Add the layout to the current document
curdoc().add_root(layout)

# Show the plot
# show(layout)