import pandas as pd
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, CustomJS, Slider
from bokeh.layouts import column
from bokeh.io import curdoc
import numpy as np

# Read data from CSV file
df = pd.read_csv("PredExperiment.csv")

# Create ColumnDataSource
source = ColumnDataSource(df)

# Create a figure
p = figure(title="Slider Example", x_axis_label="X", y_axis_label="Y", width=400, height=300)

# Plot the initial data
p.circle('Time', 'C_X', source=source, size=8)

# Create a slider
slider = Slider(start=0, end=10, value=5, step=0.1, title="Slider")

# Define the callback function
# callback = CustomJS(args=dict(source=source, slider=slider), code="""
#     const data = source.data;
#     const x = data['Time'];
#     const y = data['C_X'];
#     const value = slider.value;

#     // Update the Y values based on the slider value
#     for (let i = 0; i < y.length; i++) {
#         y[i] = Math.sin(x[i] * value);
#     }

#     // Emit the change event to update the plot
#     source.change.emit();
# """)

# # Attach the callback to the slider
# slider.js_on_change('value', callback)

def update_data(attr, old, new):
    a = slider.value

    x = source.data['Time']
    y1 = source.data['C_X']

    updated_y1 = y1 * a

    source.data = dict(Time=x, C_X=updated_y1)

# Add the callback to the sliders
slider.on_change('value', update_data)


# Create the layout
layout = column(p, slider)

# Show the plot
curdoc().add_root(layout)