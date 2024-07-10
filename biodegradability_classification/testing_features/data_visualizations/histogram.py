import pandas as pd
from bokeh.plotting import figure, curdoc
from bokeh.transform import dodge
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool, Select
import numpy as np

# Load your data
file_path = '../../data/option_1.csv'
df = pd.read_csv(file_path)

df['Class'] = df['Class'].astype(str)  # Convert to string if it's not already

# Split data based on the class
class_0 = df[df['Class'] == '0']
class_1 = df[df['Class'] == '1']

# Default histogram column
default_hist_column = 'MaxEStateIndex'

# Define the number of bins
num_bins = 20
bins = np.linspace(df[default_hist_column].min(), df[default_hist_column].max(), num_bins)

# Calculate histogram for each class
hist_0, edges_0 = np.histogram(class_0[default_hist_column], bins=bins)
hist_1, edges_1 = np.histogram(class_1[default_hist_column], bins=bins)

# Prepare data for plotting

# Calculate the center positions of each bin
centers = (edges_0[:-1] + edges_0[1:]) / 2

# Create a new ColumnDataSource
source = ColumnDataSource(data=dict(
    centers=centers,
    top_class_0=hist_0,
    top_class_1=hist_1
))

# Create the figure
p = figure(title=f"Histogram of {default_hist_column} with Class Color Coding",
           x_axis_label=default_hist_column, y_axis_label='Frequency',
           tools="pan,wheel_zoom,box_zoom,reset,hover,save",
           width=800, height=400)

# Add class 0 bars
bars_class_0 = p.vbar(x=dodge('centers', -0.15, range=p.x_range), top='top_class_0', width=0.3*(centers[1] - centers[0]),
                      color='blue', alpha=0.6, legend_label='Class 0', source=source)

# Add class 1 bars
bars_class_1 = p.vbar(x=dodge('centers', 0.15, range=p.x_range), top='top_class_1', width=0.3*(centers[1] - centers[0]),
                      color='red', alpha=0.6, legend_label='Class 1', source=source)

# Add hover tool for interaction
hover = HoverTool()
hover.tooltips = [("Range", "@centers"),
                  ("Class 0 Frequency", "@top_class_0"),
                  ("Class 1 Frequency", "@top_class_1")]
p.add_tools(hover)

# Style the plot
p.legend.click_policy = "hide"
p.legend.location = "top_right"
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = "gray"
p.ygrid.grid_line_dash = [6, 4]

# Create a Select widget for choosing histogram column
options = df.columns[4:16].tolist() # CHANGE THIS LINE FOR THE OPTIONS. 
select_x_axis = Select(title="X Axis:", value=default_hist_column, options=options)

# Callback function for Select widget
def update_plot(attrname, old, new):
    selected_column = select_x_axis.value

    # Update histogram data based on selected column
    hist_0, edges_0 = np.histogram(class_0[selected_column], bins=bins)
    hist_1, edges_1 = np.histogram(class_1[selected_column], bins=bins)

    centers = (edges_0[:-1] + edges_0[1:]) / 2

    source.data = dict(
        centers=centers,
        top_class_0=hist_0,
        top_class_1=hist_1
    )
    bars_class_0.data_source.data['top'] = hist_0
    bars_class_1.data_source.data['top'] = hist_1

    p.xaxis.axis_label = selected_column

# Attach callback to Select widget
select_x_axis.on_change('value', update_plot)

# Layout the widgets and plot
layout = column(select_x_axis, p)

# Add layout to curdoc
curdoc().add_root(layout)
