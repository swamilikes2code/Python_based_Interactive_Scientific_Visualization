import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.transform import dodge, factor_cmap
from bokeh.models import ColumnDataSource, HoverTool
import numpy as np

# Use output_file for standalone script
output_file("histogram_class_split.html")

# Load your data
file_path = '../rdkit_test.csv'
df = pd.read_csv(file_path)

df['Class'] = df['Class'].astype(str)  # Convert to string if it's not already

# Split data based on the class
class_0 = df[df['Class'] == '0']
class_1 = df[df['Class'] == '1']

# Check if the column exists
hist_column = 'MaxEStateIndex'
if hist_column not in df.columns:
    print(f"'{hist_column}' column not found in the data.")
    exit(1)

# Define the number of bins
num_bins = 20
bins = np.linspace(df[hist_column].min(), df[hist_column].max(), num_bins)

# Calculate histogram for each class
hist_0, edges_0 = np.histogram(class_0[hist_column], bins=bins)
hist_1, edges_1 = np.histogram(class_1[hist_column], bins=bins)

# Prepare data for plotting
# Combine heights of both classes into a single array
top_combined = np.array([hist_0, hist_1]).T

# Calculate the center positions of each bin
centers = (edges_0[:-1] + edges_0[1:]) / 2

# Create a new ColumnDataSource
source = ColumnDataSource(data=dict(
    centers=centers,
    top_class_0=hist_0,
    top_class_1=hist_1
))

# Create the figure
p = figure(title=f"Histogram of {hist_column} with Class Color Coding",
           x_axis_label=hist_column, y_axis_label='Frequency',
           tools="pan,wheel_zoom,box_zoom,reset,hover,save",
           width=800, height=400)

# Add class 0 bars
p.vbar(x=dodge('centers', -0.15, range=p.x_range), top='top_class_0', width=0.3*(centers[1] - centers[0]),
       color='blue', alpha=0.6, legend_label='Class 0', source=source)

# Add class 1 bars
p.vbar(x=dodge('centers', 0.15, range=p.x_range), top='top_class_1', width=0.3*(centers[1] - centers[0]),
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

# Show the plot
show(p)
