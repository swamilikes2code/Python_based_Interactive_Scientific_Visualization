import pandas as pd
from sklearn.manifold import TSNE
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import linear_cmap
from bokeh.palettes import Spectral6

# Load data from CSV
df = pd.read_csv('./option_4.csv')

# Extract features and labels
features = df.iloc[:, 3:].values
labels = df['Class'].tolist()

# Perform tSNE dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(features)

# Create Bokeh plot
output_file('option_4_PathFP_tSNE.html')

# Convert data for Bokeh plotting
source = ColumnDataSource(data=dict(
    x=tsne_result[:, 0],
    y=tsne_result[:, 1],
    label=labels
))

# Create a color mapper
color_mapper = linear_cmap(field_name='label', palette=Spectral6, low=min(labels), high=max(labels))

# Create the figure
p = figure(title='Path FP tSNE plot', tools='pan, wheel_zoom, reset', tooltips=[('Label', '@label')])

# Plot data points
p.scatter('x', 'y', size=5, source=source, legend_field='label', fill_color=color_mapper, line_color=None, alpha=0.6)

# Add legend
p.legend.title = 'Class'
p.legend.location = 'top_left'

# Add hover tool
hover = HoverTool()
hover.tooltips = [('Label', '@label')]
p.add_tools(hover)

# Show the plot
show(p)
