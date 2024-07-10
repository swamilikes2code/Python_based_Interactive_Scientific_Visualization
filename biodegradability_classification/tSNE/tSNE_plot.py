import pandas as pd
from sklearn.manifold import TSNE
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import linear_cmap
from bokeh.palettes import Spectral6

df = pd.read_csv('./option_4.csv') # change this line with your file path

features = df.iloc[:, 3:].values 
labels = df['Class'].tolist()

tsne = TSNE(n_components=2, random_state=42) 
tsne_result = tsne.fit_transform(features)

output_file('option_4_PathFP_tSNE.html') # change this line to what you want the name of your file to be

source = ColumnDataSource(data=dict(
    x=tsne_result[:, 0],
    y=tsne_result[:, 1],
    label=labels
))

# Create a color mapper
color_mapper = linear_cmap(field_name='label', palette=Spectral6, low=min(labels), high=max(labels))

p = figure(title='Path FP tSNE plot', tools='pan, wheel_zoom, reset', tooltips=[('Label', '@label')]) #change the name of the plot here

p.scatter('x', 'y', size=5, source=source, legend_field='label', fill_color=color_mapper, line_color=None, alpha=0.6) #plot

p.legend.title = 'Class' #legend
p.legend.location = 'top_left'

hover = HoverTool() 
hover.tooltips = [('Label', '@label')]
p.add_tools(hover)

show(p)
