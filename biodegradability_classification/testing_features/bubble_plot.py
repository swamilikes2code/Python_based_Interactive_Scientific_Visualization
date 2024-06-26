import pandas as pd
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker, PrintfTickFormatter, HoverTool
from bokeh.palettes import Viridis256
from bokeh.transform import transform

scale = 10
d = {'T_range': ['Positive', 'Positive',
                 'Negative', 'Negative'],
     'Subject': ['Negative', 'Positive',
                 'Negative', 'Positive'],
     'count': [603, 240,
               220, 118],
     'count_scaled': [603 / scale, 240 / scale,
           220 / scale, 118 / scale]}

df = pd.DataFrame(data = d)
source = ColumnDataSource(df)
p = figure(x_range = df['T_range'].unique(), y_range = df['Subject'].unique())

color_mapper = LinearColorMapper(palette = Viridis256, low = df['count'].min(), high = df['count'].max())
color_bar = ColorBar(color_mapper = color_mapper,
                     location = (0, 0),
                     ticker = BasicTicker())
p.add_layout(color_bar, 'right')
p.scatter(x = 'T_range', y = 'Subject', size = 'count_scaled', fill_color = transform('count', color_mapper), source = source)
p.add_tools(HoverTool(tooltips = [('Count', '@count')]))
curdoc().add_root(p)