import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Select, ColumnDataSource
from bokeh.plotting import figure
from bokeh.palettes import Category10

# Load your dataset
df = pd.read_csv('temp_data.csv')

# Assuming your dataset has a "class" column and several other numeric columns
columns = df.columns.tolist()
columns.remove("Class")

# Convert the class column to a categorical column if it's not already
df['Class'] = df['Class'].astype('category')

# Create a ColumnDataSource
source = ColumnDataSource(data=dict(x=[], y=[], class_color=[]))

# Create a figure
p = figure(title="Interactive Scatter Plot", x_axis_label='X', y_axis_label='Y', 
           tools="pan,wheel_zoom,box_zoom,reset,hover,save")

# Create an initial scatter plot
scatter = p.scatter(x='x', y='y', color='class_color', source=source, legend_field='class_color')

# Create dropdown menus for X and Y axis
select_x = Select(title="X Axis", value=columns[0], options=columns)
select_y = Select(title="Y Axis", value=columns[1], options=columns)

# Update the data based on the selections
def update_data(attrname, old, new):
    x = select_x.value
    y = select_y.value
    
    source.data = {
        'x': df[x],
        'y': df[y],
        'class_color': [Category10[3][0] if cls == df['Class'].cat.categories[0] else Category10[3][1] for cls in df['Class']]
    }
    p.xaxis.axis_label = x
    p.yaxis.axis_label = y

# Attach the update_data function to the dropdowns
select_x.on_change('value', update_data)
select_y.on_change('value', update_data)

# Initial data setup
update_data(None, None, None)

# Layout setup
layout = column(row(select_x, select_y), p)

# Add the layout to the current document
curdoc().add_root(layout)

# To run this script, save it to a file, for example, 'app.py' and run `bokeh serve --show app.py`
