import bokeh #Base Bokeh import
import bokeh.plotting #Bokeh plotting tools
import numpy as np #Numpy for data manipulation
#extra bokeh imports to access specific functions
from bokeh.plotting import figure, show, curdoc
from bokeh.layouts import row, column
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import Slider, Div, CustomJS, ColumnDataSource

#create our sine wave--x is the range of values, y is the sine of x
x = np.linspace(0, 10, 500)
y = np.sin(x)

# Create a ColumnDataSource object to store our data
source = ColumnDataSource(data=dict(x=x, y=y))

# Create the figure and plot the line
plot = figure(y_range=(-10, 10), width=400, height=400)
plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

# Create the slider widgets
amp = Slider(start=0.1, end=10, value=1, step=.1, title="Amplitude")
freq = Slider(start=0.1, end=10, value=1, step=.1, title="Frequency")
phase = Slider(start=-6.4, end=6.4, value=0, step=.1, title="Phase")
offset = Slider(start=-9, end=9, value=0, step=.1, title="Offset")

#callback uses JS code to update the data in the ColumnDataSource object
#we get the current values of each slider
#x remains the same, so we cal call the source.data.x for it
#we then do the math to find our new line using the slider values
#and update the source.data with the new x and y vals
callback = CustomJS(args=dict(source=source, amp=amp, freq=freq, phase=phase, offset=offset),
                    code="""
    const A = amp.value
    const k = freq.value
    const phi = phase.value
    const B = offset.value

    const x = source.data.x
    const y = Array.from(x, (x) => B + A*Math.sin(k*x+phi))
    source.data = { x, y }
""")

# Attach the callback to the slider widgets' value change events
amp.js_on_change('value', callback)
freq.js_on_change('value', callback)
phase.js_on_change('value', callback)
offset.js_on_change('value', callback)

# Create a layout for the slider widgets
inputs = column(amp, freq, phase, offset)

# Create a layout for the plot and the widgets
layout = row(plot, inputs)

# Add the layout to the current document (this is the only difference from the notebook!)
curdoc().add_root(layout)