from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.io import curdoc
from bokeh.models import Text
from bokeh.models import Circle
from bokeh.layouts import column
import numpy as np

# Acceleration due to gravity, m.s-2.
g = 9.81
# The maximum x-range of ball's trajectory to plot.
XMAX = 5
# The coefficient of restitution for bounces (-v_up/v_down).
cor = 0.65
# The time step for the animation.
dt = 0.005

# Initial position and velocity vectors.
x0, y0 = 0, 4
vx0, vy0 = 1, 0

# Create a Bokeh figure
bouncing_ball_figure = figure(x_range=(0, XMAX), y_range=(0, y0), 
                              width=400, height=300, 
                              title="Ball Animation")
bouncing_ball_figure.xaxis.axis_label = 'x /m'
bouncing_ball_figure.yaxis.axis_label = 'y /m'

# Initialize data source
source = ColumnDataSource(data=dict(x=[], y=[]))
bouncing_ball_figure.line(x="x", y="y", line_width=2, source=source)

# Add the ball as a circle glyph
ball = Circle(x=x0, y=y0, size=15, fill_color="red", line_color="black", line_width=2)
bouncing_ball_figure.add_glyph(ball)

# Add the height text
height_text = Text(x=XMAX * 0.5, y=y0 * 0.8, text=f'Height: {y0:.1f} m', text_font_size="12pt")
bouncing_ball_figure.add_layout(height_text)

# Function to update the plot data
def update():
    new_data = dict(x=[], y=[])
    x, y, vx, vy = x0, y0, vx0, vy0
    while x < XMAX:
        x += vx0 * dt
        y += vy * dt
        vy -= g * dt
        if y < 0:
            # bounce!
            y = 0
            vy = -vy * cor
        new_data['x'].append(x)
        new_data['y'].append(y)
        yield new_data

# Create a periodic callback to update the plot
def update_animation():
    new_data = next(update_gen)
    source.stream(new_data, rollover=100)

update_gen = update()

curdoc().add_periodic_callback(update_animation, 50)

# Display the plot
layout = column(bouncing_ball_figure)
show(layout)
