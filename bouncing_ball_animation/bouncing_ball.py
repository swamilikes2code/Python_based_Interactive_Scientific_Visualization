import numpy as np
from bokeh.models import ColumnDataSource, Slider, Div, HoverTool, Grid, Tabs, Button
from bokeh.plotting import figure,show
from bokeh.io import curdoc
from bokeh.layouts import row, column, gridplot

# Acceleration due to gravity, m.s-2.
g = 9.81
# The maximum x-range of ball's trajectory to plot.
XMAX = 5
# The coefficient of restitution for bounces (-v_up/v_down).
cor_start = 0.65
# The time step for the animation.
dt = 0.005
# Initial position and velocity vectors.
x0, y0, t0 = 0.0, 4.0, 0.0
vx0, vy0 = 1.0, 0.0

initial_data = dict(x=[], y=[], t=[])
x, y, vx, vy, t = x0, y0, vx0, vy0, t0
start_time = 0.0
end_time = 8.0
time_step = 0.1

while t < end_time:
    x += vx0 * dt
    y += vy * dt
    vy -= g * dt
    t += dt
    if y < 0:
        # bounce!
        y = 0
        vy = -vy * cor_start
    initial_data['x'].append(x)
    initial_data['y'].append(y)
    t=np.around(t,3)
    #print(t)
    initial_data['t'].append(t)
source=ColumnDataSource(data=initial_data)
initial_pos_data = dict(x=[initial_data['x'][10]], y=[initial_data['y'][10]])
source_time = ColumnDataSource(data=initial_pos_data)

# Set up plot for concentrations
TOOLTIPS = [("x (m)","@x{0,0.000}"), ("y (m)","@y{0,0.000}")]
TOOLS = "pan,undo,redo,reset,save,wheel_zoom,box_zoom"
plot_ball = figure(height=450, width=550, tools=TOOLS, tooltips=TOOLTIPS,
              title="Bouncing Ball Animation")
plot_ball.line('x', 'y', source=source, line_width=3, line_alpha=0.6, line_color="mediumblue",
               legend_label="Trace")
plot_ball.circle('x', 'y', source=source_time, color="navy", size=15.0, alpha=0.75,
               legend_label="Current Location")
plot_ball.xaxis.axis_label = "x (m)"
plot_ball.yaxis.axis_label = "y (m)"
plot_ball.legend.location = "top_right"
plot_ball.legend.click_policy="hide"
plot_ball.legend.background_fill_alpha = 0.5
plot_ball.grid.grid_line_color = "silver"

slider_cor = Slider(title="Co-efficient of Restitution"+" (initial: "+str(cor_start)+")", value=cor_start, start=0.1, end=1.0, step=0.05)

slider_time = Slider(title="Time Slider (s)", value=start_time, start=start_time, end=end_time, step=time_step, width=500)

def animate_update():
    current_time = slider_time.value + time_step
    if current_time > end_time:
        current_time = start_time
    slider_time.value = current_time

# Function to update the plot data
def update_data(attrname, old, new):
    cor_value = slider_cor.value
    time_value = slider_time.value
    new_data = dict(x=[], y=[], t=[])
    x_data, y_data, vx_data, vy_data, t_data = x0, y0, vx0, vy0, t0
    while t_data < end_time:
        t_data += dt
        x_data += vx0 * dt
        y_data += vy_data * dt
        vy_data -= g * dt
        if y_data < 0:
            # bounce!
            y_data = 0
            vy_data = -vy_data * cor_value
        new_data['x'].append(x_data)
        new_data['y'].append(y_data)
        t_data=np.around(t_data, 3)
        new_data['t'].append(t_data)
    source.data=new_data
    index_time_value = new_data['t'].index(np.float(np.around(time_value, 3)))
    print(index_time_value)
    circle_dict = dict(x=[new_data['x'][index_time_value]], y=[new_data['y'][index_time_value]])
    source_time.data=circle_dict
        
for w in [slider_cor, slider_time]:
    w.on_change('value', update_data)
   
def animate():
    global callback_id
    if animate_button.label == '► Play':
        animate_button.label = '❚❚ Pause'
        callback_id = curdoc().add_periodic_callback(animate_update, time_step*1000.0) # s to milliseconds conversion
    else:
        animate_button.label = '► Play'
        curdoc().remove_periodic_callback(callback_id)

animate_button = Button(label='► Play', width=50)
animate_button.on_event('button_click', animate)

# Set up layouts and add to document
layout_bouncing_ball = row(column(slider_cor, slider_time, animate_button), plot_ball)
        
# Setup server
curdoc().add_root(layout_bouncing_ball)

curdoc().title = "Bouncing Ball"
