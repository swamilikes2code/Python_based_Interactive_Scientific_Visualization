''' Present an interactive function explorer with slider widgets.
Sequential reactions example where a user can manipulate parameters.
Sliders are used to change reaction rate constants k_AB and k_BC.
Order of reactions is also controlled by sliders order_AB and order_BC.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve --show sliders_reaction_kinetics.py
at your command prompt. If default port is taken, you can specify
port using ' --port 5010' at the end of the bokeh command.
'''
import numpy as np
from scipy.integrate import odeint

from bokeh.io import curdoc
from bokeh.layouts import row, column, gridplot
from bokeh.models import ColumnDataSource, ColorBar, LinearColorMapper, Slider, TextInput
from bokeh.plotting import figure
from bokeh.palettes import Blues8

#
def dconc_dt(conc, t, params):
    """
    Defines the differential equations for a reaction kinetics system.

    Arguments:
        conc :  vector of the state variables:
                  conc = [vec_A,vec_B,vec_C]
        t :  time
        params :  vector of the parameters:
                  params = [O_AB,O_BC,k_AB,k_BC]
    """
    vec_A, vec_B, vec_C = conc
    O_AB, O_BC, k_AB, k_BC = params

    # create df_dt vector
    df_dt = [-k_AB * np.power(vec_A,O_AB), k_AB * np.power(vec_A,O_AB) - k_BC * np.power(vec_B,O_BC),
             k_BC * np.power(vec_B,O_BC)]

    return df_dt

# Set up data
t_start = 0.0
t_end = 8.0
N = 200 # number of data points
vec_time = np.linspace(t_start, t_end, N) # vector for time

# Starting values of all parameters
order_AB_start = 1
order_BC_start = 1
k_AB_start = 3.0
k_BC_start = 1.0
params = [order_AB_start, order_BC_start, k_AB_start, k_BC_start]

# Starting concentration of A, B, C
vec_conc_t0 = np.zeros(3)
vec_conc_t0[0] = 1.0

# Solve ODE
vec_conc_t = odeint(dconc_dt, vec_conc_t0, vec_time, args=(params,))
int_vec_A = vec_conc_t[:,0]
int_vec_B = vec_conc_t[:,1]
int_vec_C = vec_conc_t[:,2]
source = ColumnDataSource(data=dict(vec_time=vec_time, int_vec_A=int_vec_A, int_vec_B=int_vec_B, int_vec_C=int_vec_C))

# Set up plot
TOOLS = "crosshair,pan,undo,redo,reset,save,wheel_zoom,box_zoom"
plot = figure(plot_height=900, plot_width=1200, tooltips = [("A","$int_vec_A"), ("Time","$vec_time")],
              title="Example: Sequential reactions with A --> B --> C, starting with [A]_0 = 1.0",
              tools=TOOLS, x_range=[t_start, t_end], y_range=[-0.05, 1.05])

plot.cross('vec_time', 'int_vec_A', source=source, size=8, alpha=0.6, color="navy", legend_label="A Concentration")
plot.line('vec_time', 'int_vec_B', source=source, line_width=3, line_alpha=0.6, line_color="navy", legend_label="B Concentration")
plot.circle('vec_time', 'int_vec_C', source=source, size=5, alpha=0.6, line_color="navy", legend_label="C Concentration")
plot.xaxis.axis_label = "Time (s)"
plot.yaxis.axis_label = "Concentration"
plot.legend.location = "top_right"
plot.legend.click_policy="hide"
plot.legend.background_fill_alpha = 0.5
plot.background_fill_color = "#efefef"
plot.grid.grid_line_color = "darkslategray"

# Set up widgets
text = TextInput(title="Exercise", value='For A -> B -> C, set Values of k_AB, k_BC, order_AB, and order_BC')
slider_k_AB = Slider(title="k_AB"+" (initial: "+str(k_AB_start)+")", value=k_AB_start, start=2.02, end=8.0, step=0.02)
slider_k_BC = Slider(title="k_BC"+" (initial: "+str(k_BC_start)+")", value=k_BC_start, start=0.02, end=2.0, step=0.02)
slider_order_AB = Slider(title="order_AB"+" (initial: "+str(order_AB_start)+")", value=order_AB_start, start=1, end=5, step=1)
slider_order_BC = Slider(title="order_BC"+" (initial: "+str(k_BC_start)+")", value=order_BC_start, start=1, end=5, step=1)

# Set up callbacks
def update_title(attrname, old, new):
    plot.title.text = text.value

text.on_change('value', update_title)

def update_data(attrname, old, new):

    # Get the current slider values
    k_AB_temp = slider_k_AB.value
    k_BC_temp = slider_k_BC.value
    O_AB_temp = slider_order_AB.value
    O_BC_temp = slider_order_BC.value

    # Generate the new curve
    vec_time = np.linspace(t_start, t_end, N)  # vector for time
    params_temp = [O_AB_temp, O_BC_temp, k_AB_temp, k_BC_temp]
    vec_conc_t = odeint(dconc_dt, vec_conc_t0, vec_time, args=(params_temp,))
    int_vec_A = vec_conc_t[:,0]
    int_vec_B = vec_conc_t[:,1]
    int_vec_C = vec_conc_t[:,2]
    source.data =  dict(vec_time=vec_time, int_vec_A=int_vec_A, int_vec_B=int_vec_B, int_vec_C=int_vec_C)

for w in [slider_k_AB, slider_k_BC, slider_order_AB, slider_order_BC]:
    w.on_change('value', update_data)

# Set up layouts and add to document
inputs = column(text, slider_k_AB, slider_k_BC, slider_order_AB, slider_order_BC)

curdoc().add_root(row(inputs, plot, width=1200))
curdoc().title = "Sliders_Sequential_Reactions"
