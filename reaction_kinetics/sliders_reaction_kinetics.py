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
from bokeh.layouts import row, column, gridplot, Spacer
from bokeh.models import ColumnDataSource, ColorBar, LinearColorMapper, Slider, Div, HoverTool, TabPanel, Tabs, Grid, LinearAxis, Button
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
specie_names = ['A', 'B', 'C']
vbar_top = [vec_conc_t0[0], vec_conc_t0[1], vec_conc_t0[2]]
specie_colors = ['darkgray', 'mediumblue', 'darkorange']

# Solve ODE
vec_conc_t = odeint(dconc_dt, vec_conc_t0, vec_time, args=(params,))
int_vec_A = vec_conc_t[:,0]
int_vec_B = vec_conc_t[:,1]
int_vec_C = vec_conc_t[:,2]
source = ColumnDataSource(data=dict(vec_time=vec_time, int_vec_A=int_vec_A, int_vec_B=int_vec_B, int_vec_C=int_vec_C))
source_vbar = ColumnDataSource(data=dict(specie_names=specie_names, vbar_top=vbar_top, color=specie_colors))

# Set up plot for concentrations
TOOLTIPS = [("Time (s)","@vec_time"), ("A","@int_vec_A{0,0.000}"), ("B","@int_vec_B{0,0.000}"), ("C","@int_vec_C{0,0.000}")]
TOOLS = "undo,redo,reset,save,box_zoom"
# original height: 450, width: 550
plot_conc = figure(height=400, width=450, tools=TOOLS, tooltips=TOOLTIPS,
              title="Sequential reactions involving A, B and C", x_range=[t_start, t_end], y_range=[-0.05, 1.05])
plot_conc.line('vec_time', 'int_vec_A', source=source, line_width=3, line_alpha=0.6, line_color="darkgray",
               legend_label="A Concentration")
plot_conc.line('vec_time', 'int_vec_B', source=source, line_width=3, line_alpha=0.6, line_color="navy",
               legend_label="B Concentration")
plot_conc.line('vec_time', 'int_vec_C', source=source, line_width=3, line_alpha=0.6, line_color="darkorange",
               legend_label="C Concentration")
plot_conc.xaxis.axis_label = "Time (s)"
plot_conc.yaxis.axis_label = "Concentration"
plot_conc.legend.location = "top_left"
plot_conc.legend.click_policy="hide"
plot_conc.legend.background_fill_alpha = 0.5
plot_conc.grid.grid_line_color = "silver"

# Set up vertical bar plot for concentrations at a certain time
TOOLTIPS_vbar = [("Specie_Name","@specie_names"), ("Concentration","@vbar_top{0,0.000}")]
plot_vbar = figure(height=400, width=450, tools=TOOLS, tooltips=TOOLTIPS_vbar, x_range=specie_names,
                   y_range=[-0.05, 1.05], title="Concentration A, B and C at time specified by slider")
plot_vbar.vbar(x='specie_names', top='vbar_top', source=source_vbar, bottom=0.0, width=0.5, alpha=0.6, color="color",
               legend_field="specie_names")
plot_vbar.xaxis.axis_label = "Species"
plot_vbar.yaxis.axis_label = "Concentration"
plot_vbar.xgrid.grid_line_color = None
plot_vbar.legend.orientation = "horizontal"
plot_vbar.legend.location = "top_center"

# Set up widgets
text = Div(text="""For a sequential reaction <b>A to B to C</b>, set Values of <b>k_AB</b>, <b>k_BC</b>, <b>order_AB</b>, and <b>order_BC</b>.
Goal is to maximize concentration of B at a certain time. Concentration of A(t=0s) = 1.0.""", width=300)
slider_k_AB = Slider(title="k_AB"+" (initial: "+str(k_AB_start)+")", value=k_AB_start, start=2.02, end=8.0, step=0.02)
slider_k_BC = Slider(title="k_BC"+" (initial: "+str(k_BC_start)+")", value=k_BC_start, start=0.02, end=2.0, step=0.02)
slider_order_AB = Slider(title="order_AB"+" (initial: "+str(order_AB_start)+")", value=order_AB_start, start=1, end=5, step=1)
slider_order_BC = Slider(title="order_BC"+" (initial: "+str(k_BC_start)+")", value=order_BC_start, start=1, end=5, step=1)
start_time = 0.0
end_time = 8.0
time_step = 0.1
slider_time = Slider(title="Time Slider (s)", value=start_time, start=start_time, end=end_time, step=time_step)

def animate_update():
    current_time = slider_time.value + time_step
    if current_time > end_time:
        current_time = start_time
    slider_time.value = current_time

def update_data(attrname, old, new):

    # Get the current slider values
    k_AB_temp = slider_k_AB.value
    k_BC_temp = slider_k_BC.value
    O_AB_temp = slider_order_AB.value
    O_BC_temp = slider_order_BC.value
    time_temp = slider_time.value

    # Generate the new curve
    vec_time = np.linspace(t_start, t_end, N)  # vector for time
    params_temp = [O_AB_temp, O_BC_temp, k_AB_temp, k_BC_temp]
    vec_conc_t = odeint(dconc_dt, vec_conc_t0, vec_time, args=(params_temp,))
    int_vec_A = vec_conc_t[:,0]
    int_vec_B = vec_conc_t[:,1]
    int_vec_C = vec_conc_t[:,2]
    source.data =  dict(vec_time=vec_time, int_vec_A=int_vec_A, int_vec_B=int_vec_B, int_vec_C=int_vec_C)
    vbar_top_temp = [np.interp(time_temp, vec_time, int_vec_A), np.interp(time_temp, vec_time, int_vec_B),
                     np.interp(time_temp, vec_time, int_vec_C)]
    specie_names = ['A', 'B', 'C']
    specie_colors = ['darkgray', 'mediumblue', 'darkorange']
    source_vbar.data = dict(specie_names=specie_names, vbar_top=vbar_top_temp, color=specie_colors)

for w in [slider_k_AB, slider_k_BC, slider_order_AB, slider_order_BC, slider_time]:
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


# Spacers
top_page_spacer = Spacer(height = 20)
left_page_spacer = Spacer(width = 20)
large_left_page_spacer = Spacer(width = 45)

# Set up layouts and add to document
# inputs_reaction = row(left_page_spacer, column(top_page_spacer, text, slider_k_AB, slider_k_BC, slider_order_AB, slider_order_BC))
# inputs_time = column(animate_button, slider_time)

inputs_reaction = row(left_page_spacer, column(text, slider_k_AB, slider_k_BC, slider_order_AB, slider_order_BC))
inputs_time = row(large_left_page_spacer, column(animate_button, slider_time))

tab1 =TabPanel(child=column(top_page_spacer, row(inputs_reaction, large_left_page_spacer, plot_conc, left_page_spacer, column(plot_vbar, inputs_time))), title="Reaction Kinetics")

tabs = Tabs(tabs = [tab1])

curdoc().add_root(tabs)

curdoc().title = "Sequential_Reactions"
