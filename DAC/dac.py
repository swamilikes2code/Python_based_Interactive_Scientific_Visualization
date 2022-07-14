''' Present an interactive function explorer with slider widgets.
There are 5 input parameters that the user can play with it
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve --show dac.py
at your command prompt. If default port is taken, you can specify
port using ' --port 5010' at the end of the bokeh command.
'''
import math
from scipy.integrate import solve_ivp, odeint
from bokeh.io import save, curdoc
from bokeh.layouts import column, row
from bokeh.model import Model
from bokeh.models import CustomJS, Slider, Callback, HoverTool, Button
from bokeh.plotting import ColumnDataSource, figure, show
from bokeh.models.widgets import Panel, Tabs
import numpy as np

# --------------------- Static Parameters    --------------------- #

b0 = 93 * (10**-5)  # 93      unit : 1/bars
deltH_0 = 95300  #               unit: j/mol 
Tw= T_in = 298.0  #  room temperature  unit: kelvin
T0 = 353.15  # temeperature in   unit: kelvin
t_h0 = .37  # heterogeneity constant 
apha = 0.33
chi = 0.0
q_s0 = 3.40 # qs_var = q_s0 = 3.4 due to chi = 0 mol/kg
# R = 8.314* (10**3) # Universal gas constant - LPa/molK
Rg = .0821 # Universal gas constant in l-atm/mol/K
kT0 = 3.5 * (10 ** -2)  # used to compute kT for r_CO2... in mol/Kg-pa-sec
EaCO2 = 15200 # activation for KT -- J/mol
ps = 880.0 #
deltH_co2 = 75000.0  # calculate temeprature change   unit: jol/mol

R_constant = 8.314 # jol/kelvin-mol

# ------------------ For Equation : Enegergy Ballance  -------------- #
pg = 1.87  # 
h = 13.8
Cp_g = 846.0  # J/kgK
Cp_s = 1500.0  # J/kgK

# ------------------ ODE Repetitive Shortcut -------------- #

def ener_balan(v0, theta, deltZ):  # replace v0  * pg* Cp_g / (theta * deltZ)
    ener_balan_part1 = v0 * pg * Cp_g
    # print(f"v0 * pg * Cp_g: ", {ener_balan_part1})
    return (ener_balan_part1 / (theta * deltZ))

def ener_balan2(episl_r):
    ener_balan2 = (1 - episl_r) * ps * deltH_co2
    # print(f"(1 - episl_r) * ps * deltH_co2: ", {ener_balan2})
    return (ener_balan2)

def ener_balan3(a_s, Tn):
    ener_balan3 = a_s * h * (Tw - Tn)
    # print(f"a_s * h * (Tw - Tn): " , {ener_balan3})
    return (ener_balan3)

# Equation 1 Mass Balance : find co2_n

def mass_balan(v0, episl_r, deltZ):
    mass_balan = v0 / (episl_r * deltZ)
    # print(f"v0 / (episl_r * deltZ): ", {mass_balan})
    return (mass_balan)

def masss_balan2(episl_r, ps):
    mass_balan2 = (1 - episl_r) * ps
    # print(f"(1 - episl_r) * ps: ", {mass_balan2})
    return (mass_balan2)

def cube(x):
    if 0<=x: return x**(1./3.)
    return -(-x)**(1./3.)

# ------------------ Equastions to calculate Rco2 -------------- #
def b(T): # will be called inside deriv
    # print(f"T", {T})
    b = b0 *(math.exp(((deltH_0 / (R_constant * T0)) * (T0 / T - 1))))
    # print(f'b, {b}')
    return b

def t_h(T): # will be call inside  deriv
    # print(T)
    # print(f'T, {T}')
    t_h = t_h0 + (apha * (1 - (T0 / T)) )
    # t_h = .37 + ( .33 * (1 - (353.15 / T)) )
    # print(t_h)
    return (t_h)

# Calculate rco2_n (not ode)
def R_co2(T, c_co2, q, b_var, t_var):

    kn = kT0 * ( math.exp(( (-EaCO2) / (R_constant*T) )))
    # print(f"T in rco2", {T})
    # print(f"kn",{kn})
    rco2_1 = Rg * T * c_co2 
    # if t_var <0:
    #     print("alert, t_var is negative")
    if q<0:
        print(f"q in rco2", {q})
        print("alert, q is nagative")
    rco2_2 = ((1 - ((q / q_s0) ** (t_var))) ** (1 / t_var))
    # rco2_2 = ((1 - ((q / q_s0) ** (t_var))) ** (1 / t_var))
    # print(f"rco2_2", {rco2_2})abs
    rco2_3 = q / (b_var * q_s0)
    # print(f"rco2_3", {rco2_3})
    # r_co2_part1 = rco2_1    
    rco2 = kn * (rco2_1 * rco2_2 - rco2_3) * 100
    # print(f"rco2", {rco2})
    # r_co2_part1 =(Rg * T * c_co2 * ((1 - ((q / q_s0) ** (t_var))) ** (1 / t_var)) - q / (b_var * q_s0))
    # print(f"rco2_part1",{r_co2_part1})
    # r_co2 = kn *r_co2_part1
    # print(f'r_co2, {r_co2}')
    # print(f"b_var * qs_var, {rco2_term5}.")
    return rco2

"""
    Defines the differential equations for odes of DAC
    Arguments:
        y :  vector of the state variables:
                  y = all 15 states
        t :  time
        params :  vector of the parameters:
                  params = [V, r, T, c_co2_0, episl_r, v0]
    
"""

def deriv1(t, y, params):
    T_n, co2_n, q_n, T_n2, co2_n2, q_n2,T_n3, co2_n3, q_n3, T_n4, co2_n4, q_n4,T_n5, co2_n5, q_n5 = y # the rest of 12 vars a_n are not used, only for the success of solve_ivp
    V, T, c_co2_0, episl_r, volumetric_flow = params

    ###############   -----  Parameters depend on input  -----  ###############
    r = cube(V/(20*math.pi))
    v0 = volumetric_flow / (math.pi *r*r )
    L = V / (math.pi * (r ** 2))
    deltZ = L / 5.0  # 5 boxes in total
    a_s = 2 / r
    theta = (1 - episl_r) * ps * Cp_s + episl_r * pg * Cp_g
    t_var = t_h(T)

    # print(f"t_var in deriv", {t_var}) compare with the one in rco2
    b_var = b(T)
   # T_n, co2_n, q_n, T_n2, co2_n2, q_n2, T_n3, co2_n3, q_n3, T_n4, co2_n4, q_n4, T_n5, co2_n5, q_n5 == y
    # rco2_ first, rate of generation
    T1 = -ener_balan(v0, theta, deltZ) * T_n + ener_balan(v0, theta, deltZ) * T_in + ener_balan2(episl_r) * (
        R_co2(T_n, co2_n, q_n,  b_var, t_var))/theta + ener_balan3(a_s, T_n)/theta
    # print(f"T1", {T1})
    co2_1dot = -mass_balan(v0, episl_r, deltZ) * co2_n + mass_balan(v0, episl_r, deltZ) * c_co2_0 - (
        R_co2(T_n, co2_n, q_n,  b_var, t_var)) * masss_balan2(episl_r, ps)/episl_r
    q1dot = R_co2(T_n, co2_n, q_n,  b_var, t_var)
    # print(f"energy balance in T1", {ener_balan(v0, theta, deltZ)})
    
    T2 = -ener_balan(v0, theta, deltZ) * T_n2 + ener_balan(v0, theta, deltZ) * T1 + ener_balan2(episl_r) * (
        R_co2(T_n2, co2_n2, q_n2,  b_var, t_var))/theta + ener_balan3(a_s,  T_n2)/theta
    # print(f"T2", {T2})
    co2_2dot = -mass_balan(v0, episl_r, deltZ) * co2_n2 + mass_balan(v0, episl_r, deltZ) * co2_1dot - (
        R_co2(T_n2, co2_n2, q_n2,  b_var, t_var)) * masss_balan2(episl_r, ps)/episl_r
    q2dot = R_co2(T_n2, co2_n2, q_n2,  b_var, t_var)
    # print(f"energy balance in T1", {ener_balan(v0, theta, deltZ)})

    T3 = -ener_balan(v0, theta, deltZ) * T_n3 + ener_balan(v0, theta, deltZ) * T2 + ener_balan2(episl_r) * (
        R_co2(T_n3, co2_n3, q_n3,  b_var, t_var))/theta + ener_balan3(a_s, T_n3)/theta
    co2_3dot = -mass_balan(v0, episl_r, deltZ) * co2_n3 + mass_balan(v0, episl_r, deltZ) * co2_2dot - (
        R_co2(T_n3, co2_n3, q_n3,  b_var, t_var)) * masss_balan2(episl_r, ps)/episl_r
    q3dot = R_co2(T_n3, co2_n3, q_n3,  b_var, t_var)

    T4 = -ener_balan(v0, theta, deltZ) * T_n4 + ener_balan(v0, theta, deltZ) * T3 + ener_balan2(episl_r) * (
        R_co2(T_n4, co2_n4, q_n4,  b_var, t_var))/theta + ener_balan3(a_s,  T_n4)/theta
    co2_4dot = -mass_balan(v0, episl_r, deltZ) * co2_n4 + mass_balan(v0, episl_r, deltZ) * co2_3dot - (
        R_co2(T_n4, co2_n4, q_n4,  b_var, t_var)) * masss_balan2(episl_r, ps)/episl_r
    q4dot = R_co2(T_n4, co2_n4, q_n4,  b_var, t_var)

    T5 = -ener_balan(v0, theta, deltZ) * T_n5 + ener_balan(v0, theta, deltZ) * T4 + ener_balan2(episl_r) * (
        R_co2(T_n5, co2_n5, q_n5,  b_var, t_var))/theta + ener_balan3(a_s,  T_n5)/theta
    co2_5dot = -mass_balan(v0, episl_r, deltZ) * co2_n5 + mass_balan(v0, episl_r, deltZ) * co2_4dot - (
        R_co2(T_n5, co2_n5, q_n5,  b_var, t_var)) * masss_balan2(episl_r, ps)/episl_r
    q5dot = R_co2(T_n5, co2_n5, q_n5,  b_var, t_var)

    # result = np.array([T1, T2, T3, T4, T5, co2_1dot, co2_2dot, co2_3dot, co2_4dot, co2_5dot, q1dot, q2dot, q3dot, q4dot, q5dot]).reshape(-1, 1)

    return [T1, co2_1dot, q1dot, T2, co2_2dot, q2dot, T3, co2_3dot, q3dot, T4, co2_4dot, q4dot, T5, co2_5dot, q5dot]

# ------------------ User generated - Slider initial value -------------- #
V = 200.0  # volume
T = 293.0 # +273 ambrient temperature
c_co2_0 = .016349 # mol/m^3    
episl_r = 0.3  # void
volumetric_flow = 5 # m^3/s

# air humidity 
# no radius and length, nonly nr *reed V, july 6th

# ------------------ Initial Conditions to set up solve_ivp -------------- #
t0, tf = 0.0, 10800.0 # 3hrs
co2_initial = 0.016
q_init_cond = 0
init_cond = [T, co2_initial, q_init_cond, T, co2_initial,q_init_cond, T, co2_initial, q_init_cond, T, co2_initial, q_init_cond, T,co2_initial, q_init_cond]
# ,20.000, 0.000, 0.000,20.000, 0.000, 0.000,20.000, 0.000, 0.000,20.000, 0.000, 0.000
params = [V, T, c_co2_0, episl_r, volumetric_flow]
tspan = np.linspace(t0, tf, 5)
soln = solve_ivp(deriv1, (t0, tf), init_cond, args=(params,), t_eval = tspan, method = "BDF")  # init_cond = (T, c_co2_0, q0)
# soln = solve_ivp(deriv1, (t0, tf), init_cond, args=(params,), method = "BDF", rtol = 1e-5, atol = 1e-8)  # init_cond = (T, c_co2_0, q0)
# deriv1([t0, tf], )
print(soln)
# Equation 3
# dq/dt = r_co2

#  Graph co2_n, T_n, and rco2_n
# temperature should be flat, c02 sould drop, q should increase, 
# need to know when it reaches to L, then stop the whole operation
# time when it reaches the breakthrough and desorption 

T1, T2, T3, T4, T5 = soln.y[0], soln.y[3], soln.y[6], soln.y[9], soln.y[12]
co2_1dot, co2_2dot, co2_3dot, co2_4dot, co2_5dot = soln.y[1], soln.y[4], soln.y[7], soln.y[10], soln.y[13]
q1dot, q2dot, q3dot, q4dot, q5dot = soln.y[2], soln.y[5], soln.y[8], soln.y[11], soln.y[14], 

# # print(co2_1dot )
# # print("\n")
# # print( co2_2dot)
# # print( co2_3dot, co2_4dot, co2_5dot)
# temperature_name = ['T1', 'T2', 'T3', 'T4', 'T5']
# temperature_colors = ['lightsteelblue','deepskyblue', 'dodgerblue', 'mediumblue', 'navy']
# # print(temperature_name)
# vec_time = np.linspace(t0, tf, T1.size)
# # vec_Z = np.linspace(0,  V / (math.pi * (r ** 2)), 5)
# box_Z = ['dz1', 'dz2', 'dz3', 'dz4', 'dz5']
# vbar_top = [T, T, T, T, T]
# tem = [T1, T2, T3, T4, T5]

# # # co2_name = [co2_1dot, co2_2dot, co2_3dot, co2_4dot, co2_5dot]
# # # co2_colors = ['darkseagreen','mediumseagreen', 'seagreen', 'green', 'drakolivegreen']
# # # q_name = [q1dot, q2dot, q3dot, q4dot, q5dot]
# # # q_colors = ['mediumpurple', 'blueviolet', 'darkorchid', 'indigo', 'mediumslateblue']

# source_temp0 = ColumnDataSource(data=dict(vec_Z = vec_time, T1=T1, T2 = T2, T3 = T3, T4=T4, T5=T5))
# TOOLTIPS = [("deltaZ","@vec_Z"), ("T1","@T1{0,0.000}"), ("T2","@T2{0,0.000}"), ("T3","@T3{0,0.000}"), ("T4","@T4{0,0.000}"), ("T5","@T5{0,0.000}")]
# TOOLS = "pan,undo,redo,reset,save,wheel_zoom,box_zoom"
# plot_conc = figure(plot_height=450, plot_width=550, tools=TOOLS, tooltips=TOOLTIPS,
#               title="Direct Air Capture", x_range=[t0, tf], y_range=[220, 350])
# plot_conc.line('vec_Z', 'T1', source=source_temp0, line_width=3, line_alpha=0.6, line_color='lightsteelblue',
#                legend_label="T1")
# plot_conc.line('vec_Z', 'T2', source=source_temp0, line_width=3, line_alpha=0.6, line_color='deepskyblue',
#                legend_label="T2")
# plot_conc.line('vec_Z', 'T3', source=source_temp0, line_width=3, line_alpha=0.6, line_color='dodgerblue',
#                legend_label="T3")
# plot_conc.line('vec_Z', 'T4', source=source_temp0, line_width=3, line_alpha=0.6, line_color='mediumblue',
#                legend_label="T4")
# plot_conc.line('vec_Z', 'T5', source=source_temp0, line_width=3, line_alpha=0.6, line_color='navy',
#                legend_label="T5")
# plot_conc.xaxis.axis_label = "deltaZ"
# plot_conc.yaxis.axis_label = "Temeprarture"
# plot_conc.legend.location = "top_left"
# plot_conc.legend.click_policy="hide"
# plot_conc.legend.background_fill_alpha = 0.5
# plot_conc.grid.grid_line_color = "silver"

# source_vbar = ColumnDataSource(data=dict(temperature_name=temperature_name,  temperature_colors=temperature_colors, vbar_top= vbar_top))
# TOOLTIPS_vbar = [("Temperature_Name","@temperature_name"), ("Temperature","@vbar_top{0,0.000}")]
# TOOLS = "pan,undo,redo,reset,save,wheel_zoom,box_zoom"

# plot_vbar = figure(plot_height=450, plot_width=550, tools=TOOLS, tooltips=TOOLTIPS_vbar, x_range=temperature_name,
#                    y_range=[0, 300.25], title="Temperature changes over boxes by time slider")
# plot_vbar.vbar(x='temperature_name', top = "vbar_top",source=source_vbar, bottom=0.0, width=0.5, alpha=0.6, color="temperature_colors",
#                legend_field= "temperature_name")
# plot_vbar.xaxis.axis_label = "Box"
# plot_vbar.yaxis.axis_label = "Temperature"
# plot_vbar.xgrid.grid_line_color = None
# plot_vbar.legend.orientation = "horizontal"
# plot_vbar.legend.location = "top_center"


# V_slider = Slider(title="Volume of bed"+" (initial: "+str(V)+")", value=V, start=98, end=105, step=1)
# # r_slider = Slider(title="Radius of bed"+" (initial: "+str(r)+")", value=r, start=4, end=6, step=0.02)
# T_slider = Slider(title="Ambient temperature"+" (initial: "+str(T)+")", value=T, start=293, end=393, step=1)
# c_co2_0_slider = Slider(title="initial CO2 concentration"+" (initial: "+str(c_co2_0)+")", value=c_co2_0, start=0, end=5, step=0.2)
# episl_r_slider = Slider(title="Episl r"+" (initial: "+str(episl_r)+")", value=episl_r, start=1, end=5, step=1)
# volumetric_flow_slider = Slider(title="Initial flow"+" (initial: "+str(volumetric_flow)+")", value=volumetric_flow, start=1, end=5, step=1)
# # source_tempVbar = ColumnDataSource(data=dict( temperature_name=temperature_name, color=temperature_colors))

# V = 200.0  # volume
# T = 293.0 # +273 ambrient temperature
# c_co2_0 = .016349 # mol/m^3    
# episl_r = 0.3  # void
# volumetric_flow = 5 # m^3/s


# start_time = 0.0
# end_time = 10.0
# time_step = 0.3
# slider_time = Slider(title="Time Slider (s)", value=start_time, start=start_time, end=end_time, step=time_step, width=500)

# def animate_update():
#     current_time = slider_time.value + time_step
#     if current_time > end_time:
#         current_time = start_time
#     slider_time.value = current_time

# def update_data(attrname, old, new):

#     # Get the current slider values
#     V_temp= V_slider.value
#     # r_temp = r_slider.value
#     T_temp = T_slider.value
#     c_co2_0_temp = c_co2_0_slider.value
#     episl_r_temp = episl_r_slider.value
#     volumetric_flow_temp = volumetric_flow_slider.value
#     time_temp = slider_time.value

#     # Generate the new curve
#     params = [V_temp, T_temp, c_co2_0_temp, episl_r_temp, volumetric_flow_temp]
#     soln = solve_ivp(deriv1, (t0, tf), init_cond, args=(params,)) 
#     # vec_Z = np.linspace(0,  V / (math.pi * (r ** 2)), 5)
#     vec_time = np.linspace(t0, tf, soln.y[0].size)  # vector for time
#     T1_temp = soln.y[0]
#     T2_temp = soln.y[3]
#     T3_temp = soln.y[6]
#     T4_temp = soln.y[9]
#     T5_temp = soln.y[12]
#     source_temp0.data =  dict(vec_Z=vec_time,   T1=T1_temp, T2 = T2_temp, T3 = T3_temp, T4=T4_temp, T5=T5_temp)
#     vbar_top_temp = [np.interp(time_temp, vec_time, T1_temp), np.interp(time_temp, vec_time, T2_temp),
#                      np.interp(time_temp, vec_time, T3_temp), np.interp(time_temp, vec_time, T4_temp), np.interp(time_temp, vec_time, T5_temp)]
#     temperature_name = ['T1', 'T2', 'T3', 'T4', 'T5']
#     temperature_colors = ['lightsteelblue','deepskyblue', 'dodgerblue', 'mediumblue', 'navy']
#     source_vbar.data = dict(temperature_name=temperature_name, temperature_colors=temperature_colors, vbar_top = vbar_top_temp)

# for w in [V_slider , T_slider, c_co2_0_slider, episl_r_slider, volumetric_flow_slider, slider_time]:
#     w.on_change('value', update_data)

# def animate():
#     global callback_id  
    
#     if animate_button.label == '► Play':
#         animate_button.label = '❚❚ Pause'
#         callback_id = curdoc().add_periodic_callback(animate_update, time_step*1000.0) # s to milliseconds conversion
#     else:
#         animate_button.label = '► Play'
#         curdoc().remove_periodic_callback(callback_id)

# animate_button = Button(label='► Play', width=50)
# animate_button.on_event('button_click', animate)

# inputs_reaction = column(V_slider , T_slider, c_co2_0_slider, episl_r_slider, volumetric_flow_slider)
# inputs_time = column(animate_button, slider_time )

# tab1 =Panel(child=row(inputs_reaction, plot_conc,  column(plot_vbar, inputs_time, height=450)), title="Desktop")
# tab2 =Panel(child=column(inputs_reaction, plot_conc,  column(plot_vbar, inputs_time, height=475)), title="Mobile")
# tabs = Tabs(tabs = [tab1, tab2])
# curdoc().add_root(tabs)
# curdoc().title = "Direct Air Capture"
# # tabs = Tabs(tabs=[ tab1, tab ])
# # show(tabs)




