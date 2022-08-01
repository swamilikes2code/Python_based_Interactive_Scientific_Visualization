# ''' Present an interactive function explorer with slider widgets.
# There are 5 input parameters that the user can play with it
# Use the ``bokeh serve`` command to run the example by executing:
#     bokeh serve --show dac.py
# at your command prompt. If default port is taken, you can specify
# port using ' --port 5010' at the end of the bokeh command.
# '''
# import math
# from matplotlib.pyplot import plot
# from scipy.integrate import solve_ivp, odeint
# from bokeh.io import save, curdoc
# from bokeh.layouts import column, row, gridplot
# from bokeh.model import Model
# from bokeh.models import CustomJS, Slider, Callback, HoverTool, Button, TextInput
# from bokeh.plotting import ColumnDataSource, figure, show
# from bokeh.models.widgets import Panel, Tabs
# import numpy as np
# import pandas as pd

# # To do list:
# # addd a input to control how much time the simulation can run - tf
# # run time input, make it as input box

# # --------------------- Static Parameters    --------------------- #

# b0 = 93 * (10**-5)  # 93      unit : 1/bars
# deltH_0 = 95300  #               unit: j/mol 
# T0 = 353.15  # reference temeperature to be used in the Toth isotherm   unit: kelvin
# t_h0 = .37  # heterogeneity constant 
# apha = 0.33
# chi = 0.0
# q_s0 = 3.40 # qs_var = q_s0 = 3.4 due to chi = 0 mol/kg
# # R = 8.314* (10**3) # Universal gas constant - LPa/molK
# Rg = .0821 # Universal gas constant in l-atm/mol/K
# kT0 = 3.5 * (10 ** -2)  # used to compute kT for r_CO2... in mol/Kg-pa-sec
# EaCO2 = 15200 # activation for KT -- J/mol
# ps = 880.0 #
# deltH_co2 = 75000.0  # calculate temeprature change   unit: jol/mol

# R_constant = 8.314 # jol/kelvin-mol = m^3-pa/mol-kelvin

# # ------------------ For Equation : Enegergy Ballance  -------------- #
# pg = 1.87  # 
# h = 13.8 # 
# Cp_g = 846.0  # J/kgKelvin
# Cp_s = 1500.0  # J/kgKelvin

# # ------------------ ODE Repetitive Shortcut -------------- #
# def cube(x):
#     if 0<=x: return x**(1./3.)
#     return -(-x)**(1./3.)

# # ------------------ Equastions to calculate Rco2 -------------- #
# def b(T): # will be called inside deriv
#     b = b0 *(math.exp(((deltH_0 / (R_constant * T0)) * (T0 / T - 1))))
#     return b

# def t_h(T): # will be called inside  deriv
#     t_h = t_h0 + (apha * (1 - (T0 / T)) )
#     return (t_h)

# # Calculate rco2_n (not ode)
# def R_co2(T, c_co2, q):
#     kn = kT0 * ( math.exp(( (-EaCO2) / (R_constant*T) )))
#     b_var = b(T)
#     t_var = t_h(T)
#     rco2_1 = R_constant * T * c_co2 
#     rco2_2 = ((1 - ((q / q_s0) ** (t_var))) ** (1 / t_var))
#     rco2_3 = q / (b_var * q_s0)
#     rco2 = kn * (rco2_1 * rco2_2 - rco2_3) 
#     return rco2

# """
#     Defines the differential equations for odes of DAC
#     Arguments:
#         y :  vector of the state variables:
#                   y = all 15 states
#         t :  time
#         params :  vector of the parameters:
#                   params = [V, r, T, c_co2_0, episl_r, v0]
    
# """
# LoverR = 2.5/20 # a constant used to calculated radius, Straight from the paper - shallow bed, so L << R

# def deriv1(t, y, params):
#     T_n, co2_n, q_n, T_n2, co2_n2, q_n2,T_n3, co2_n3, q_n3, T_n4, co2_n4, q_n4,T_n5, co2_n5, q_n5 = y # the rest of 12 vars a_n are not used, only for the success of solve_ivp
#     V, T_in, c_co2_0, episl_r, volumetric_flow, Tw = params

#     ###############   -----  Parameters depend on input  -----  ###############

#     r = cube(V/(LoverR*math.pi))
#     v0 = volumetric_flow / (math.pi *r*r )
#     L = V / (math.pi * (r ** 2))
#     deltZ = L / 5.0  # 5 boxes in total
#     a_s = 150 #Straight from the paper
#     theta = (1 - episl_r) * ps * Cp_s + episl_r * pg * Cp_g
    
#     ###############   -----  Constants used in Equations  -----  ###############
#     temperature_constant = ((v0  * pg* Cp_g) / (theta * deltZ))
#     temperature_constant2 = (1 - episl_r) * ps * deltH_co2 /theta 
#     temperature_constant3 = a_s * h /theta
#     concentration_constant = v0 / (episl_r * deltZ)
#     concentration_constant2 = (1 - episl_r) * ps/episl_r 
    

#     T1dot = -temperature_constant* T_n + temperature_constant* T_in + temperature_constant2* (
#         R_co2(T_n, co2_n, q_n))+ temperature_constant3*(Tw - T_n)
#     co2_1dot = -concentration_constant * co2_n + concentration_constant * c_co2_0 - (
#         R_co2(T_n, co2_n, q_n)) * concentration_constant2
#     q1dot = R_co2(T_n, co2_n, q_n)
    
#     T2dot = -temperature_constant * T_n2 + temperature_constant * T_n +temperature_constant2 *(
#         R_co2(T_n2, co2_n2, q_n2)) + temperature_constant3*(Tw - T_n2)
#     co2_2dot = -concentration_constant* co2_n2 + concentration_constant * co2_n - (
#         R_co2(T_n2, co2_n2, q_n2)) * concentration_constant2
#     q2dot = R_co2(T_n2, co2_n2, q_n2)

#     T3dot = -temperature_constant* T_n3 + temperature_constant * T_n2 + temperature_constant2* (
#         R_co2(T_n3, co2_n3, q_n3)) + temperature_constant3 * (Tw - T_n3)
#     co2_3dot = -concentration_constant * co2_n3 + concentration_constant * co2_n2 - (
#         R_co2(T_n3, co2_n3, q_n3)) * concentration_constant2
#     q3dot = R_co2(T_n3, co2_n3, q_n3)

#     T4dot = -temperature_constant* T_n4 + temperature_constant* T_n3 + temperature_constant2*(
#         R_co2(T_n4, co2_n4, q_n4)) + temperature_constant3 * (Tw -  T_n4)
#     co2_4dot = -v0 / (episl_r * deltZ)* co2_n4 + v0 / (episl_r * deltZ)* co2_n3 - (
#         R_co2(T_n4, co2_n4, q_n4)) * concentration_constant2
#     q4dot = R_co2(T_n4, co2_n4, q_n4)

#     T5dot = -temperature_constant * T_n5 + temperature_constant * T_n4 + temperature_constant2 * (
#         R_co2(T_n5, co2_n5, q_n5))+ temperature_constant3*(Tw - T_n5)
#     co2_5dot = -concentration_constant * co2_n5 + concentration_constant * co2_n4 - (
#         R_co2(T_n5, co2_n5, q_n5)) * concentration_constant2
#     q5dot = R_co2(T_n5, co2_n5, q_n5)

#     return [T1dot, co2_1dot, q1dot, T2dot, co2_2dot, q2dot, T3dot, co2_3dot, q3dot, T4dot, co2_4dot, q4dot, T5dot, co2_5dot, q5dot]

# # ------------------ User generated - Slider initial value -------------- #
# V = .003  # volume
# T_in= 298.0 # +273 ambient temperature, T_in ambient temperature,  also inlet temperature, in kelvin  unit: kelvin, depends on location
# c_co2_0 =co2_initial = .016349 # mol/m^3    
# episl_r = 0.3  # void
# volumetric_flow = .01 # m^3/s
# Tw = 293.0 # water temperature, utility
# # air humidity 

# # ------------------ Initial Conditions to set up solve_ivp -------------- #
# # text_input = TextInput(name='Text Input', value = "6", width = 80)
# # # text_input.js_on_change('value',code='''source.change.emit()''')
# # tf = float((text_input.value)) * 3600 # convert to seconds from hrs
# # print(tf)
# tf = 432000
# t0= 0.0 # 12hrs
# q_init_cond = 0
# T_initial = T_in # initial temperature
# init_cond = [T_initial, co2_initial, q_init_cond] * 5
# # ,20.000, 0.000, 0.000,20.000, 0.000, 0.000,20.000, 0.000, 0.000,20.000, 0.000, 0.000
# params = [V, T_in, c_co2_0, episl_r, volumetric_flow, Tw]
# N = 25 # Number of points 
# tspan = np.linspace(t0, tf, N)

# soln = solve_ivp(deriv1, (t0, tf), init_cond, args=(params,), t_eval = tspan, method = "BDF", rtol = 1e-5, atol = 1e-8)  # init_cond = (T, c_co2_0, q0)
# # soln = solve_ivp(deriv1, (t0, tf), init_cond, args=(params,), method = "BDF", rtol = 1e-5, atol = 1e-8)  # init_cond = (T, c_co2_0, q0)
# # deriv1([t0, tf], )
# # print(soln)
# ## --------------------  Extract Figures from returned solve results and match them with Z 
# def Extract(lst, term):
#     return [item[term] for item in lst]
# dotT= [soln.y[0], soln.y[3], soln.y[6], soln.y[9], soln.y[12]]
# dotCo2 = [soln.y[1], soln.y[4], soln.y[7], soln.y[10], soln.y[13]]
# dotQ = [soln.y[2], soln.y[5], soln.y[8], soln.y[11], soln.y[14]]

# def mapWithL(input_array, initial_value):
#     res = {}
#     for i in range(0, N):
#         res[str(i)] = Extract(input_array, i)

#     res_list = list(res.values()) # make it as np array
#     for i in range(0, N):
#         res_list[i].insert(0, initial_value)

#     np.array(res_list)
#     return res_list

# # Set up sliders 
# V_slider = Slider(title="Volume of bed"+" (default: "+str(V)+" m^3)", value=V, start=.001, end=.005, step=.001)
# T_in_slider = Slider(title="Ambient temperature"+" (default: "+str(T_in)+" K)", value=T_in, start=285, end=310, step=1)
# c_co2_0_slider = Slider(title="Inlet CO2 concentration"+" (default: "+str(c_co2_0)+" mol/m^3)", value=c_co2_0, start=0.0, end=0.03, step=0.005)
# episl_r_slider = Slider(title="Episl r"+" (default: "+str(episl_r)+")", value=episl_r, start= .3, end= .5, step=.03)
# volumetric_flow_slider = Slider(title="Initial flow"+" (default: "+str(volumetric_flow)+")", value=volumetric_flow, start=.001, end=1, step=.005)
# Tw_slider = Slider(title="Water temperature"+" (default: "+str(Tw)+" K)", value=Tw, start=293, end=310, step=1)
# time_step = tspan[1] # since t_span[0] is 0
# slider_time = Slider(title="Time Slider (s)", value=t0, start=t0, end=tf, step=time_step, width=300)

# def getVecZ():
    
#     V0 = V_slider.value
#     r = cube(V0/(LoverR*math.pi))
#     L = V0 / (math.pi * (r ** 2))
#     vec_Z = np.linspace(0, L, 6) # 
#     return vec_Z

# temp_list = mapWithL(dotT, T_in)
# co2_array = mapWithL(dotCo2, co2_initial)
# q_array = mapWithL(dotQ, q_init_cond)
# # r = cube(V/(20*math.pi))
# # L = V / (math.pi * (r ** 2))
# vec_Z = getVecZ()
# # print(vec_Z)
# L = vec_Z[5]
# temp_df = pd.DataFrame(temp_list, tspan)
# co2_df = pd.DataFrame(co2_array, tspan)
# q_df =  pd.DataFrame(q_array, tspan)
# # temp_list
# Tools = "crosshair,pan,reset,undo,box_zoom, save,wheel_zoom",

# source_temperature = ColumnDataSource(data=dict(x=vec_Z, y=temp_df.iloc[0]))
# plot_temperature = figure(height=370, width=400, title="Axial Profile of Column Temperature ",
#               tools= Tools,
#               x_range=[0, L], y_range=[296, 299])
# plot_temperature.line('x', 'y',  line_width=3, source = source_temperature, line_alpha=0.6, color = "navy")
# plot_temperature.xaxis.axis_label = "L (m)"
# plot_temperature.yaxis.axis_label = "Temperature (K)"

# source_co2 = ColumnDataSource(data=dict(co2_x=vec_Z, co2_y = co2_df.iloc[0]))
# plot_co2 = figure(height=370, width=400, title="Axial Profile of Gas Phase CO2",
#               tools=Tools,
#               x_range=[0, L], y_range=[0, .03])
# plot_co2.line('co2_x', 'co2_y',  line_width=3, source = source_co2, line_alpha=0.6, color = "navy")
# plot_co2.xaxis.axis_label = "L (m)"
# plot_co2.yaxis.axis_label = "Gaseous Concentration of CO2 (mol/m^3)"

# source_q = ColumnDataSource(data=dict(q_x=vec_Z, q_y = q_df.iloc[0]))
# plot_q = figure(height=370, width=400, title="Axial profile of adsorbed CO2",
#               tools=Tools,
#               x_range=[0, L], y_range=[0, 1.2])
# plot_q.line('q_x', 'q_y',  line_width=3, source = source_q, line_alpha=0.6, color = "navy")
# plot_q.xaxis.axis_label = "L (m)"
# plot_q.yaxis.axis_label = "CO2 Adsorbed (mol/kg)"

# def update_data(attrname, old, new):

#     # Get the current slider values
#     V_temp = V_slider.value
#     T_in_temp = T_in_slider.value
#     c_co2_0_temp = c_co2_0_slider.value
#     episl_r_temp = episl_r_slider.value
#     # print(episl_r_temp)
#     volumetric_flow_temp = volumetric_flow_slider.value

#     Tw_temp = Tw_slider.value
#     # time_temp = slider_time.value

#     # Generate the new curve
#     params_temp = [V_temp, T_in_temp , c_co2_0_temp, episl_r_temp, volumetric_flow_temp, Tw_temp]
#     init_cond_temp = [T_in_temp, c_co2_0_temp, q_init_cond] * 5
#     soln = solve_ivp(deriv1, (t0, tf), init_cond_temp, args=(params_temp,), t_eval = tspan, method = "BDF", rtol = 1e-5, atol = 1e-8) 
#     dotT = [soln.y[0], soln.y[3], soln.y[6], soln.y[9], soln.y[12]]
#     dotCo2 = [soln.y[1], soln.y[4], soln.y[7], soln.y[10], soln.y[13]]
#     dotQ = [soln.y[2], soln.y[5], soln.y[8], soln.y[11], soln.y[14]]

#     temp_list = mapWithL(dotT, T_initial)
#     co2_array = mapWithL(dotCo2, c_co2_0_temp)
#     q_array = mapWithL(dotQ, q_init_cond)    

#     vec_Z = getVecZ()
#     # L = vec_Z[5]
#     temp_df = pd.DataFrame(temp_list, tspan)
#     co2_df = pd.DataFrame(co2_array, tspan)
#     q_df =  pd.DataFrame(q_array, tspan)

#     source_temperature.data = dict(x=vec_Z, y=temp_df.iloc[1])
#     source_co2.data = dict(co2_x = vec_Z, co2_y = co2_df.iloc[1])
#     source_q.data = dict(q_x = vec_Z, q_y = q_df.iloc[1])

# def animate_update():
#     current_time = slider_time.value +  time_step
#     if current_time > tf:
#         current_time = t0

#     source_temperature.data = dict(x=vec_Z, y=temp_df.loc[current_time])
#     source_co2.data = dict(co2_x=vec_Z, co2_y=co2_df.loc[current_time])
#     source_q.data = dict(q_x=vec_Z, q_y=q_df.loc[current_time])
#     slider_time.value = current_time

# for w in [V_slider , T_in_slider, c_co2_0_slider, episl_r_slider, volumetric_flow_slider, Tw_slider]:
#     w.on_change('value', update_data)

# # time_button = Button(label='Reset', width = 80)
# # time_button.on_event('button_click', reset)


# def animate():
#     global callback_id
#     if animate_button.label == '► Play':

#         animate_button.label = '❚❚ Pause'

#         callback_id = curdoc().add_periodic_callback(animate_update, 1*1000.0) # s to milliseconds conversion
#     else:
#         animate_button.label = '► Play'
#         curdoc().remove_periodic_callback(callback_id)

# animate_button = Button(label='► Play', width=80)
# animate_button.on_event('button_click', animate)

# def reset():
#     source_temperature.data = dict(x=vec_Z, y=temp_df.loc[1])
#     source_co2.data = dict(co2_x=vec_Z, co2_y=co2_df.loc[1])
#     source_q.data = dict(q_x=vec_Z, q_y=q_df.loc[1])
#     slider_time.value = 0.0
#     # V_slider.value = 0.003
#     # T_in_slider.value = 298
#     # c_co2_0_slider.value = 0.016349 
#     # episl_r_slider.value = 0.30
#     # volumetric_flow_slider.value = 0.01
#     # Tw_slider.value = 293
    
# reset_button = Button(label='Reset', width = 80)
# reset_button.on_event('button_click', reset)

# inputs_reaction = (column(V_slider , T_in_slider, c_co2_0_slider, episl_r_slider, volumetric_flow_slider, Tw_slider))

# inputs_button = row(slider_time, animate_button, reset_button)

# inputs = column(inputs_reaction, inputs_button)

# grid = gridplot([[inputs, plot_q], [plot_co2, plot_temperature ]])

# tab1 =Panel(child= grid, title="Desktop")
# tab2 =Panel(child=column(plot_temperature, row( inputs_reaction, height=450)), title="Phone")
# tabs = Tabs(tabs = [tab1, tab2])


# curdoc().add_root(tabs)
# curdoc().title = "Direct Air Caoture"


''' Present an interactive function explorer with slider widgets.
There are 5 input parameters that the user can play with it
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve --show dac.py
at your command prompt. If default port is taken, you can specify
port using ' --port 5010' at the end of the bokeh command.
'''
import math
from matplotlib.pyplot import plot
from scipy.integrate import solve_ivp, odeint
from bokeh.io import save, curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.model import Model
from bokeh.models import CustomJS, Slider, Callback, HoverTool, Button
from bokeh.plotting import ColumnDataSource, figure, show
from bokeh.models.widgets import Panel, Tabs
import numpy as np
import pandas as pd

# To do list:
# addd a input to control how much time the simulation can run - tf
# run time input, make it as input box
# reset the slider


# --------------------- Static Parameters    --------------------- #

b0 = 93 * (10**-5)  # 93      unit : 1/bars
deltH_0 = 95300  #               unit: j/mol 
T0 = 353.15  # reference temeperature to be used in the Toth isotherm   unit: kelvin
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

R_constant = 8.314 # jol/kelvin-mol = m^3-pa/mol-kelvin

# ------------------ For Equation : Enegergy Ballance  -------------- #
pg = 1.87  # 
h = 13.8 # 
Cp_g = 846.0  # J/kgKelvin
Cp_s = 1500.0  # J/kgKelvin

# ------------------ ODE Repetitive Shortcut -------------- #
def cube(x):
    if 0<=x: return x**(1./3.)
    return -(-x)**(1./3.)

# ------------------ Equastions to calculate Rco2 -------------- #
def b(T): # will be called inside deriv
    b = b0 *(math.exp(((deltH_0 / (R_constant * T0)) * (T0 / T - 1))))
    return b

def t_h(T): # will be called inside  deriv
    t_h = t_h0 + (apha * (1 - (T0 / T)) )
    return (t_h)

# Calculate rco2_n (not ode)
def R_co2(T, c_co2, q):
    kn = kT0 * ( math.exp(( (-EaCO2) / (R_constant*T) )))
    b_var = b(T)
    t_var = t_h(T)
    rco2_1 = R_constant * T * c_co2 
    rco2_2 = ((1 - ((q / q_s0) ** (t_var))) ** (1 / t_var))
    rco2_3 = q / (b_var * q_s0)
    rco2 = kn * (rco2_1 * rco2_2 - rco2_3) 
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
LoverR = 2.5/20 # Straight from the paper - shallow bed, so L << R
def deriv1(t, y, params):
    T_n, co2_n, q_n, T_n2, co2_n2, q_n2,T_n3, co2_n3, q_n3, T_n4, co2_n4, q_n4,T_n5, co2_n5, q_n5 = y # the rest of 12 vars a_n are not used, only for the success of solve_ivp
    V, T_in, c_co2_0, episl_r, volumetric_flow, Tw = params

    ###############   -----  Parameters depend on input  -----  ###############
    # LoverR = 2.5/20 # Straight from the paper - shallow bed, so L << R
    r = cube(V/(LoverR*math.pi))
#     print(f"r",{r})
    v0 = volumetric_flow / (math.pi *r*r )
    L = V / (math.pi * (r ** 2))
    deltZ = L / 5.0  # 5 boxes in total
    a_s = 150 #Straight from the paper
    theta = (1 - episl_r) * ps * Cp_s + episl_r * pg * Cp_g
    
    temperature_constant = ((v0  * pg* Cp_g) / (theta * deltZ))
    temperature_constant2 = (1 - episl_r) * ps * deltH_co2 /theta 
    temperature_constant3 = a_s * h /theta
    concentration_constant = v0 / (episl_r * deltZ)
    concentration_constant2 = (1 - episl_r) * ps/episl_r 
    

    T1dot = -temperature_constant* T_n + temperature_constant* T_in + temperature_constant2* (
        R_co2(T_n, co2_n, q_n))+ temperature_constant3*(Tw - T_n)
    # print(f"T1", {T1})
    co2_1dot = -concentration_constant * co2_n + concentration_constant * c_co2_0 - (
        R_co2(T_n, co2_n, q_n)) * concentration_constant2
    q1dot = R_co2(T_n, co2_n, q_n)
    # print(f"energy balance in T1", {ener_balan(v0, theta, deltZ)})
    
    T2dot = -temperature_constant * T_n2 + temperature_constant * T_n +temperature_constant2 *(
        R_co2(T_n2, co2_n2, q_n2)) + temperature_constant3*(Tw - T_n2)
    # print(f"T2", {T2})
    co2_2dot = -concentration_constant* co2_n2 + concentration_constant * co2_n - (
        R_co2(T_n2, co2_n2, q_n2)) * concentration_constant2
    q2dot = R_co2(T_n2, co2_n2, q_n2)
    # print(f"energy balance in T1", {ener_balan(v0, theta, deltZ)})

    T3dot = -temperature_constant* T_n3 + temperature_constant * T_n2 + temperature_constant2* (
        R_co2(T_n3, co2_n3, q_n3)) + temperature_constant3 * (Tw - T_n3)
    co2_3dot = -concentration_constant * co2_n3 + concentration_constant * co2_n2 - (
        R_co2(T_n3, co2_n3, q_n3)) * concentration_constant2
    q3dot = R_co2(T_n3, co2_n3, q_n3)

    T4dot = -temperature_constant* T_n4 + temperature_constant* T_n3 + temperature_constant2*(
        R_co2(T_n4, co2_n4, q_n4)) + temperature_constant3 * (Tw -  T_n4)
    co2_4dot = -v0 / (episl_r * deltZ)* co2_n4 + v0 / (episl_r * deltZ)* co2_n3 - (
        R_co2(T_n4, co2_n4, q_n4)) * concentration_constant2
    q4dot = R_co2(T_n4, co2_n4, q_n4)

    T5dot = -temperature_constant * T_n5 + temperature_constant * T_n4 + temperature_constant2 * (
        R_co2(T_n5, co2_n5, q_n5))+ temperature_constant3*(Tw - T_n5)
    co2_5dot = -concentration_constant * co2_n5 + concentration_constant * co2_n4 - (
        R_co2(T_n5, co2_n5, q_n5)) * concentration_constant2
    q5dot = R_co2(T_n5, co2_n5, q_n5)

    # result = np.array([T1, T2, T3, T4, T5, co2_1dot, co2_2dot, co2_3dot, co2_4dot, co2_5dot, q1dot, q2dot, q3dot, q4dot, q5dot]).reshape(-1, 1)

    return [T1dot, co2_1dot, q1dot, T2dot, co2_2dot, q2dot, T3dot, co2_3dot, q3dot, T4dot, co2_4dot, q4dot, T5dot, co2_5dot, q5dot]

# ------------------ User generated - Slider initial value -------------- #
V = .003  # volume
T_in= 298.0 # +273 ambient temperature, T_in ambient temperature,  also inlet temperature, in kelvin  unit: kelvin, depends on location
c_co2_0 = .016349 # mol/m^3    
episl_r = 0.3  # void
volumetric_flow = .01 # m^3/s
Tw = 293.0 # water temperature, utility
# air humidity 
# no radius and length, nonly nr *reed V, july 6th

# ------------------ Initial Conditions to set up solve_ivp -------------- #
t0, tf = 0.0, 43200.0 # 12hrs
co2_initial = 0 
q_init_cond = 0
T_initial = T_in # initial temperature
init_cond = [T_initial, co2_initial, q_init_cond] * 5
# ,20.000, 0.000, 0.000,20.000, 0.000, 0.000,20.000, 0.000, 0.000,20.000, 0.000, 0.000
params = [V, T_in, c_co2_0, episl_r, volumetric_flow, Tw]
N = 25 # Number of points 
tspan = np.linspace(t0, tf, N)

soln = solve_ivp(deriv1, (t0, tf), init_cond, args=(params,), t_eval = tspan, method = "BDF", rtol = 1e-5, atol = 1e-8)  # init_cond = (T, c_co2_0, q0)
# soln = solve_ivp(deriv1, (t0, tf), init_cond, args=(params,), method = "BDF", rtol = 1e-5, atol = 1e-8)  # init_cond = (T, c_co2_0, q0)
# deriv1([t0, tf], )
# print(soln)
## --------------------  Extract Figures from returned solve results and match them with Z 
def Extract(lst, term):
    return [item[term] for item in lst]
dotT= [soln.y[0], soln.y[3], soln.y[6], soln.y[9], soln.y[12]]
dotCo2 = [soln.y[1], soln.y[4], soln.y[7], soln.y[10], soln.y[13]]
dotQ = [soln.y[2], soln.y[5], soln.y[8], soln.y[11], soln.y[14]]

def mapWithL(input_array, initial_value):
    res = {}
    for i in range(0, N):
        res[str(i)] = Extract(input_array, i)

    res_list = list(res.values()) # make it as np array
    for i in range(0, N):
        res_list[i].insert(0, initial_value)

    np.array(res_list)
    return res_list

# Set up sliders 
V_slider = Slider(title="Volume of bed"+" (default: "+str(V)+" m^3)", value=V, start=.001, end=.005, step=.001)
T_in_slider = Slider(title="Ambient temperature"+" (default: "+str(T_in)+" K)", value=T_in, start=285, end=310, step=1)
c_co2_0_slider = Slider(title="Inlet CO2 concentration"+" (default: "+str(c_co2_0)+" mol/m^3)", value=c_co2_0, start=0.0, end=0.03, step=0.005)
episl_r_slider = Slider(title="Porocity"+" (default: "+str(episl_r)+")", value=episl_r, start= .3, end= .5, step=.03)
volumetric_flow_slider = Slider(title="Initial flow"+" (default: "+str(volumetric_flow)+")", value=volumetric_flow, start=.001, end=1, step=.005)
Tw_slider = Slider(title="Water temperature"+" (default: "+str(Tw)+" K)", value=Tw, start=293, end=310, step=1)
time_step = tspan[1] # since t_span[0] is 0
slider_time = Slider(title="Time Slider (s)", value=t0, start=t0, end=tf, step=time_step, width=300)

def getVecZ():
    
    V0 = V_slider.value
    r = cube(V0/(LoverR*math.pi))
    L = V0 / (math.pi * (r ** 2))
    vec_Z = np.linspace(0, L, 6) # 
    return vec_Z

temp_list = mapWithL(dotT, T_in)
co2_array = mapWithL(dotCo2, co2_initial)
q_array = mapWithL(dotQ, q_init_cond)
# r = cube(V/(20*math.pi))
# L = V / (math.pi * (r ** 2))
vec_Z = getVecZ()
# print(vec_Z)
L = vec_Z[5]
temp_df = pd.DataFrame(temp_list, tspan)
co2_df = pd.DataFrame(co2_array, tspan)
q_df =  pd.DataFrame(q_array, tspan)
# temp_list
Tools = "crosshair,pan,reset,undo,box_zoom, save,wheel_zoom",

source_temperature = ColumnDataSource(data=dict(x=vec_Z, y=temp_df.iloc[1]))
plot_temperature = figure(height=370, width=400, title="Axial Profile of Column Temperature ",
              tools= Tools,
              x_range=[0, L], y_range=[296, 299])
plot_temperature.line('x', 'y',  line_width=3, source = source_temperature, line_alpha=0.6, color = "navy")
plot_temperature.xaxis.axis_label = "L (m)"
plot_temperature.yaxis.axis_label = "Temperature (K)"

source_co2 = ColumnDataSource(data=dict(co2_x=vec_Z, co2_y = co2_df.iloc[1]))
plot_co2 = figure(height=370, width=400, title="Axial Profile of Gas Phase CO2",
              tools=Tools,
              x_range=[0, L], y_range=[0, .02])
plot_co2.line('co2_x', 'co2_y',  line_width=3, source = source_co2, line_alpha=0.6, color = "navy")
plot_co2.xaxis.axis_label = "L (m)"
plot_co2.yaxis.axis_label = "Gaseous Concentration of CO2 (mol/m^3)"

source_q = ColumnDataSource(data=dict(q_x=vec_Z, q_y = q_df.iloc[1]))
plot_q = figure(height=370, width=400, title="Axial profile of adsorbed CO2",
              tools=Tools,
              x_range=[0, L], y_range=[0, 1.2])
plot_q.line('q_x', 'q_y',  line_width=3, source = source_q, line_alpha=0.6, color = "navy")
plot_q.xaxis.axis_label = "L (m)"
plot_q.yaxis.axis_label = "CO2 Adsorbed (mol/kg)"

def update_data(attrname, old, new):

    # Get the current slider values
    V_temp = V_slider.value
    T_in_temp = T_in_slider.value
    c_co2_0_temp = c_co2_0_slider.value
    episl_r_temp = episl_r_slider.value
    volumetric_flow_temp = volumetric_flow_slider.value
    Tw_temp = Tw_slider.value
    # time_temp = slider_time.value

    # Generate the new curve
    params_temp = [V_temp, T_in_temp , c_co2_0_temp, episl_r_temp, volumetric_flow_temp, Tw_temp]
    init_cond_temp = [T_in_temp, c_co2_0, q_init_cond] * 5
    soln = solve_ivp(deriv1, (t0, tf), init_cond_temp, args=(params_temp,), t_eval = tspan, method = "BDF", rtol = 1e-5, atol = 1e-8) 
    dotT = [soln.y[0], soln.y[3], soln.y[6], soln.y[9], soln.y[12]]
    dotCo2 = [soln.y[1], soln.y[4], soln.y[7], soln.y[10], soln.y[13]]
    dotQ = [soln.y[2], soln.y[5], soln.y[8], soln.y[11], soln.y[14]]

    temp_list = mapWithL(dotT, T_initial)
    co2_array = mapWithL(dotCo2, co2_initial)
    q_array = mapWithL(dotQ, q_init_cond)    

    vec_Z = getVecZ()
    # L = vec_Z[5]
    temp_df = pd.DataFrame(temp_list, tspan)
    co2_df = pd.DataFrame(co2_array, tspan)
    q_df =  pd.DataFrame(q_array, tspan)

    source_temperature.data = dict(x=vec_Z, y=temp_df.iloc[1])
    source_co2.data = dict(co2_x = vec_Z, co2_y = co2_df.iloc[1])
    source_q.data = dict(q_x = vec_Z, q_y = q_df.iloc[1])

def animate_update():
    current_time = slider_time.value +  time_step
    if current_time > tf:
        current_time = t0

    source_temperature.data = dict(x=vec_Z, y=temp_df.loc[current_time])
    source_co2.data = dict(co2_x=vec_Z, co2_y=co2_df.loc[current_time])
    source_q.data = dict(q_x=vec_Z, q_y=q_df.loc[current_time])
    slider_time.value = current_time
    V_slider.value = 0.003
    T_in_slider.value = 298
    c_co2_0_slider.value = 0.016349 
    episl_r_slider.value = 0.30
    volumetric_flow_slider.value = 0.01
    Tw_slider.value = 293

for w in [V_slider , T_in_slider, c_co2_0_slider, episl_r_slider, volumetric_flow_slider, Tw_slider]:
    w.on_change('value', update_data)

def animate():
    global callback_id
    if animate_button.label == '► Play':

        animate_button.label = '❚❚ Pause'

        callback_id = curdoc().add_periodic_callback(animate_update, 1*1000.0) # s to milliseconds conversion
    else:
        animate_button.label = '► Play'
        curdoc().remove_periodic_callback(callback_id)

animate_button = Button(label='► Play', width=80)
animate_button.on_event('button_click', animate)

def reset():
    source_temperature.data = dict(x=vec_Z, y=temp_df.loc[0])
    source_co2.data = dict(co2_x=vec_Z, co2_y=co2_df.loc[0])
    source_q.data = dict(q_x=vec_Z, q_y=q_df.loc[0])
    slider_time.value = 0.0
reset_button = Button(label='Reset', width = 80)
reset_button.on_event('button_click', reset)

inputs_reaction = (column(V_slider , T_in_slider, c_co2_0_slider, episl_r_slider, volumetric_flow_slider, Tw_slider))

inputs_button = row(slider_time, animate_button, reset_button)

inputs = column(inputs_reaction, inputs_button)

grid = gridplot([[inputs, plot_q], [plot_co2, plot_temperature ]])

tab1 =Panel(child= grid, title="Desktop")
tab2 =Panel(child=column(plot_temperature, row( inputs_reaction, height=450)), title="Phone")
tabs = Tabs(tabs = [tab1, tab2])


curdoc().add_root(tabs)
curdoc().title = "Direct Air Caoture"
