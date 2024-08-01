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
from bokeh.layouts import column, row, gridplot
from bokeh.model import Model
from bokeh.models import CustomJS, Slider, Callback, HoverTool, Button, TabPanel, Tabs, Spacer
from bokeh.plotting import ColumnDataSource, figure, show
import numpy as np
import pandas as pd

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
    Defines the 15 systems of differential equations for odes of DAC
    Arguments:
        y :  vector of the state variables:
                  y = all 15 states in 5 columns
        t :  time
        params :  vector of the parameters:
                  params = [V, T_in, c_co2_0, episl_r, volumetric_flow, Tw]
    
"""
LoverR = 2.5/20 # Straight from the paper - shallow bed, so L << R
def deriv1(t, y, params):
    T_n, co2_n, q_n, T_n2, co2_n2, q_n2,T_n3, co2_n3, q_n3, T_n4, co2_n4, q_n4,T_n5, co2_n5, q_n5 = y # the rest of 12 vars a_n are not used, only for the success of solve_ivp
    V, T_in, c_co2_0, episl_r, volumetric_flow, Tw = params

    ###############   -----  Parameters depend on Inputs  -----  ###############
    #------ LoverR = 2.5/20 # Straight from the paper - shallow bed, so L << R ----- #
    r = cube(V/(LoverR*math.pi))

    v0 = volumetric_flow / (math.pi *r*r )
    L = V / (math.pi * (r ** 2))
    deltZ = L / 5.0  # 5 boxes in total
    a_s = 150 #Straight from the paper
    theta = (1 - episl_r) * ps * Cp_s + episl_r * pg * Cp_g
    
    # ------------------ Repetitive Terms in Equastions  -------------- #
    temperature_constant = ((v0  * pg* Cp_g) / (theta * deltZ))
    temperature_constant2 = (1 - episl_r) * ps * deltH_co2 /theta 
    temperature_constant3 = a_s * h /theta
    concentration_constant = v0 / (episl_r * deltZ)
    concentration_constant2 = (1 - episl_r) * ps/episl_r 
    
    # ------------------ Set up the Differential Equastions -------------- #
    T1dot = -temperature_constant* T_n + temperature_constant* T_in + temperature_constant2* (
        R_co2(T_n, co2_n, q_n))+ temperature_constant3*(Tw - T_n)
    co2_1dot = -concentration_constant * co2_n + concentration_constant * c_co2_0 - (
        R_co2(T_n, co2_n, q_n)) * concentration_constant2
    q1dot = R_co2(T_n, co2_n, q_n)
    
    T2dot = -temperature_constant * T_n2 + temperature_constant * T_n +temperature_constant2 *(
        R_co2(T_n2, co2_n2, q_n2)) + temperature_constant3*(Tw - T_n2)
    # print(f"T2", {T2})
    co2_2dot = -concentration_constant* co2_n2 + concentration_constant * co2_n - (
        R_co2(T_n2, co2_n2, q_n2)) * concentration_constant2
    q2dot = R_co2(T_n2, co2_n2, q_n2)

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

# ------------------ Initial Conditions to set up solve_ivp -------------- #
t0, tf = 0.0, 43200.0 # 12hrs
co2_initial = 0 
q_init_cond = 0
T_initial = T_in # initial temperature
init_cond = [T_initial, co2_initial, q_init_cond] * 5
# print(init_cond)
params = [V, T_in, c_co2_0, episl_r, volumetric_flow, Tw]
N = 25 # Number of points 
tspan = np.linspace(t0, tf, N)

soln = solve_ivp(deriv1, (t0, tf), init_cond, args=(params,), t_eval = tspan, method = "BDF", rtol = 1e-5, atol = 1e-8)  # init_cond = (T, c_co2_0, q0)

## --------------------  Extract Figures from returned solve results and match them with Z  ------------- #
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

## --------------------  Set up Sliders ------------------------- ##
V_slider = Slider(title="Volume of bed"+" (default: "+str(V*1000)+" L)", value=V*1000, start=1, end=5, step=1)
T_in_slider = Slider(title="Ambient temperature"+" (default: "+str(T_in)+" K)", value=T_in, start=285, end=310, step=1)
c_co2_0_slider = Slider(title="Inlet CO2 concentration"+" (default: "+str(c_co2_0)+" mol/m^3)", value=c_co2_0, start=0.0, end=0.03, step=0.005)
episl_r_slider = Slider(title="Porosity"+" (default: "+str(episl_r)+")", value=episl_r, start= .3, end= .5, step=.03)
volumetric_flow_slider = Slider(title="Inlet flow"+" (default: "+str(volumetric_flow)+")", value=volumetric_flow, start=.001, end=1, step=.005)
Tw_slider = Slider(title="Water temperature"+" (default: "+str(Tw)+" K)", value=Tw, start=293, end=310, step=1)
time_step = tspan[1] # since t_span[0] is 0
slider_time = Slider(title="Time Slider (s)", value=t0, start=t0, end=tf, step=time_step, width=300)

## --------------------  Map the discretized data with time ------------------------- #
def getVecZ():
    V0 = V_slider.value
    r = cube(V0/(LoverR*math.pi))
    L = V0 / (math.pi * (r ** 2))
    vec_Z = np.linspace(0, L, 6) # 
    return vec_Z

temp_list = mapWithL(dotT, T_in)
co2_array = mapWithL(dotCo2, c_co2_0_slider.value)
q_array = mapWithL(dotQ, q_init_cond)
vec_Z = getVecZ()
L = vec_Z[5]
temp_df = pd.DataFrame(temp_list, tspan) # Turn list into pandadf
co2_df = pd.DataFrame(co2_array, tspan)
q_df =  pd.DataFrame(q_array, tspan)

## --------------------  Start Plotting ------------------------- ##
# Tools = "crosshair,pan,reset,undo,box_zoom, save,wheel_zoom",
Tools = "crosshair,reset,undo,box_zoom, save,wheel_zoom",

source_temperature = ColumnDataSource(data=dict(temp_x=vec_Z, temp_y=temp_df.iloc[1]))
plot_temperature = figure(height=370, width=400, title="Axial Profile of Column Temperature ",
              tools= Tools,
              x_range=[0, L], y_range=[292, 299])
plot_temperature.line('temp_x', 'temp_y',  line_width=3, source = source_temperature, line_alpha=0.6, color = "navy")
plot_temperature.xaxis.axis_label = "L (m)"
plot_temperature.yaxis.axis_label = "Temperature (K)"

source_co2 = ColumnDataSource(data=dict(co2_x=vec_Z, co2_y = co2_df.iloc[1]))
plot_co2 = figure(height=370, width=400, title="Axial Profile of Gas Phase CO2",
              tools=Tools,
              x_range=[0, L], y_range=[0, .03])
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

    ## --------------------  Update the graphs when changing data ------------------------- ##
    params_temp = [V_temp, T_in_temp , c_co2_0_temp, episl_r_temp, volumetric_flow_temp, Tw_temp]
    init_cond_temp = [T_in_temp, c_co2_0, q_init_cond] * 5
    soln = solve_ivp(deriv1, (t0, tf), init_cond_temp, args=(params_temp,), t_eval = tspan, method = "BDF", rtol = 1e-5, atol = 1e-8) 
    dotT = [soln.y[0], soln.y[3], soln.y[6], soln.y[9], soln.y[12]]
    dotCo2 = [soln.y[1], soln.y[4], soln.y[7], soln.y[10], soln.y[13]]
    dotQ = [soln.y[2], soln.y[5], soln.y[8], soln.y[11], soln.y[14]]

    temp_list = mapWithL(dotT, T_initial)
    co2_array = mapWithL(dotCo2, c_co2_0_slider.value)
    q_array = mapWithL(dotQ, q_init_cond)    

    vec_Z = getVecZ()
    # L = vec_Z[5]
    temp_df = pd.DataFrame(temp_list, tspan)
    co2_df = pd.DataFrame(co2_array, tspan)
    q_df =  pd.DataFrame(q_array, tspan)

# Map data
    source_temperature.data = dict(temp_x=vec_Z, temp_y=temp_df.iloc[1])
    
    # print(temp_df)
    # print(co2_df)
    # print(q_df)
    source_co2.data = dict(co2_x = vec_Z, co2_y = co2_df.iloc[1])
    source_q.data = dict(q_x = vec_Z, q_y = q_df.iloc[1])

for w in [V_slider , T_in_slider, c_co2_0_slider, episl_r_slider, volumetric_flow_slider, Tw_slider]:
    w.on_change('value', update_data)
## --------------------  Animation with Play Button ------------------------- ##
def animate_update():
    # temp = update_animate_helper

    # temp = update_animate_helper.__new__
    # print(temp)
    current_time = slider_time.value + time_step
    if current_time > tf:
        current_time = t0
    vec_Z = getVecZ()
    # temp_df_animate = temp[0]
    # co2_df_animate = temp[1]
    # q_df_animate = temp[2]
    source_temperature.data = dict(temp_x=vec_Z, temp_y=temp_df.loc[current_time])
    source_co2.data = dict(co2_x=vec_Z, co2_y=co2_df.loc[current_time])
    source_q.data = dict(q_x=vec_Z, q_y=q_df.loc[current_time])
    slider_time.value = current_time

    # w.on_change('value', update_animate_helper)

def animate():
    global callback_id
    if animate_button.label == '► Play':

        animate_button.label = '❚❚ Pause'

        callback_id = curdoc().add_periodic_callback(animate_update, 1*450.0) # s to milliseconds conversion
    else:
        animate_button.label = '► Play'
        curdoc().remove_periodic_callback(callback_id)

## --------------------  Reset the values to default ------------------------- ##
def reset():
    source_temperature.data = dict(temp_x=vec_Z, temp_y=temp_df.loc[0])
    source_co2.data = dict(co2_x=vec_Z, co2_y=co2_df.loc[0])
    source_q.data = dict(q_x=vec_Z, q_y=q_df.loc[0])
    slider_time.value = 0.0
    V_slider.value = 3
    T_in_slider.value = 298
    c_co2_0_slider.value = 0.016349 
    episl_r_slider.value = 0.30
    volumetric_flow_slider.value = 0.01
    Tw_slider.value = 293

reset_button = Button(label='Reset', width = 80)
reset_button.on_event('button_click', reset)


#------------------ Start of Reverse Process ----------------------
# desorption: 
# heat temperature 
# have another slider 
# run for 10 seconds - supposed to have small changes
# plot other two plots
# send the results to professor
# all the input parameters, initial conditions, 
# screenshot of simulations for 10 secondes with three graphs

#------------------ Initial Vaalues Revesrse Process  

Tw_temp_desorption = 363.15 # in kelvin = 90 celsius
T_in_desorp= 348.0 # inlet temperature 50 celcius
c_co2_0_desorption = 0.0000000001
volumetric_flow_desorption = 8.333*(10**-5) #     or    5 NL /min  

temperature_reverse_initial_cond= [soln.y[0][24], soln.y[3][24], soln.y[6][24], soln.y[9][24], soln.y[12][24]]
co2_reverse_initial_cond = [soln.y[1][24], soln.y[4][24], soln.y[7][24], soln.y[10][24], soln.y[13][24]]
q_reverse_initial_cond = [soln.y[2][24], soln.y[5][24], soln.y[8][24], soln.y[11][24], soln.y[14][24]]


## ------------- Reverse Process - Desorption ---------------- ##
init_cond_reverse = [temperature_reverse_initial_cond[0], co2_reverse_initial_cond[0], q_reverse_initial_cond[0],
                    temperature_reverse_initial_cond[1], co2_reverse_initial_cond[1], q_reverse_initial_cond[1],
                    temperature_reverse_initial_cond[2], co2_reverse_initial_cond[2], q_reverse_initial_cond[2],
                    temperature_reverse_initial_cond[3], co2_reverse_initial_cond[3], q_reverse_initial_cond[3],
                    temperature_reverse_initial_cond[4], co2_reverse_initial_cond[4], q_reverse_initial_cond[4]]
print(init_cond_reverse)

params_reverse = [V, T_in_desorp, c_co2_0_desorption, episl_r, volumetric_flow_desorption, Tw_temp_desorption]
# # N = 25 # Number of points 
# # tspan = np.linspace(t0, tf, N)

tf_desorb =  7200.0 
tspan_desorb = np.linspace(t0, tf_desorb, N)
time_step_desorb = tspan_desorb[1]
print(tspan_desorb)
print(time_step_desorb)
soln_desorb = solve_ivp(deriv1, (t0, tf), init_cond_reverse, args=(params_reverse,), t_eval = tspan_desorb, method = "BDF", rtol = 1e-5, atol = 1e-19) 

dotT_reverse= [soln_desorb.y[0], soln_desorb.y[3], soln_desorb.y[6], soln_desorb.y[9], soln_desorb.y[12]]
dotCo2_reverse = [soln_desorb.y[1], soln_desorb.y[4], soln_desorb.y[7], soln_desorb.y[10], soln_desorb.y[13]]
dotQ_reverse = [soln_desorb.y[2], soln_desorb.y[5], soln_desorb.y[8], soln_desorb.y[11], soln_desorb.y[14]]

co2_reverse_array = mapWithL(dotCo2_reverse, c_co2_0_slider.value)
co2_reverse_df = pd.DataFrame(co2_reverse_array, tspan_desorb)

source_co2_desorption = ColumnDataSource(data=dict(reverse_x=vec_Z, reverse_y=co2_reverse_df.iloc[1]))
plot_desorption_co2 = figure(height=370, width=400, title="Desorption Process",
              tools= Tools,
              x_range=[0, L], y_range=[0, .03])
plot_desorption_co2.line('reverse_x', 'reverse_y',  line_width=3, source = source_co2_desorption, line_alpha=0.6, color = "red")
plot_desorption_co2.xaxis.axis_label = "L (m)"
plot_desorption_co2.yaxis.axis_label = "Desorption of CO2 (mol/m^3)"

T_reverse_array = mapWithL(dotT_reverse, T_in_desorp)
T_reverse_df = pd.DataFrame(T_reverse_array, tspan_desorb)

source_temperature_reverse = ColumnDataSource(data=dict(temp_x_reverse=vec_Z, temp_y_reverse=temp_df.iloc[1]))
plot_temperature_reverse = figure(height=370, width=400, title="Axial Profile of Column Temperature ",
              tools= Tools,
              x_range=[0, L], y_range=[295, 310])
plot_temperature_reverse.line('temp_x_reverse', 'temp_y_reverse',  line_width=3, source = source_temperature_reverse, line_alpha=0.6, color = "navy")
plot_temperature_reverse.xaxis.axis_label = "L (m)"
plot_temperature_reverse.yaxis.axis_label = "Temperature (K)"

q_reverse_array = mapWithL(dotQ_reverse, q_reverse_initial_cond[4])
q_reverse_df = pd.DataFrame(q_reverse_array, tspan_desorb)
source_q_reverse = ColumnDataSource(data=dict(q_x_reverse=vec_Z, q_y_reverse = q_reverse_df.iloc[1]))
plot_q_reverse = figure(height=370, width=400, title="Axial profile of desorpted CO2",
              tools=Tools,
              x_range=[0, L], y_range=[0, 1.2])
plot_q_reverse.line('q_x_reverse', 'q_y_reverse',  line_width=3, source = source_q_reverse, line_alpha=0.6, color = "navy")
plot_q_reverse.xaxis.axis_label = "L (m)"
plot_q_reverse.yaxis.axis_label = "CO2 Desorbed (mol/kg)"

#--------------------------- Set up Reverse Slider ------------------------------
slider_reverse_time = Slider(title=" Reverse Time Slider (s)", value=t0, start=t0, end=tf_desorb, step=time_step_desorb, width=300)
T_in_desorp_slider = Slider(title="Inlet temperature"+" (default: "+str(T_in)+" K)", value=T_in_desorp, start=285, end=365, step=1)
volumetric_flow_desorption_slider = Slider(title="Inlet flow"+" (default: "+str(volumetric_flow)+")", value=volumetric_flow, start=.001, end=1, step=.005)
Tw_temp_desorption_slider = Slider(title="Water temperature"+" (default: "+str(Tw)+" K)", value=Tw_temp_desorption, start=293, end=365, step=1)

# reverse_process = (column(T_in_desorp_slider, c_co2_0_desorption_slider, volumetric_flow_desorption_slider, Tw_temp_desorption_slider, slider_reverse_time , plot_desorption_co2))

def update_reverse_data(attrname, old, new):

    # Get the current slider values
    V_temp = V_slider.value
    T_in_temp = T_in_desorp_slider.value
    c_co2_0_temp = c_co2_0_desorption_slider.value
    episl_r_temp = episl_r_slider.value
    volumetric_flow_temp = volumetric_flow_desorption_slider.value
    Tw_temp = Tw_temp_desorption_slider.value

    ## --------------------  Update the graphs when changing data ------------------------- ##
    params_temp = [V_temp, T_in_temp , c_co2_0_temp, episl_r_temp, volumetric_flow_temp, Tw_temp]
    # init_cond_temp = [T_in_temp, c_co2_0, q_init_cond] * 5
    soln = solve_ivp(deriv1, (t0, tf), init_cond_reverse, args=(params_temp,), t_eval = tspan_desorb, method = "BDF", rtol = 1e-5, atol = 1e-8) 
    dotReverseCo2 = [soln.y[1], soln.y[4], soln.y[7], soln.y[10], soln.y[13]]
    dotT_reverse = [soln.y[0], soln.y[3], soln.y[6], soln.y[9], soln.y[12]]
    dotQ_reverse = [soln.y[2], soln.y[5], soln.y[8], soln.y[11], soln.y[14]]

# need to fix
    co2_reverse_array = mapWithL(dotReverseCo2, c_co2_0_desorption)
    T_reverse_array = mapWithL(dotT_reverse, T_in_desorp)
    q_reverse_array = mapWithL(dotQ_reverse)

    vec_Z = getVecZ()
    # L = vec_Z[5]
    dotCo2_reverse = pd.DataFrame(co2_reverse_array, tspan_desorb)
    q_reverse_df = pd.DataFrame(q_reverse_array, tspan_desorb)
    T_reverse_df = pd.DataFrame(T_reverse_array, tspan_desorb)
# Map data
    source_co2_desorption.data = dict(reverse_x = vec_Z, reverse_y = dotCo2_reverse.iloc[1])
    source_temperature_reverse.data = dict(temp_x_reverse = vec_Z, temp_y_reverse = T_reverse_df.iloc[1])
    source_q_reverse.data = dict(q_x_reverse = vec_Z, q_y_reverse = q_reverse_df.iloc[1])


for w in [T_in_desorp_slider , volumetric_flow_desorption_slider, Tw_temp_desorption_slider]:
    w.on_change('value', update_reverse_data)


def animate_reverse():
    global callback_id1
    if reverse_animate_button.label == '► Play':

        reverse_animate_button.label = '❚❚ Pause'

        callback_id1 = curdoc().add_periodic_callback(animate_update_reverse, 1*450.0) # s to milliseconds conversion
    else:
        reverse_animate_button.label = '► Play'
        curdoc().remove_periodic_callback(callback_id1)
        
def animate_update_reverse():
    current_time = slider_reverse_time.value + time_step_desorb
    if current_time > tf_desorb:
        current_time = t0
    vec_Z = getVecZ()
    source_co2_desorption.data = dict(reverse_x=vec_Z, reverse_y=co2_reverse_df.loc[current_time])
    source_temperature_reverse.data = dict(temp_x_reverse=vec_Z, temp_y_reverse=T_reverse_df.loc[current_time])
    source_q_reverse.data = dict(q_x_reverse=vec_Z, q_y_reverse=q_reverse_df.loc[current_time])

    # source_temperature_reverse.data = dict(temp_x_reverse = vec_Z, temp_y_reverse = T_reverse_df.iloc[current_time])
    # source_q_reverse.data = dict(q_x_reverse = vec_Z, q_y_reverse = q_reverse_df.iloc[current_time])
    slider_reverse_time.value = current_time

reverse_animate_button = Button(label='► Play', width=80)
reverse_animate_button.on_event('button_click', animate_reverse)

animate_button = Button(label='► Play', width=80)
animate_button.on_event('button_click', animate)

## --------------------  Set up gridplot layout ------------------------- ##
top_page_spacer = Spacer(height = 20)
left_page_spacer = Spacer(width = 20)



constant_slider = (column (V_slider , episl_r_slider))
inputs_reaction = (column(T_in_slider, c_co2_0_slider, volumetric_flow_slider, Tw_slider, slider_time,))
inputs_button = row( animate_button, reset_button)

inputs = column(inputs_reaction, inputs_button)

reverse_slider = (column (slider_reverse_time, T_in_desorp_slider , volumetric_flow_desorption_slider, Tw_temp_desorption_slider))
reverse_button = (row(reverse_animate_button))
reverse_process =(column(reverse_slider, reverse_animate_button, plot_desorption_co2))

column1 = row(left_page_spacer, column(top_page_spacer, constant_slider, inputs, plot_q))
column2 = row(left_page_spacer, column(top_page_spacer, plot_co2, plot_temperature))
column3 = row(left_page_spacer, column(top_page_spacer, reverse_process))
column4 = row(left_page_spacer, column(plot_q_reverse, plot_temperature_reverse))
# grid = gridplot([[column1, column2, column3, column4]])
grid = row(column1, column2, column3, column4)
# grid = gridplot([[constant_slider, inputs, plot_q], [plot_co2, plot_temperature], [reverse_process]])

tab1 =TabPanel(child= grid, title="Direct Air Capture")
# tab1 =TabPanel(child= grid, title="Desktop")
# tab2 =TabPanel(child=column(plot_co2, inputs_button,  row( inputs_reaction, height=450)), title="Phone")
tabs = Tabs(tabs = [tab1])


curdoc().add_root(tabs)
curdoc().title = "Direct Air Capture"
