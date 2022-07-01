#coding:utf-8
import math
from scipy.integrate import solve_ivp, odeint
from bokeh.io import save, curdoc
from bokeh.layouts import column, row
from bokeh.model import Model
from bokeh.models import CustomJS, Slider, Callback, HoverTool
from bokeh.plotting import ColumnDataSource, figure, show
from bokeh.models.widgets import Panel, Tabs
import numpy as np

# three plots , co2 as y and z as x

###############  ---          Static Parameters        ---  ###############
b0 = 93.0 * (10 ** (-5))
deltH_0 = 95.3  # calculate b
Tw = -5.0  # room temperature
T0 = 353.15  # temeperature
t_h0 = .37  # heterogeneity constant, in paper is denoted as t_h0
apha = 0.33
chi = 0.0
q_s0 = 3.40 # qs_var = q_s0 = 3.4 due to chi = 0
R = 8.314
kT = 3.5 * (10 ** 3)  # calculate rA
ps = 880.0
deltH_co2 = 75.0  # calculate temeprature change

# ------------------ For Equation 4 : Enegergy Ballance  --------------
pg = 1.87  # ?
h = 13.8
Cp_g = 37.55  # J/molK
Cp_s = 1580.0  # J/molK

# ODE Part
# Repetitive shortcut
def ener_balan(v0, theta, deltZ):  # replace v0  * pg* Cp_g / (theta * deltZ)
    ener_balan_part1 = v0 * pg * Cp_g
    
    return (ener_balan_part1 / (theta * deltZ))

def ener_balan2(episl_r):
    ener_balan2 = (1 - episl_r) * ps * deltH_co2
    return (ener_balan2)

def ener_balan3(a_s, Tn):
    ener_balan3 = a_s * h * (Tw - Tn)
    # print(ener_balan3)
    return (ener_balan3)

# Equation 1 Mass Balance : find co2_n

def mass_balan(v0, episl_r, deltZ):
    mass_balan = v0 / (episl_r * deltZ)
    # print(mass_balan)
    return (mass_balan)

def masss_balan2(episl_r, ps):
    mass_balan2 = (1 - episl_r) * ps
    # print(mass_balan2)
    return (mass_balan2)
# Equations are calclulated in order
def b(T):
    # print(T)
    b = b0 *2.71**((deltH_0 / (R * T0)) * (T0 / T - 1))
    # print(b)
    return b

def t_h(T):
    # print(T)
    # print(f'T, {T}')
    # T0 = 353
    
    t_h = t_h0 + (apha * (1 - (T0 / T)) )
    # t_h = .37 + ( .33 * (1 - (353.15 / T)) )
    # print(t_h)
    return (t_h)


def q_s(T):
    q_s = q_s0 *2.71**(chi * (1 - T / T0))
    # print(q_s)
    return (q_s)
    # q_s = math(q_s0, (chi * (1 - T / T0)))
   
    # return (q_s)


# Calculate rco2_n (not ode)
# change it to q
def R_co2(T, c_co2, q, b_var, t_var):
    # b_var = b(T)
    # t_var = t_h(T)
    # # print(t_var)
    # print(a)
    # print(q)
    # print(c_co2)
    rco2_term1 = R* T * c_co2 
    # print(f"R * T * c_co2, {R * T * c_co2}.  ")
    # print(f'((1 - ((q / q_s0) ** (t_var))) ** (1 / t_var)), {((1 - ((q / q_s0) ** (t_var))) ** (1 / t_var))}')
    # print(f'R * T * c_co2 * ((1 - ((q / q_s0) ** (t_var))) ** (1 / t_var)), {R * T * c_co2 * ((1 - ((q / q_s0) ** (t_var))) ** (1 / t_var))}')
    
    rco2_term2 = 1 - ((q / 3.4) ** (1/t_var))
    rco2_term3 = (rco2_term1  * rco2_term2) ** (1/t_var)
    # rco2_term4 = rco2_term3 - q / (b_var * q_s0)
    # rco2_term5 = kT * rco2_term4
    # term = kT * (R * T * c_co2 * ((1 - ((q / q_s0) ** (t_var))) ** (1 / t_var)) - q / (b_var * q_s0))
    # print(f'term, {term}')
    # print(f"rco2_term1, {rco2_term1}.  ")
    # print(f"rco2_term2, {rco2_term2}. ")
    # print(f"(rco2_term3), {rco2_term3}. ")
    # print(f"(rco2_term4), {rco2_term4}. ")
    r_co2 = kT * (R * T * c_co2 * ((1 - ((q / q_s0) ** (t_var))) ** (1 / t_var)) - q / (b_var * q_s0))
    # print(f"r_co2",{r_co2})

    # print(f"b_var * qs_var, {rco2_term5}.")
    return r_co2


def deriv(t, y, params):
    T_n, co2_n, q_n, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 = y # the rest of 12 vars a_n are not used, only for the success of solve_ivp   , 
    V, r, T, c_co2_0, episl_r, v0 = params
    ###############   -----  Parameters depend on input  -----  ###############
    L = V / (math.pi * (r ** 2))
    deltZ = L / 5.0  # 5 boxes in total
    a_s = deltZ / r
    theta = (1 - episl_r) * ps * Cp_s + episl_r * pg * Cp_g

    b_var = b(T)
    t_var = t_h(T)
   # T_n, co2_n, q_n, T_n2, co2_n2, q_n2, T_n3, co2_n3, q_n3, T_n4, co2_n4, q_n4, T_n5, co2_n5, q_n5 == y
    # rco2_ first, rate of generation
    T1 = -ener_balan(v0, theta, deltZ) * T_n + ener_balan(v0, theta, deltZ) * T0 + ener_balan2(episl_r) * (
        R_co2(T_n, co2_n, q_n, b_var, t_var)) + ener_balan3(a_s, T_n)
    # print(T1)
    co2_1 = -mass_balan(v0, episl_r, deltZ) * co2_n + mass_balan(v0, episl_r, deltZ) * c_co2_0 - (
        R_co2(T_n, co2_n, q_n,  b_var, t_var)) * masss_balan2(episl_r, ps)
    q_1 = R_co2(T_n, co2_n, q_n,  b_var, t_var)

    T2 = -ener_balan(v0, theta, deltZ) * T_n + ener_balan(v0, theta, deltZ) * T1 + ener_balan2(episl_r) * (
        R_co2(T_n, co2_n, q_n, b_var, t_var)) + ener_balan3(a_s,  T_n)
    co2_2 = -mass_balan(v0,episl_r, deltZ) * co2_n + mass_balan(v0,episl_r, deltZ) * co2_1 - (
        R_co2(T_n, co2_n, q_n, b_var, t_var)) * masss_balan2(episl_r, ps)
    q_2 = R_co2(T_n, co2_n, q_n, b_var, t_var)

    T3 = -ener_balan(v0, theta, deltZ) * T_n + ener_balan(v0, theta, deltZ) * T2 + ener_balan2(episl_r) * (
        R_co2(T_n, co2_n, q_n, b_var, t_var)) + ener_balan3(a_s, T_n)
    co2_3 = -mass_balan(v0,episl_r, deltZ) * co2_n + mass_balan(v0,episl_r, deltZ) * co2_2 - (
        R_co2(T_n, co2_n, q_n, b_var, t_var)) * masss_balan2(episl_r, ps)
    q_3 = R_co2(T_n, co2_n, q_n, b_var, t_var)

    T4 = -ener_balan(v0, theta, deltZ) * T_n + ener_balan(v0, theta, deltZ) * T3 + ener_balan2(episl_r) * (
        R_co2(T_n, co2_n, q_n, b_var, t_var)) + ener_balan3(a_s,  T_n)
    co2_4 = -mass_balan(v0,episl_r, deltZ) * co2_n + mass_balan(v0,episl_r, deltZ) * co2_3 - (
        R_co2(T_n, co2_n, q_n,b_var, t_var)) * masss_balan2(episl_r, ps)
    q_4 = R_co2(T_n, co2_n, q_n, b_var, t_var)

    T5 = -ener_balan(v0, theta, deltZ) * T_n + ener_balan(v0, theta, deltZ) * T4 + ener_balan2(episl_r) * (
        R_co2(T_n, co2_n, q_n, b_var, t_var)) + ener_balan3(a_s,  T_n)
    co2_5 = -mass_balan(v0,episl_r, deltZ) * co2_n + mass_balan(v0,episl_r, deltZ) * co2_4 - (
        R_co2(T_n, co2_n, q_n, b_var, t_var)) * masss_balan2(episl_r, ps)
    q_5 = R_co2(T_n, co2_n, q_n, b_var, t_var)

    # result = np.array([T1, T2, T3, T4, T5, co2_1, co2_2, co2_3, co2_4, co2_5, q_1, q_2, q_3, q_4, q_5]).reshape(-1, 1)

    return [T1, co2_1, q_1, T2, co2_2, q_2, T3, co2_3, q_3, T4, co2_4, q_4, T5, co2_5, q_5 ]
    #  

def deriv1(t, y, params):
    T_n, co2_n, q_n, T_n2, co2_n2, q_n2,T_n3, co2_n3, q_n3, T_n4, co2_n4, q_n4,T_n5, co2_n5, q_n5 = y # the rest of 12 vars a_n are not used, only for the success of solve_ivp
    V, r, T, c_co2_0, episl_r, v0 = params
    ###############   -----  Parameters depend on input  -----  ###############
    L = V / (math.pi * (r ** 2))
    deltZ = L / 5.0  # 5 boxes in total
    a_s = deltZ / r
    theta = (1 - episl_r) * ps * Cp_s + episl_r * pg * Cp_g

    b_var = b(T)
    t_var = t_h(T)
    # print(t_var)

   # T_n, co2_n, q_n, T_n2, co2_n2, q_n2, T_n3, co2_n3, q_n3, T_n4, co2_n4, q_n4, T_n5, co2_n5, q_n5 == y
    # rco2_ first, rate of generation
    T1 = -ener_balan(v0, theta, deltZ) * T_n + ener_balan(v0, theta, deltZ) * T0 + ener_balan2(episl_r) * (
        R_co2(T_n, co2_n, q_n,  b_var, t_var)) + ener_balan3(a_s, T_n)
    co2_1 = -mass_balan(v0, episl_r, deltZ) * co2_n + mass_balan(v0, episl_r, deltZ) * c_co2_0 - (
        R_co2(T_n, co2_n, q_n,  b_var, t_var)) * masss_balan2(episl_r, ps)
    q_1 = R_co2(T_n, co2_n, q_n,  b_var, t_var)

    T2 = -ener_balan(v0, theta, deltZ) * T_n2 + ener_balan(v0, theta, deltZ) * T1 + ener_balan2(episl_r) * (
        R_co2(T_n2, co2_n2, q_n2,  b_var, t_var)) + ener_balan3(a_s,  T_n2)
    co2_2 = -mass_balan(v0, episl_r, deltZ) * co2_n2 + mass_balan(v0, episl_r, deltZ) * co2_1 - (
        R_co2(T_n2, co2_n2, q_n2,  b_var, t_var)) * masss_balan2(episl_r, ps)
    q_2 = R_co2(T_n2, co2_n2, q_n2,  b_var, t_var)

    T3 = -ener_balan(v0, theta, deltZ) * T_n3 + ener_balan(v0, theta, deltZ) * T2 + ener_balan2(episl_r) * (
        R_co2(T_n3, co2_n3, q_n3,  b_var, t_var)) + ener_balan3(a_s, T_n3)
    co2_3 = -mass_balan(v0, episl_r, deltZ) * co2_n3 + mass_balan(v0, episl_r, deltZ) * co2_2 - (
        R_co2(T_n3, co2_n3, q_n3,  b_var, t_var)) * masss_balan2(episl_r, ps)
    q_3 = R_co2(T_n3, co2_n3, q_n3,  b_var, t_var)

    T4 = -ener_balan(v0, theta, deltZ) * T_n4 + ener_balan(v0, theta, deltZ) * T3 + ener_balan2(episl_r) * (
        R_co2(T_n4, co2_n4, q_n4,  b_var, t_var)) + ener_balan3(a_s,  T_n4)
    co2_4 = -mass_balan(v0, episl_r, deltZ) * co2_n4 + mass_balan(v0, episl_r, deltZ) * co2_3 - (
        R_co2(T_n4, co2_n4, q_n4,  b_var, t_var)) * masss_balan2(episl_r, ps)
    q_4 = R_co2(T_n4, co2_n4, q_n4,  b_var, t_var)

    T5 = -ener_balan(v0, theta, deltZ) * T_n5 + ener_balan(v0, theta, deltZ) * T4 + ener_balan2(episl_r) * (
        R_co2(T_n5, co2_n5, q_n5,  b_var, t_var)) + ener_balan3(a_s,  T_n5)
    co2_5 = -mass_balan(v0, episl_r, deltZ) * co2_n5 + mass_balan(v0, episl_r, deltZ) * co2_4 - (
        R_co2(T_n5, co2_n5, q_n5,  b_var, t_var)) * masss_balan2(episl_r, ps)
    q_5 = R_co2(T_n5, co2_n5, q_n5,  b_var, t_var)

    # result = np.array([T1, T2, T3, T4, T5, co2_1, co2_2, co2_3, co2_4, co2_5, q_1, q_2, q_3, q_4, q_5]).reshape(-1, 1)

    return [T1, co2_1, q_1, T2, co2_2, q_2, T3, co2_3, q_3, T4, co2_4, q_4, T5, co2_5, q_5]


###############    User generated - Slider initial value   ###############
V = 100.0  # volume
r = 5.0
T = 293.0 # +273
c_co2_0 = 5.0  # concentration
episl_r = 0.3  # void
v0 = 2.0  # initial vilocity

t0, tf = 0.0, 10.0
############# initial condition
# init_cond = [20.000, 0.000, 0.000]
init_cond = [20.000, 0.000, 0.000,20.000, 0.000, 0.000,20.000, 0.000, 0.000,20.000, 0.000, 0.000,20.000, 0.000, 0.000]
# ,20.000, 0.000, 0.000,20.000, 0.000, 0.000,20.000, 0.000, 0.000,20.000, 0.000, 0.000
# t_span = np.linspace(t0, tf, N)
# soln = odeint(deriv, init_cond, t_span)
params = [V, r, T, c_co2_0, episl_r, v0]
soln = solve_ivp(deriv, (t0, tf), init_cond, args=(params,))  # init_cond = (T, c_co2_0, q0)
print(soln)
# Equation 3
# dq/dt = r_co2

#  Graph co2_n, T_n, and rco2_n
# T1= soln.y[0]
# co2_1 = soln.y[1]
# q_1 = soln.y[2]

# T2= soln.y[3]
# co2_2 = soln.y[4]
# q_2 = soln.y[5]

# T3= soln.y[6]
# co2_3 = soln.y[7]
# q_3 = soln.y[8]

# T4= soln.y[9]
# co2_4 = soln.y[10]
# q_4 = soln.y[11]

# T5= soln.y[12]
# co2_5 = soln.y[13]
# q_5 = soln.y[14]

# temperature_name = [T1, T2, T3, T4, T5]
# print(temperature_name)

# vec_Z = np.linspace(0,  V / (math.pi * (r ** 2)), 12)
# temperature_name = [T1, T2, T3, T4, T5]
# temperature_colors = ['lightsteelblue','deepskyblue', 'dodgerblue', 'mediumblue', 'navy']
# co2_name = [co2_1, co2_2, co2_3, co2_4, co2_5]
# co2_colors = ['darkseagreen','mediumseagreen', 'seagreen', 'green', 'drakolivegreen']
# q_name = [q_1, q_2, q_3, q_4, q_5]
# q_colors = ['mediumpurple', 'blueviolet', 'darkorchid', 'indigo', 'mediumslateblue']
# source_temp0 = ColumnDataSource(data=dict(vec_Z = vec_Z, T1=T1, T2 = T2, T3 = T3, T4=T4, T5=T5))
# source_tempVbar = ColumnDataSource(data=dict( temeperature_name=temperature_name, color=temperature_colors))
# # vbar_top = [20, 0, 0]
# # source_vbar1 = ColumnDataSource(data=dict(vbar_top=vbar_top))

# source_temp0 = ColumnDataSource(data=dict(vec_Z = vec_Z, T1=T1, T2 = T2, T3 = T3, T4=T4, T5=T5))
# source_tempVbar = ColumnDataSource(data=dict( temeperature_name=temperature_name, color=temperature_colors))

# TOOLTIPS = [("Z","@vec_Z"), ("T1","@T1{0,0.000}"), ("T2","@T2{0,0.000}"), ("T3","@T3{0,0.000}"), ("T4","@T4{0,0.000}"), ("T5","@T5{0,0.000}")]
# TOOLS = "pan,undo,redo,reset,save,wheel_zoom,box_zoom"
# plot_conc = figure(plot_height=450, plot_width=550, tools=TOOLS, tooltips=TOOLTIPS,
#               title="Direct Air Capture", x_range=[t0, tf], y_range=[-30, 20])
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

# plot_conc.xaxis.axis_label = "Z"
# plot_conc.yaxis.axis_label = "Temeprarture"
# plot_conc.legend.location = "top_left"
# plot_conc.legend.click_policy="hide"
# plot_conc.legend.background_fill_alpha = 0.5
# plot_conc.grid.grid_line_color = "silver"

# V_slider = Slider(title="Volume of bed"+" (initial: "+str(V)+")", value=V, start=90, end=150, step=5)
# r_slider = Slider(title="Radius of bed"+" (initial: "+str(r)+")", value=r, start=1, end=7, step=0.02)
# T_slider = Slider(title="initial temperature"+" (initial: "+str(T)+")", value=T, start=-2, end=40, step=1)
# c_co2_0_slider = Slider(title="initial CO2 concentration"+" (initial: "+str(c_co2_0)+")", value=c_co2_0, start=0, end=5, step=0.2)
# episl_r_slider = Slider(title="Episl r"+" (initial: "+str(episl_r)+")", value=episl_r, start=1, end=5, step=1)
# v0_slider = Slider(title="Initial Velocity"+" (initial: "+str(v0)+")", value=v0, start=1, end=5, step=1)

# def update_data(attrname, old, new):

#     # Get the current slider values
#     V_temp= V_slider.value
#     r_temp = r_slider.value
#     T_temp = T_slider.value
#     c_co2_0_temp = c_co2_0_slider.value
#     episl_r_temp = episl_r_slider.value
#     v0_temp = v0_slider.value
#     # Generate the new curve
#     vec_time = np.linspace(t0, tf, 12)  # vector for time
#     params = [V_temp, r_temp, T_temp, c_co2_0_temp, episl_r_temp, v0_temp]
#     soln = solve_ivp(deriv, (t0, tf), init_cond, args=(params,)) 
#     T1_temp = soln.y[0]
#     T2_temp = soln.y[3]
#     T3_temp = soln.y[6]
#     T4_temp = soln.y[9]
#     T5_temp = soln.y[12]
#     source_temp0.data =  dict(vec_time=vec_time,   T1_temp=T1_temp, T2_temp = T2_temp, T3_temp = T3_temp, T4_temp=T4_temp, T5_temp=T5_temp)
#     # vbar_top_temp = [np.interp(time_temp, vec_time, int_vec_A), np.interp(time_temp, vec_time, int_vec_B),
#     #                  np.interp(time_temp, vec_time, int_vec_C)]
#     temperature_nameTemp = [T1_temp, T2_temp, T3_temp, T4_temp, T5_temp]
#     temperature_colorsTemp = ['lightsteelblue','deepskyblue', 'dodgerblue', 'mediumblue', 'navy']
#     source_tempVbar.data = dict(temperature_nameTemp=temperature_nameTemp, temperature_colorsTemp=temperature_colorsTemp)

# for w in [V_slider , r_slider, T_slider, c_co2_0_slider, episl_r_slider, v0_slider]:
#     w.on_change('value', update_data)


# inputs_reaction = column(V_slider , r_slider, T_slider, c_co2_0_slider, episl_r_slider, v0_slider)
# tab =Panel(child = row(inputs_reaction, plot_conc))
# tab1 = Panel(child = column(inputs_reaction, plot_conc))
# tabs = Tabs(tabs = [tab, tab1])
# curdoc().add_root(tabs)
# curdoc().title = "Direct Air Capture"
# # tabs = Tabs(tabs=[ tab1, tab ])
# # show(tabs)


