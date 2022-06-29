#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import math

from bokeh.io import save, curdoc
from bokeh.layouts import column, row
from bokeh.model import Model
from bokeh.models import CustomJS, Slider, Callback, Panel
from bokeh.plotting import ColumnDataSource, figure, show
import numpy as np
from scipy.integrate import solve_ivp, odeint
# three plots , co2 as y and z as x

###############    User generated - Slider initial value   ###############
# V = 100.0  # volume
# r = 5.0
# T = 20.0
# c_co2_0 = 5.0  # concentration
# episl_r = 0.3  # void
# v0 = 2.0  # initial vilocity
# N = 5.0 #number of boxes

###############  ---          Static Parameters        ---  ###############
b0 = 93.0 * (10 ** (-5))
deltH_0 = 95.3  # calculate b
Tw = -5.0  # room temperature
T0 = 353.15  # temeperature
t0 = .37  # heterogeneity constant, in paper is denoted as t_h0
alpha = 0.33
chi = 0.0
q_s0 = 3.40
R = 8.314
kT = 3.5 * (10 ** 3)  # calculate rA
ps = 880.0
deltH_co2 = 75.0  # calculate temeprature change
pg = 1.87  # 
h = 13.8
Cp_g = 37.55  # J/molK
Cp_s = 1580.0  # J/molK

###############   -----  Parameters depend on input  -----  ###############
def find_L(V, r):
    return( V / (math.pi * (r ** 2)) )
#L = V / (math.pi * r ** 2)

def find_Z(L, N):
    return(L/N)
#deltZ = L / 5.0  # 5 boxes in total

def find_aS(deltZ, r):
    return (deltZ / r)
#a_s = deltZ / r

def find_theta(episl_r):
    return ((1 - episl_r) * ps * Cp_s + episl_r * pg * Cp_g)
#theta = (1 - episl_r) * ps * Cp_s + episl_r * pg * Cp_g


# V = 100.0  # volume
# r = 5.0
# T = 20.0
# c_co2_0 = 5.0  # concentration
# episl_r = 0.3  # void
# v0 = 2.0  # initial vilocity

# Equations are calclulated in order
def b(T):
    b = b0 * math.exp((deltH_0 / (R * T0)) * (T0 / T - 1))
    return b

def t_h(T):
    return (t0 + alpha * (1 - T0 / T))

def q_s(T):
    return (q_s0 *  math.exp(chi * (1 - T / T0)))

# Calculate rco2_n (not ode)
# change it to q
def R_co2(T, c_co2, q):
    b_var = b(T)
    t_var = t_h(T)
    qs_var = q_s(T)
    # print(qs_var)
    r_co2 = kT * (R * T * c_co2 * ((1 - ((q / qs_var) ** abs(t_var))) ** (1 / t_var)) - q / (b_var * qs_var))
    # kT * (R * T * c_co2 * ((1 - ((q / qs_var) ** t_var)) ** (1 / t_var)) - q / (b_var * qs_var))
    # print(r_co2)
    return r_co2


# ODE Part
# Repetitive shortcut

# Equation 2
def ener_balan_part1(v0):
    term = v0 * pg * Cp_g
    return (term)


def ener_balan(episl_r, V, r, v0): 
    L = find_L(V, r)
    theta = find_theta(episl_r) 
    deltZ = find_Z(L, N)
    term = ener_balan_part1(v0)
    return (term /  (theta * deltZ))# replace v0  * pg* Cp_g / (theta * deltZ)


def ener_balan2(episl_r):
    return ((1 - episl_r) * ps * deltH_co2)


def ener_balan3(V, r):
    L = find_L(V, r)
    deltZ = find_Z(L, N)
    a_s = find_aS(deltZ, r)
    return (a_s * h * (Tw - T0))


# Equation 1 Mass Balance : find co2_n

def mass_balan(episl_r, V, r, v0):
    L = find_L(V, r)
    deltZ = find_Z(L, N)
    return (v0 / (episl_r * deltZ))


def masss_balan2(episl_r, ps):
    return ((1 - episl_r) * ps)

def deriv(t, y, params):
    T_n, co2_n, q_n, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 = y # the rest of 12 vars a_n are not used, only for the success of solve_ivp
    episl_r, V, r, v0, c_co2_0, T = params
    # T_n, co2_n, q_n, T_n2, co2_n2, q_n2, T_n3, co2_n3, q_n3, T_n4, co2_n4, q_n4, T_n5, co2_n5, q_n5 == y
    # rco2_ first, rate of generation
    T1 = -ener_balan(episl_r, V, r, v0) * T_n + ener_balan(episl_r, V, r, v0) * T0 + ener_balan2(episl_r) * (
        R_co2(T_n, co2_n, q_n)) + ener_balan3(V, r)
    co2_1 = -mass_balan(episl_r, V, r, v0) * co2_n + mass_balan(episl_r, V, r, v0) * c_co2_0 - (
        R_co2(T_n, co2_n, q_n)) * masss_balan2(episl_r, ps)
    q_1 = R_co2(T_n, co2_n, q_n)

    T2 = -ener_balan(episl_r, V, r, v0) * T_n + ener_balan(episl_r, V, r, v0) * T1 + ener_balan2(episl_r) * (
        R_co2(T_n, co2_n, q_n)) + ener_balan3(V, r)
    co2_2 = -mass_balan(episl_r, V, r, v0) * co2_n + mass_balan(episl_r, V, r, v0) * co2_1 - (
        R_co2(T_n, co2_n, q_n)) * masss_balan2(episl_r, ps)
    q_2 = R_co2(T_n, co2_n, q_n)

    T3 = -ener_balan(episl_r, V, r, v0) * T_n + ener_balan(episl_r, V, r, v0) * T2 + ener_balan2(episl_r) * (
        R_co2(T_n, co2_n, q_n)) + ener_balan3(V, r)
    co2_3 = -mass_balan(episl_r, V, r, v0) * co2_n + mass_balan(episl_r, V, r, v0) * co2_2 - (
        R_co2(T_n, co2_n, q_n)) * masss_balan2(episl_r, ps)
    q_3 = R_co2(T_n, co2_n, q_n)

    T4 = -ener_balan(episl_r, V, r, v0) * T_n + ener_balan(episl_r, V, r, v0) * T3 + ener_balan2(episl_r) * (
        R_co2(T_n, co2_n, q_n)) + ener_balan3(V, r)
    co2_4 = -mass_balan(episl_r, V, r, v0) * co2_n + mass_balan(episl_r, V, r, v0) * co2_3 - (
        R_co2(T_n, co2_n, q_n)) * masss_balan2(episl_r, ps)
    q_4 = R_co2(T_n, co2_n, q_n)

    T5 = -ener_balan(episl_r, V, r, v0) * T_n + ener_balan(episl_r, V, r, v0) * T4 + ener_balan2(episl_r) * (
        R_co2(T_n, co2_n, q_n)) + ener_balan3(V, r)
    co2_5 = -mass_balan(episl_r, V, r, v0) * co2_n + mass_balan(episl_r, V, r, v0) * co2_4 - (
        R_co2(T_n, co2_n, q_n)) * masss_balan2(episl_r, ps)
    q_5 = R_co2(T_n, co2_n, q_n)

    # result = np.array([T1, T2, T3, T4, T5, co2_1, co2_2, co2_3, co2_4, co2_5, q_1, q_2, q_3, q_4, q_5]).reshape(-1, 1)

    return [T1, co2_1, q_1, T2, co2_2, q_2, T3, co2_3, q_3, T4, co2_4, q_4, T5, co2_5, q_5]

V = 100.0  # volume
r = 5.0
T = 20.0
c_co2_0 = 5.0  # concentration
episl_r = 0.3  # void
v0 = 2.0  # initial vilocity
N = 5.0 #number of boxes

t0, tf = 0.0, 10.0
############# initial condition
T_initial = 20
c_co2_0 = 0
q0 = 0
init_cond = [20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
num = 5
t_span = np.linspace(t0, tf, num)
params = [episl_r, V, r, v0, c_co2_0, T] 
# soln = odeint(deriv, init_cond, t_span)
soln = solve_ivp(deriv, (t0, tf), init_cond, args=(params,))  # init_cond = (T, c_co2_0, q0)
print(soln)

#  Graph co2_n, T_n, and rco2_n
T1= soln.y[0]
co2_1 = soln.y[1]
q_1 = soln.y[2]

T2= soln.y[3]
co2_2 = soln.y[4]
q_2 = soln.y[5]

T3= soln.y[6]
co2_3 = soln.y[7]
q_3 = soln.y[8]

T4= soln.y[9]
co2_4 = soln.y[10]
q_4 = soln.y[11]

T5= soln.y[12]
co2_5 = soln.y[13]
q_5 = soln.y[14]
L = find_L(V, r)
vec_Z = np.linspace(0, L, 5)
temperature_name = [T1, T2, T3, T4, T5]
temperature_colors = ['lightsteelblue','deepskyblue', 'dodgerblue', 'mediumblue', 'navy']
co2_name = [co2_1, co2_2, co2_3, co2_4, co2_5]
co2_colors = ['darkseagreen','mediumseagreen', 'seagreen', 'green', 'drakolivegreen']
q_name = [q_1, q_2, q_3, q_4, q_5]
q_colors = ['mediumpurple', 'blueviolet', 'darkorchid', 'indigo', 'mediumslateblue']
source_temp0 = ColumnDataSource(data=dict(vec_Z = vec_Z, T1=T1, T2 = T2, T3 = T3, T4=T4, T5=T5))
source_vbar = ColumnDataSource(data=dict( temeperature_name=temperature_name, color=temperature_colors))
vbar_top = [20, 0, 0]
source_vbar1 = ColumnDataSource(data=dict(vbar_top=vbar_top))



TOOLTIPS = [("vec_Z Z","@vec_Z"), ("T1","@T1{0,0.000}"), ("T2","@T2{0,0.000}"), ("T3","@T3{0,0.000}"), ("T4","@T4{0,0.000}"), ("T5","@T5{0,0.000}")]
TOOLS = "pan,undo,redo,reset,save,wheel_zoom,box_zoom"
plot_conc = figure(plot_height=450, plot_width=550, tools=TOOLS, tooltips=TOOLTIPS,
              title="Direct Air Capture", x_range=[t0, tf], y_range=[0, 15])
plot_conc.line('vec_Z', 'T1', source=source_temp0, line_width=3, line_alpha=0.6, line_color='lightsteelblue',
               legend_label="T1")
plot_conc.line('vec_Z', 'T2', source=source_temp0, line_width=3, line_alpha=0.6, line_color='deepskyblue',
               legend_label="T2")
plot_conc.line('vec_Z', 'T3', source=source_temp0, line_width=3, line_alpha=0.6, line_color='dodgerblue',
               legend_label="T3")
plot_conc.line('vec_Z', 'T4', source=source_temp0, line_width=3, line_alpha=0.6, line_color='mediumblue',
               legend_label="T4")
plot_conc.line('vec_Z', 'T5', source=source_temp0, line_width=3, line_alpha=0.6, line_color='navy',
               legend_label="T5")

plot_conc.xaxis.axis_label = "Z"
plot_conc.yaxis.axis_label = "Temeprarture"
plot_conc.legend.location = "top_left"
plot_conc.legend.click_policy="hide"
plot_conc.legend.background_fill_alpha = 0.5
plot_conc.grid.grid_line_color = "silver"

V_slider = Slider(title="Volume of bed"+" (initial: "+str(V)+")", value=V, start=2.02, end=8.0, step=0.02)
r_slider = Slider(title="Radius of bed"+" (initial: "+str(r)+")", value=r, start=0.02, end=2.0, step=0.02)
T_slider = Slider(title="initial temperature"+" (initial: "+str(T)+")", value=T, start=1, end=25, step=1)
c_co2_0_slider = Slider(title="initial CO2 concentration"+" (initial: "+str(c_co2_0)+")", value=c_co2_0, start=1, end=5, step=1)
episl_r_slider = Slider(title=""+" (initial: "+str(episl_r)+")", value=episl_r, start=1, end=5, step=1)
v0_slider = Slider(title="Initial Velocity"+" (initial: "+str(v0)+")", value=v0, start=1, end=5, step=1)

def update_data(attrname, old, new):

    # Get the current slider values
    V_temp= V_slider.value
    r_temp = r_slider.value
    T_temp = T_slider.value
    c_co2_0_temp = c_co2_0_slider.value
    episl_r_temp = episl_r_slider.value
    v0_temp = v0_slider.value
    # Generate the new curve
    params_temp  = [episl_r_temp, V_temp, r_temp, v0_temp, c_co2_0_temp]
    vec_time = np.linspace(t0, tf, N)  # vector for time
    soln = solve_ivp(deriv, (t0, tf), init_cond, args=(params_temp,)) 
    T1_temp = soln.y[0]
    T2_temp = soln.y[3]
    T3_temp = soln.y[6]
    T4_temp = soln.y[9]
    T5_temp = soln.y[12]
    source_temp0.data =  dict(vec_time=vec_time,   T1_temp=T1_temp, T2_temp = T2_temp, T3_temp = T3_temp, T4_temp=T4_temp, T5_temp=T5_temp)
    # vbar_top_temp = [np.interp(time_temp, vec_time, int_vec_A), np.interp(time_temp, vec_time, int_vec_B),
    #                  np.interp(time_temp, vec_time, int_vec_C)]
    temperature_nameTemp = [T1_temp, T2_temp, T3_temp, T4_temp, T5_temp]
    temperature_colorsTemp = ['lightsteelblue','deepskyblue', 'dodgerblue', 'mediumblue', 'navy']
    source_vbar.data = dict(temperature_nameTemp=temperature_nameTemp, temperature_colorsTemp=temperature_colorsTemp)

for w in [V_slider , r_slider, T_slider, c_co2_0_slider, episl_r_slider, v0_slider]:
    w.on_change('value', update_data)


inputs_reaction = column(V_slider , r_slider, T_slider, c_co2_0_slider, episl_r_slider, v0_slider)
tab =Panel(child=row(inputs_reaction, plot_conc, column( height=450)))

curdoc().add_root(tab)
curdoc().title = "Direct Air Capture"



