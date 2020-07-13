#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:30:12 2020

@author: annamoragne
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Slider, Select, Paragraph, TableColumn, DataTable, Button, Panel, Tabs, LinearAxis, Range1d
from bokeh.plotting import figure, show
TOOLS = "pan,undo,redo,reset,save,box_zoom,tap"

# Total population, N.
N = 1000 #starting total population
# Initial number of infected and recovered individuals, I0 and R0.
Is0 = 1 #initial number of symptomatic infected individuals
Ia0=1 #initial number of asymptomatic infected individuals
R0 = 0 #intial number of recovered individuals
D0=0 # initial number of dead individuls
E0=0 # initial number of people exposed to the virus but not yet transmitable
# Everyone else, S0, is susceptible to infection initially.
S0 = N - Is0-Ia0 - R0-D0-E0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta_S=0.1 #the contact/infection rate per day for symptomatic infecteds
beta_A=0.2 #the contact/infection rate per day for asymptomatic infecteds
gamma = 1/14  #the recovery rate per day
death_rate_S=0.006 #death rate for symptomatic infecteds
nat_death=0.0002
nat_birth=0.001
E_to_I_forS=0.4 #rate at which individuals from the exposed class become symptomatic infecteds
E_to_I_forA=0.1 #rate at which as individuals from the exposed class become asymptomatic infecteds
return_rate=0.002 #rate at which recovered people once again become susceptible
# A grid of time points (in days)
t = np.linspace(0, 160, 160) #160 days

# The SIR model differential equations.
def deriv(y, t, N, beta_A, beta_S, gamma, nat_death, death_rate_S, E_to_I_forA, E_to_I_forS, return_rate):
    S, E, Ia, Is, R, D = y
    dSdt = (-beta_S * S * Is / N)-(beta_A * S * Ia / N)-(nat_death*S)+(nat_birth*(N-D))+(return_rate*R)
    dEdt = (beta_S * S * Is / N)+(beta_A * S * Ia / N) - (E_to_I_forA * E)-(E_to_I_forS*E)-(nat_death*E)
    dIadt= (E_to_I_forA*E)-(nat_death*Ia)-(gamma*Ia)
    dIsdt = (E_to_I_forS*E)-(nat_death*Is)-(death_rate_S*Is)-(gamma*Is)
    dRdt = (gamma * (Ia+Is))-(nat_death*R)-(return_rate*R)
    dDdt=nat_death*(S+E+Ia+Is+R)+(death_rate_S*Is)
    return dSdt, dEdt, dIadt, dIsdt, dRdt, dDdt

# Initial conditions vector
y0 = S0, E0, Ia0, Is0, R0, D0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta_A, beta_S,  gamma, nat_death, death_rate_S, E_to_I_forA, E_to_I_forS, return_rate))
S, E, Ia, Is, R, D = ret.T
sourcePops=ColumnDataSource(data=dict(time=t, S=S/1000, E=E/1000, Ia=Ia/1000, Is=Is/1000, R=R/1000, D=D/1000))

pops=figure(title="SEIR Model Class Populations", x_axis_label="Time (in days)", y_axis_label="Proportion of people in each class", tools=TOOLS)
pops.title.text_font_size='15pt'
pops.line('time', 'S', source=sourcePops, legend_label="Susceptible", line_width=2, color='darkblue')
pops.line('time', 'E', source=sourcePops, legend_label="Exposed", line_width=2, color='lightblue')
pops.line('time', 'Ia', source=sourcePops, legend_label="Asymptomatic Infected", line_width=2, color='red', line_dash=[4,4])
pops.line('time', 'Is', source=sourcePops, legend_label="Symptomatic Infected", line_width=2, color='orange', line_dash=[3,3])
pops.line('time', 'R', source=sourcePops, legend_label="Recovered", line_width=2, color='green', line_dash=[8,2])
pops.line('time', 'D', source=sourcePops, legend_label="Dead", line_width=2, color='black')

S_infection_rate_slide=Slider(title="Contact Rate of Symptomatic Infection", value=beta_S, start=0, end=1, step=0.01)
A_infection_rate_slide=Slider(title="Contact Rate of Asymptomatic Infection", value=beta_A, start=0, end=1, step=0.01)
recovery_slider=Slider(title="Rate of Recovery", value=gamma, start=0, end=1, step=0.01)
death_rate_slide=Slider(title="Death Rate for Infection", value=death_rate_S, start=0, end=1, step=0.001)
#latent_time_slide=Slider(title="Exposure Latency Rate", value=E_to_I_rate, start=0, end=1, step=0.05)

def update_data(attr, old, new):
    S_infect_rate=S_infection_rate_slide.value
    A_infect_rate=A_infection_rate_slide.value
    recov_rate=recovery_slider.value
    death_rate=death_rate_slide.value
    #E_to_I=latent_time_slide.value
    
    ret = odeint(deriv, y0, t, args=(N, A_infect_rate, S_infect_rate, recov_rate, nat_death, death_rate, E_to_I_forA, E_to_I_forS, return_rate))
    S, E, Ia, Is, R, D = ret.T
    sourcePops.data=dict(time=t, S=S/1000, E=E/1000, Ia=Ia/1000, Is=Is/1000, R=R/1000, D=D/1000)

updates=[S_infection_rate_slide, A_infection_rate_slide, recovery_slider, death_rate_slide]
for u in updates:
    u.on_change('value', update_data)
widgets=column(A_infection_rate_slide, S_infection_rate_slide, recovery_slider, death_rate_slide)
curdoc().add_root(row(widgets, pops))
curdoc().title="Infectious Disease Model"




