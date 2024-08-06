#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:19:17 2020

@author: annamoragne
"""
import networkx as nx
import numpy as np
import math
from scipy.integrate import solve_ivp
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import (ColumnDataSource, Slider, TableColumn, DataTable, Button, TabPanel, Tabs, GraphRenderer, Div, Arrow, OpenHead, 
                          BoxSelectTool, Scatter, EdgesAndLinkedNodes, HoverTool, MultiLine, NodesAndLinkedEdges, Plot, Range1d, TapTool, ResetTool, Spacer, EdgesOnly)
from bokeh.plotting import figure, from_networkx
from math import exp
from bokeh.palettes import Spectral4, Colorblind8
from bokeh.models.annotations import LabelSet
TOOLS = "pan,reset,save,box_zoom"

#Setting up info and parameters for SIR model equations
N = 1000 #starting total population
# Initial number of individuals in each class
Is_nh0 = 1 #initial number of symptomatic infected individuals that are not hospitalized
Is_h0=0 #initial number of symptomatic infected individuals that are hospitalized
Ia_uk0=1 #initial number of asymptomatic infected individuals that are unaware they are infected
Ia_k0=0 #initial number of asymptomatic infected individuals that are aware they are infected
R0 = 0 #intial number of recovered individuals
D0=0 # initial number of dead individuls
E0=0 # initial number of people exposed to the virus but not yet transmitable
# Everyone else, S0, is susceptible to infection initially.
S0 = N - Is_nh0 - Is_h0 - Ia_uk0 - Ia_k0 - R0 - D0 - E0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta_S_h=0.001 #the contact/infection rate per day for hospitalized symptomatic infecteds
beta_S_nh=0.14 #the contact/infection rate per day for non-hospitalized symptomatic infecteds
beta_A_uk=0.35 #the contact/infection rate per day for unknown asymptomatic infecteds
beta_A_k=0.18 #the contact/infection rate per day for known asymptomatic infecteds
gamma = 0.02  #the recovery rate per day
gamma_hosp=1/25 #the recoverrate for hospitalized individuals
death_rate_S=0.004 #death rate for symptomatic infecteds
death_rate_hosp=0.008 #death rate for hospitalized patients
nat_death=0.0002 #natural death rate
nat_birth=0.001 #natural birth rate
E_to_I_forS=0.35 #rate at which individuals from the exposed class become symptomatic infecteds
E_to_I_forA=0.08 #rate at which as individuals from the exposed class become asymptomatic infecteds
return_rate=0.00002 #rate at which recovered people once again become susceptible
sd=1 # if social distancing is put into effect, the rate at which contact rates will decrease
v_freq=0 #frequency of people getting vaccinated
v_eff=0.98 #how effective the vaccine is 
vaccination_rate_t0 = 0.01 # Initial vaccination rate
test_rate_inc=1 #rate at which testing increases over time
hosp=0.1 #rate at which symptomatic infected individuals become hospitalized
health_capacity=150 #number of people the hospital can handle at one time in the population
hcd=1 #health capacity effecting the death rate
hcr=1 #health capacity effecting the recovery rate
# A grid of time points (in days)
t = np.linspace(0, 365, 365) #365 days
t_vac=365 #time at which vaccine is introduced

def vac_freq(t_vac, current_time, vaccination_rate): #function that gives the vaccine rate, it will be 0 before the vaccine is introduced and 0.01 after the vaccine is introduced
    #vf=(.01*exp(10*(current_time-t_vac)))/(1+exp(10*(current_time-t_vac)))
    vf=vaccination_rate*(math.atan(current_time-t_vac)+(math.pi/2))
    return vf

def health_cap_effect(health_capacity, Is_h): #function that shows death rates and recovery rates being effected if hospital capacity is surpased
    diff=(health_capacity-Is_h)
    hcd=1+((0.5*exp(-diff))/(1+exp(-diff)))
    hcr=0.7+((0.3*exp(diff))/(1+exp(diff)))
    return hcd, hcr    

# The SIR model differential equations.
def deriv(t, y, N, vaccination_rate, beta_A_uk, beta_A_k, beta_S_nh, beta_S_h, gamma, gamma_hosp, nat_death, death_rate_S, death_rate_hosp, E_to_I_forA, E_to_I_forS, return_rate, sd, test_rate_inc, t_vac, health_capacity):
    S, E, Ia_uk, Ia_k, Is_nh, Is_h, R, D = y
    v_freq=vac_freq(t_vac, t, vaccination_rate)
    test_rate=.001*t*test_rate_inc
    hce=health_cap_effect(health_capacity, Is_h)
    #below are the 8 ODEs for each of the 8 classes 
    dSdt = (-beta_S_nh * sd* S * Is_nh / N)-(beta_S_h * S * Is_h / N)-(beta_A_uk * sd * S * Ia_uk / N)-(beta_A_k * sd * S * Ia_k / N)-(nat_death*S)+(nat_birth*(N-D))+(return_rate*R)-(v_freq*v_eff*S)
    dEdt = (beta_S_nh * sd* S * Is_nh / N)+(beta_S_h * S * Is_h / N)+(beta_A_uk * sd * S * Ia_uk / N)+(beta_A_k * sd * S * Ia_k / N) - (E_to_I_forA * E)-(E_to_I_forS*E)-(nat_death*E)
    dIa_uk_dt= (E_to_I_forA*E)-(nat_death*Ia_uk)-(gamma*Ia_uk)-(test_rate*Ia_uk)
    dIa_k_dt =(test_rate*Ia_uk)-(nat_death*Ia_k)-(gamma*Ia_k)
    dIs_nh_dt = (E_to_I_forS*E)-(nat_death*Is_nh)-(death_rate_S*Is_nh)-(gamma*Is_nh)-(hosp*Is_nh)
    dIs_h_dt = (hosp*Is_nh)-(hce[0]*nat_death*Is_h)-(hce[1]*death_rate_hosp*Is_h)-(gamma_hosp*Is_h)
    dRdt = (gamma * (Ia_uk+Ia_k+Is_nh))+(gamma_hosp*Is_h)-(nat_death*R)-(return_rate*R)+(v_freq*v_eff*S)
    dDdt=nat_death*(S+E+Ia_uk+Ia_k+Is_nh+Is_h+R)+(death_rate_S*Is_nh)+(death_rate_hosp*Is_h)
    return dSdt, dEdt, dIa_uk_dt, dIa_k_dt, dIs_nh_dt, dIs_h_dt, dRdt, dDdt

# Initial conditions vector
y0 = S0, E0, Ia_uk0, Ia_k0, Is_nh0, Is_h0, R0, D0
# Integrate the SIR equations over the time grid, t.
ret = solve_ivp(deriv, t_span=(0,365), y0=y0, t_eval=t, args=(N, vaccination_rate_t0, beta_A_uk, beta_A_k, beta_S_nh, beta_S_h, gamma, gamma_hosp, nat_death, death_rate_S, death_rate_hosp, E_to_I_forA, E_to_I_forS, return_rate, sd, test_rate_inc, t_vac, health_capacity))
S, E, Ia_uk, Ia_k, Is_nh, Is_h, R, D = ret.y #solving the system of ODEs
#Creating a data source for all of class values over time 
sourcePops=ColumnDataSource(data=dict(time=t, S=S, E=E, Ia_uk=Ia_uk, Ia_k=Ia_k, Is_nh=Is_nh, Is_h=Is_h, R=R, D=D, hc=([health_capacity]*365)))
#hover_line=HoverTool(names=["S_line", "E_line"])
#creating a graph with lines for the different classes of the model
# pops=figure(title="SEIR Model Class Populations", x_axis_label="Time (in days)", y_axis_label="Proportion of people in each class", tools=TOOLS, aspect_ratio=4/3, height=450, width=600, margin=(10, 20, 10, 40))
pops=figure(title="SEIR Model Class Populations", x_axis_label="Time (in days)", y_axis_label="Proportion of people in each class", tools=TOOLS, aspect_ratio=4/3, height=350, width=350)
# pops.title.text_font_size='14pt'
#adding a line for each of the 8 different class populations
l1=pops.line('time', 'S', source=sourcePops, legend_label="Susceptible", line_width=2, color=Colorblind8[0], name="S_line")
l2=pops.line('time', 'E', source=sourcePops, legend_label="Exposed", line_width=2, color=Colorblind8[1], name="E_line")
l3=pops.line('time', 'Ia_uk', source=sourcePops, legend_label="Unknown Asymptomatic Infected", line_width=2, color=Colorblind8[2], line_dash=[4,4], name="Ia_uk_line")
l4=pops.line('time', 'Ia_k', source=sourcePops, legend_label="Known Asymptomatic Infected", line_width=2, color=Colorblind8[3], line_dash=[2,2], name="Ia_k_line")
l5=pops.line('time', 'Is_nh', source=sourcePops, legend_label="Symptomatic Infected", line_width=2, color=Colorblind8[4], line_dash=[3,3], name="Is_nh_line")
l6=pops.line('time', 'Is_h', source=sourcePops, legend_label="Hospitalized Infecteds", line_width=2, color=Colorblind8[5], name="Is_h_line")
l7=pops.line('time', 'R', source=sourcePops, legend_label="Recovered", line_width=2, color=Colorblind8[6], line_dash=[8,2], name="R_line")
l8=pops.line('time', 'D', source=sourcePops, legend_label="Dead", line_width=2, color=Colorblind8[7], name="D_line")
pops.line('time', 'hc', source=sourcePops, legend_label="Health Capacity", color="black", line_alpha=0.5, line_dash='dashed')
#legend attributes
pops.legend.click_policy="hide"
pops.legend.location='top_left'
pops.legend.background_fill_alpha=0.5

#creating a graph that only displays the 4 different types of infecteds
# infecteds=figure(title="All Infected Individuals", x_axis_label="Time (in days)", y_axis_label="Proportion of Individuals in Population", x_range=pops.x_range, tools=TOOLS, height=450, width=600, margin=(10, 20, 10, 40))
infecteds=figure(title="All Infected Individuals", x_axis_label="Time (in days)", y_axis_label="Proportion of Individuals in Population", x_range=pops.x_range, tools=TOOLS, height=350, width=350)
# infecteds.title.text_font_size='14pt'
la=infecteds.line('time', 'Ia_uk', source=sourcePops, legend_label="Unknown Asymptomatic", color=Colorblind8[2], line_width=2)
lb=infecteds.line('time', 'Ia_k', source=sourcePops, legend_label="Known Asymptomatic", line_width=2, color=Colorblind8[3], line_dash='dashed')
lc=infecteds.line('time', 'Is_nh', source=sourcePops, legend_label="Non-Hospitalized Symptomatic", line_width=2, color=Colorblind8[4])
ld=infecteds.line('time', 'Is_h', source=sourcePops, legend_label="Hospitalized", line_width=2, color=Colorblind8[5], line_dash='dashed')
infecteds.line('time', 'hc', source=sourcePops, legend_label="Health Capacity", color="black", line_alpha=0.5, line_dash='dashed')
infecteds.legend.click_policy='hide'
infecteds.legend.location='top_left'
infecteds.legend.background_fill_alpha=0.5

#Adding tool so that when user hovers over line, the current class population will be displayed
h1=HoverTool(tooltips=[("Susceptible Population", "@S")], renderers=[l1])
h2=HoverTool(tooltips=[("Exposed Population", "@E")], renderers=[l2])
h3=HoverTool(tooltips=[("Unknown Asymptomatic Population", "@Ia_uk")], renderers=[l3, la])
h4=HoverTool(tooltips=[("Known Asymptomatic Population", "@Ia_k")], renderers=[l4, lb])
h5=HoverTool(tooltips=[("Symptomatic Population", "@Is_nh")], renderers=[l5, lc])
h6=HoverTool(tooltips=[("Hospitalized Symptomatic Population", "@Is_h")], renderers=[l6, ld])
h7=HoverTool(tooltips=[("Recovered Population", "@R")], renderers=[l7])
h8=HoverTool(tooltips=[("Dead Population", "@D")], renderers=[l8])
pops.add_tools(h1, h2, h3, h4, h5, h6, h7, h8)
infecteds.add_tools(h3, h4, h5, h6)

#creating sliders for user-adjustable values
S_infection_rate_slide=Slider(title="Infection Rate of Non-Hospitalized Symptomatic", value=beta_S_nh, start=0, end=1, step=0.01, margin=(0, 5, 0, 20))
A_infection_rate_slide=Slider(title="Infection Rate of Unknown Asymptomatic", value=beta_A_uk, start=0, end=1, step=0.01, margin=(10, 5, 0, 20))
A_k_infection_rate_slide=Slider(title="Infection Rate of Known Asymptomatic", value=beta_A_k, start=0, end=1, step=.01, margin=(0, 5, 0, 20))
social_distancing=Slider(title="Rate of Social Distancing", value=(1-sd), start=0, end=1, step=0.01, margin=(0, 5, 0, 20))
recovery_slider=Slider(title="Rate of Recovery", value=gamma, start=0, end=.3, step=0.01, margin=(0, 5, 0, 20))
death_rate_slide=Slider(title="Death Rate for Infection", value=death_rate_S, start=0, end=0.5, step=0.001, margin=(0, 5, 0, 20))
testing_rate=Slider(title="Rate of Increase of Testing", value=test_rate_inc, start=1, end=5, step=0.1, margin=(0, 5, 0, 20))
vaccine_slide=Slider(title="Time at Which the Vaccine is Introduced", value=t_vac, start=0, end=365, step=1, margin=(0, 5, 0, 20))
vaccination_rate_slider=Slider(title="Rate of Vaccination in the Population", value=vaccination_rate_t0, start=0.001, end=0.02, step=0.001, format = '0.000',  margin=(0, 5, 0, 20))
hosp_space_slide=Slider(title="Additional Hospital Beds / Ventilators", value=0, start=0, end=60, step=5, margin=(0, 5, 0, 20))
return_rate_slide=Slider(title="Rate at which  Individuals Lose Immunity", value=return_rate, start=0, end=1, step=0.01, margin=(0, 5, 20, 20))

#creating a data table that will display all of the current values for certain parameters
rate_values=[nat_birth, nat_death, N, beta_A_uk, beta_A_k, beta_S_nh, beta_S_h, return_rate, E_to_I_forA, E_to_I_forS, "0.001*t*"+str(test_rate_inc), hosp, gamma , gamma_hosp, death_rate_S, death_rate_hosp, .01, 1-sd]
rate_names=["Natural Birth Rate", "Natural Death Rate", "Starting Population", "Rate unknown asymptomatics infect suscpetibles", "Rate known asymptomatics infect suscpetibles", "Rate non-hospitalized symptomatics infect suscpetibles", "Rate hospitalized symptomatics infect suscpetibles", "Rate of recovered losing immunity", "Rate of exposed class becoming asymptomatic infected", "Rate of exposed class becoming symptomatic infected", "Rate of testing", "Rate symptomatic infected become hospitalized", "Rate of recovery", "Rate of hospitalized recovery", "Death rate for non-hospitalized symptomatic infected", "Death rate for hospitalized infected", "Rate of vaccination once introduced", "Social Distancing Rate"]
data_for_table=ColumnDataSource(data=dict(names=rate_names, values=rate_values))
columnsT=[TableColumn(field='names', title="Parameter Name"), TableColumn(field='values', title="Current Value")]
# data_table=DataTable(source=data_for_table, columns=columnsT, margin=(20, 10, 10, 20), width=500, height=800)
data_table=DataTable(source=data_for_table, columns=columnsT, width=350, index_position=None)

def update_data(attr, old, new): #when slider values are adjusted this function will be called and then update the data appropriately 
    #retrieving the current value of all of the sliders
    S_infect_rate=S_infection_rate_slide.value
    A_infect_rate=A_infection_rate_slide.value
    A_k_infect=A_k_infection_rate_slide.value
    sd=(1-social_distancing.value)
    recov_rate=recovery_slider.value
    death_rate=death_rate_slide.value
    test_rate=testing_rate.value
    vaccine=vaccine_slide.value
    vaccination_rate_t = vaccination_rate_slider.value
    increase_hc=hosp_space_slide.value
    health_cap=health_capacity+increase_hc
    return_rate=return_rate_slide.value
    
    #re-solving the system of ODEs with the new parameter values from the sliders
    ret = solve_ivp(deriv, t_span=(0,365), y0=y0, t_eval=t, args=(N, vaccination_rate_t, A_infect_rate, A_k_infect, S_infect_rate, beta_S_h, recov_rate, gamma_hosp, nat_death, death_rate, death_rate_hosp, E_to_I_forA, E_to_I_forS, return_rate, sd, test_rate, vaccine, health_cap))
    S, E, Ia_uk, Ia_k, Is_nh, Is_h, R, D = ret.y
    sourcePops.data=dict(time=t, S=S, E=E, Ia_uk=Ia_uk, Ia_k=Ia_k, Is_nh=Is_nh, Is_h=Is_h,  R=R, D=D, hc=([health_cap]*365))
    data_for_table.data=dict(names=rate_names, values=[nat_birth, nat_death, N, A_infect_rate, beta_A_k, S_infect_rate, beta_S_h, return_rate, E_to_I_forA, E_to_I_forS, "0.001*t*"+str(test_rate), hosp, recov_rate, gamma_hosp, death_rate, death_rate_hosp, np.around(vaccination_rate_t, 3), 1-sd])

#this calls the update_data function when slider values are adjusted
updates=[S_infection_rate_slide, social_distancing, recovery_slider, death_rate_slide, testing_rate, A_infection_rate_slide, vaccine_slide, vaccination_rate_slider, hosp_space_slide, return_rate_slide, A_k_infection_rate_slide]
for u in updates:
    u.on_change('value', update_data)

#Creating visual layout for the program 
widgets=column(A_infection_rate_slide, A_k_infection_rate_slide, S_infection_rate_slide, social_distancing, recovery_slider, death_rate_slide, testing_rate, vaccine_slide, vaccination_rate_slider,hosp_space_slide, return_rate_slide)


#########################################################################################

#NETWORK GRAPH (will be first tab)
class_names=['Susceptible', 'Exposed', 'Unknown Asymptomatic Infected', 'Known Asymptomatic', 'Non-Hospitalized Symptomatic', 'Hospitalized Infected', 'Recovered', 'Dead']
needed_edges=[(0, 1), (0, 7), (0, 6), (1, 2), (1,4), (2, 3), (2,6), (2,7), (3, 6), (3,7), (4,5), (4,6), (4,7), (5,6), (5,7), (6,7), (6,0)] #coordinate pairs that represent an edge going from the #x node to the #y node in each pair
practice_sizes=[5, 10, 15, 20, 25, 30, 35, 40] #temporary numbers just to set up the graph

#Creating the network graph
G=nx.DiGraph()
G.add_nodes_from(range(8), name=class_names)
G.add_edges_from(needed_edges)
# plot = Plot(height=450, width=450, margin=(10, 5, 5, 20),
#             x_range=Range1d(-1.3,2.7), y_range=Range1d(-1.6,1.2))

plot = Plot(height=350, width=350, x_range=Range1d(-1.3,2.7), y_range=Range1d(-1.6,1.2))
plot.title.text = "Class Populations for Infectious Disease Outbreak"
# plot.title.text_font_size='14pt'
graph_renderer = from_networkx(G, nx.circular_layout, scale=1, center=(0,0))

#creating the nodes/circles for the network graph
graph_renderer.node_renderer.data_source.add(Colorblind8, 'color')
graph_renderer.node_renderer.data_source.add(practice_sizes, 'size')
graph_renderer.node_renderer.glyph = Scatter(size='size', fill_color='color')
graph_renderer.node_renderer.data_source.data['name'] =['Susceptible', 'Exposed', 'Unknown Asymptomatic Infected', 'Known Asymptomatic Infected', 'Non-Hospitalized Symptomatic Infected', 'Hospitalized Symptomatic Infected', 'Recovered', 'Dead']
graph_renderer.node_renderer.selection_glyph = Scatter(size='size', fill_color=Spectral4[2])
graph_renderer.node_renderer.hover_glyph = Scatter(size='size', fill_color=Spectral4[1])
graph_renderer.node_renderer.data_source

#Creating the edges for the network graph
graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=6)
graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=6)
graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=8)
graph_renderer.edge_renderer.data_source.data['edge_names']=["Susceptibles becoming exposed to disease", "Susceptibles dying of natural causes", "Susceptibles who have received the vaccine", "Exposed individuals becoming Asymptomatic Infecteds", "Exposed individuals becoming Symptomatic Infected", "Asymptomatic individuals get tested and and are then aware they are a carrier of the disease", "Individuals recover and are no longer infectious", "Dying of natural causes", "Individuals recover and are no longer infectious", "Dying of natural causes", "Becoming hospitalized", "Symptomatic individuals recover", "Symptomatic individuals die of disease or of natural causes", "Hospitalized patients recover", "Hospitalized patients die of the disease or of natural causes", "Recovered individuals die of natural causes", "Recovered individuals lose their immunity and become susceptible again"]

graph_renderer.selection_policy = NodesAndLinkedEdges()
# graph_renderer.inspection_policy = EdgesAndLinkedNodes()
graph_renderer.inspection_policy = EdgesOnly()

# add the labels to the nodes on the graph
xcoord = [1.15, .85, -.45, -1.2, -1.25, -1.25, -.15, .85] #location for the label
ycoord = [0, .75, 1.05, .85, 0.1, -.95, -1.2, -.95] #location for the label
label_source=ColumnDataSource(data=dict(x=xcoord, y=ycoord, names=class_names))
labels = LabelSet(x='x',y='y',text='names', text_font_size="13px",
                  source=label_source)
plot.add_layout(labels)
plot.renderers.append(graph_renderer)

#solving the system of ODEs with original parameters to determine size of nodes
ret = solve_ivp(deriv, t_span=(0,365), y0=y0, t_eval=t, args=(N, vaccination_rate_t0, beta_A_uk, beta_A_k, beta_S_nh, beta_S_h, gamma, gamma_hosp, nat_death, death_rate_S, death_rate_hosp, E_to_I_forA, E_to_I_forS, return_rate, sd, test_rate_inc, t_vac, health_capacity))
Sb, Eb, Ia_ukb, Ia_kb, Is_nhb, Is_hb, Rb, Db = ret.y

#creating slider for the time
# time_slider=Slider(start=0, end=365, value=0, step=1, title="Time (in Days)", width=500, margin=(10, 10, 10, 20))
time_slider=Slider(start=0, end=365, value=0, step=1, title="Time (in Days)", width=320)
start_vals=[Sb[0]/2.3, Eb[0], Ia_ukb[0], Ia_kb[0], Is_nhb[0], Is_hb[0], Rb[0]/2.3, Db[0]]
current_source=ColumnDataSource(data=dict(sizes=start_vals))
#updating the node sizes
graph_renderer.node_renderer.data_source.add(current_source.data['sizes'], 'size')
graph_renderer.node_renderer.glyph = Scatter(size='size', fill_color='color')

#when edge is hovered over, will display a description of the movement of individuals along that edge

# Define custom tooltip style
custom_tooltip = """
    <div style="width: 150px; word-wrap: break-word;">
        <span style="font-size: 12px; color: #696;">Path Movement:</span>
        <span style="font-size: 12px; font-weight: bold;">@edge_names</span>
    </div>
"""

# hover_tool = HoverTool(tooltips=[("Path Movement", "@edge_names")])
hover_tool = HoverTool(tooltips=custom_tooltip)
plot.add_tools(hover_tool, TapTool(), BoxSelectTool(), ResetTool())


####### Bar Graph
proportion_pops=[Sb[0]/1000, Eb[0]/1000, Ia_ukb[0]/1000, Ia_kb[0]/1000, Is_nhb[0]/1000, Is_hb[0]/1000, Rb[0]/1000, Db[0]/1000]
bar_source=ColumnDataSource(data=dict(tall=proportion_pops, names=class_names, colors=Colorblind8))
# bargraph=figure(x_range=class_names, y_range=Range1d(0, 1.04), title="Proportion of Population in Each Class", tools=("reset, box_zoom"), height=450, width=600, margin=(15, 10, 10, 10))
bargraph=figure(x_range=class_names, y_range=Range1d(0, 1.04), title="Proportion of Population in Each Class", tools=("reset, box_zoom"), height=450, width=520)
bargraph.vbar(x='names', top='tall', color='colors', source=bar_source, width=0.5)
# bargraph.title.text_font_size='14pt'
bargraph.xaxis.major_label_orientation=45
bar_hover=HoverTool(tooltips=[("Current Proportion", "@tall")])
bargraph.add_tools(bar_hover)


def update_data_bubble(attr, old, new): #when time slider value changes the graphs update to show class sizes at that specific time
    t=time_slider.value #current time
    #retrieving class values for time t
    newS=Sb[t]
    newE=Eb[t]
    newI1=Ia_ukb[t]
    newI2=Ia_kb[t]
    newI3=Is_nhb[t]
    newI4=Is_hb[t]
    newR=Rb[t]
    newD=Db[t]
    new_dict=[newS/2.3, newE, newI1, newI2, newI3, newI4, newR/2.3, newD]
    new_bar=[newS/1000, newE/1000, newI1/1000, newI2/1000, newI3/1000, newI4/1000, newR/1000, newD/1000]
    #updating graph values
    current_source.data=dict(sizes=new_dict)
    bar_source.data=dict(tall=new_bar, names=class_names, colors=Colorblind8)
    graph_renderer.node_renderer.data_source.add(current_source.data['sizes'], 'size')
    graph_renderer.node_renderer.glyph = Scatter(size='size', fill_color='color')
    
def animate_update(): #this function animates the graph by continually increasing the time point being looked at 
    day = time_slider.value
    if day <= 365:
        new_dict=[Sb[day]/2.3, Eb[day], Ia_ukb[day], Ia_kb[day], Is_nhb[day], Is_hb[day], Rb[day]/2.3, Db[day]]
        new_bar=[Sb[day]/1000, Eb[day]/1000, Ia_ukb[day]/1000, Ia_kb[day]/1000, Is_nhb[day]/1000, Is_hb[day]/1000, Rb[day]/1000, Db[day]/1000]
        current_source.data=dict(sizes=new_dict)
        bar_source.data=dict(tall=new_bar, names=class_names, colors=Colorblind8)
        graph_renderer.node_renderer.data_source.add(current_source.data['sizes'], 'size')
        graph_renderer.node_renderer.glyph = Scatter(size='size', fill_color='color')
        time_slider.value = day+1 #progress to next time unit
        

callback_id = None
def animate(): #this function calls the animate_update() function to animate when the button is pressed
    global callback_id
    if button.label == '► Play':
        button.label = '❚❚ Pause'
        callback_id = curdoc().add_periodic_callback(animate_update, 100)
    else:
        button.label = '► Play'
        curdoc().remove_periodic_callback(callback_id)

#adding arrows to edges to make it a directed graph
start_coord=[[.95, .117], [.97, -.07], [.9, -.1],[.64, .7257], [.625, .6691], [-.15, .9357], [0, .9], [.05, .878], [-.65, .5785], [-.65, .65], [-.94, -.14], [-.9, -.1], [-.85, -.0618], [-.6, -.7], [-.6, -.743], [.1, -.957], [.1, -.9]]
end_coord=[[.72, .653], [.73, -.63], [.1, -.9], [.1, .957], [-.84, .06], [-.61, .738], [0, -.9], [.585, -.4207], [-.12, -.708], [.6, -.6], [-.75, -.5833], [-.1, -.9], [.54, -.631], [.6, -.7], [-.2, -.914], [.6, -.743], [.9, -.1]]
for i in range(0, 17):
    plot.add_layout(Arrow(end=OpenHead(line_color="black", line_width=2, size=10, line_alpha=.65), x_start=start_coord[i][0], y_start=start_coord[i][1], x_end=end_coord[i][0], y_end=end_coord[i][1], line_alpha=0.25))

#function called when button is pressed    
time_slider.on_change('value', update_data_bubble)
# button = Button(label='► Play', width=120, margin=(1, 1, 1, 20))
button = Button(label='► Play', width=120)
button.on_click(animate)

#adding descriptive info
note1=Div(text="Note that the size of all circles are proportional to their population size, except for the Susceptible and Recovered classes, which are shown at half capacity for ease of visualization", width=350)
note2=Div(text="The outbreak modeled is based on the initial conditions of the infection rate for unknown asymptomatic infected being 0.35, for known asymptomatic infecteds being 0.18, for non-hospitalized symptomatic infected being 0.14, and for hospitalized infecteds being 0.001. The recovery rate is assumed to be 0.02. The Death rate is assumed to be 0.004 for those not hospitalized and 0.008 for those hospitalized. The rate at which people lose their immunity is 0.0002. There is no vaccine in this simulation", width=350)
# note1=Div(text="Note that the size of all circles are proportional to their population size, except for the Susceptible and Recovered classes, which are shown at half capacity for ease of visualization", width=600, margin=(20, 1, 5, 20))
# note2=Div(text="The outbreak modeled is based on the initial conditions of the infection rate for unknown asymptomatic infected being 0.35, for known asymptomatic infecteds being 0.18, for non-hospitalized symptomatic infected being 0.14, and for hospitalized infecteds being 0.001. The recovery rate is assumed to be 0.02. The Death rate is assumed to be 0.004 for those not hospitalized and 0.008 for those hospitalized. The rate at which people lose their immunity is 0.0002. There is no vaccine in this simulation", width=600, margin=(5, 1, 5, 20))
# note3=Div(text="Definition of each of the 8 classes", margin=(20, 0, 10, 10))
# n_S=Div(text="<b>Susceptible:</b> A person who is in the susceptible class is susceptible to contracting the disease and becoming infected. Everyone initially starts out in the susceptible class.", width=600, margin=(2, 0, 2, 10))
# n_E=Div(text="<b>Exposed:</b> Someone who is in the exposed class has contracted the disease but is not infected yet, which also means they are not able to infect any susceptibles while in the exposed class. This is what is known as a 'latency period'.", width=600, margin=(2, 0, 2, 10))
# n_Iuk=Div(text="<b>Unknown Asymptomatic Infected:</b> Someone in this class is an asymptomatic infected, meaning they show no symptoms of the disease. In this class they are also unaware that they are infected.", width=600, margin=(2, 0, 2, 10))
# n_Ik=Div(text="<b>Known Asymptomatic Infected:</b> Someone in this class is an asymptomatic infected, meaning they show no symptoms of the disease. However, in this class they are aware they are infected. This would happen if an individual were able to be tested for the disease and their results showed up as positive", width=600, margin=(2, 0, 2, 10))
# n_Inh=Div(text="<b>Non-Hospitalized Symptomatic Infected:</b> An individual in this class is infected and shows symptoms of the disease. In this class the individual's symptoms are not bad enough to warrant being hospitalized (however, someone can move from this class to the hospitalized class).", width=600, margin=(2, 0, 2, 10))
# n_Ih=Div(text="<b>Hospitalized Symptomatic Infected:</b> An individual in this class is infected and shows symptoms. Their symptoms are bad enough that they need medical care and are hospitalized. Someone moves into this class from the non-hospitalized symptomatic infected class.", width=600, margin=(2, 0, 2, 10))
# n_R=Div(text="<b>Recovered:</b> A person in this class was previously infected with the disease and has now recovered. While in the recovered class, a person is considered immune to the disease (although immunity can ware off an an individual can return to the susceptible class) so an individual in this class cannot spread the disease nor contract the disease. Once an individual enters the recovered class they can only move to the susceptible class (if they loose their immunity) or the dead class (if they die of natural causes).", width=600, margin=(2, 0, 2, 10))
# n_D=Div(text="<b>Dead:</b> This class represents everyone who has died. It includes people who have died from the disease or from other natural causes. Once an individual enters this class they remain in this class.", width=600, margin=(2, 0, 2, 10))

# Spacers
top_page_spacer = Spacer(height = 20)
large_top_page_spacer = Spacer(height = 20)
left_page_spacer = Spacer(width = 20)
large_left_page_spacer = Spacer(width = 20)

#layout for this tab
# display=row(column(plot, time_slider, button, note1, note2), column(bargraph, note3, n_S, n_E, n_Iuk, n_Ik, n_Inh, n_Ih, n_R, n_D))
display=row(large_left_page_spacer, column(top_page_spacer, plot, time_slider, button, note1, note2), left_page_spacer, column(top_page_spacer, bargraph))
tabA=TabPanel(child=display, title="General Outbreak") #first panel


tabB=TabPanel(child=row(large_left_page_spacer, column(top_page_spacer, pops, large_top_page_spacer, infecteds), left_page_spacer, column(top_page_spacer, widgets, data_table)), title="Adjustable SEIR Model")


##################################################################################

# Text Description of the Model 
# div1=Div(text="The general SEIR model displays the populations of 8 different classes of individuals in an infectious disease outbreak; Susceptible, Exposed, Unknown Asymptomatic Infected, Known Symptomatic Infected, Non-Hospitalized Symptomatic Infected, Hospitalized Symptomatic Infected, Recovered, and Dead. The current model displays an initial population of 1,000 individuals over 160 days.", margin=(20, 20, 10, 20), width=750)
# div2=Div(text="The Exposed class is for individuals who have been infected but are not yet showing symptoms and are not yet able to infect others. They then become either asymptomatic or symptomatic infected. Individuals can become Known Asymptomatic through testing. Testing is assumed to be available through a function of 0.001*t but the user can increase this rate through one of the available sliders.", margin=(10, 20, 10, 20), width=750)
# div3=Div(text="Symptomatic Infecteds become hospitalized at a rate of 0.1 and hospitalized individuals have a longer recovery time and higher death rate because their cases are more severe.", margin=(10, 20, 10, 20), width=750)
# div4=Div(text="Once a vaccine is introduced, individuals can move directly from the susceptible class to the recovered class. The vaccine is assumed to be 98% effective and be distributed at a rate of 0.01 once it is introduced. <br> The inclusion of social distancing will reduce the rate of infection.", margin=(10, 20, 10, 20), width=750)
# div5=Div(text="By adding hospital beds and ventilators, the health capacity of the population increases. If the amount of hospitalized symptomatic individuals surpasses the health capacity then hospitalized deaths will increase and recovery rate will decrease due to the health system being overwhelmed.", margin=(10, 20, 10, 20), width=750)
# div6=Div(text="There is assumed to be a natural birth rate of .001 and a natural death rate of 0.0002. This accounts for individuals entering and exiting the system, for causes unrelated to the outbreak.", margin=(10, 20, 10, 20), width=750)
# text_descriptions=column(div1, div2, div3, div4, div5, div6)

# tabC=TabPanel(child=text_descriptions, title="Model Description") #third panel

##########################

# Putting it all together for final output
# tabs=Tabs(tabs=[tabA, tabB, tabC])
tabs=Tabs(tabs=[tabA, tabB])
curdoc().add_root(tabs)
curdoc().title="Modeling Infectious Disease Outbreaks"
