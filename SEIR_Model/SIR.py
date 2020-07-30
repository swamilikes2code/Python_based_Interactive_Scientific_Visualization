#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:19:17 2020

@author: annamoragne
"""
import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import (ColumnDataSource, Slider, TableColumn, DataTable, Button, Panel, Tabs, GraphRenderer, Div, Arrow, OpenHead, 
                          BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool, MultiLine, NodesAndLinkedEdges, Plot, Range1d, TapTool, ResetTool)
from bokeh.plotting import figure
from math import exp
from bokeh.palettes import RdYlBu8, Spectral4
from bokeh.models.graphs import from_networkx
from bokeh.models.annotations import LabelSet
TOOLS = "pan,undo,redo,reset,save,box_zoom,tap"

# Total population, N.
N = 1000 #starting total population
# Initial number of infected and recovered individuals, I0 and R0.
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
beta_S_nh=0.1 #the contact/infection rate per day for non-hospitalized symptomatic infecteds
beta_A_uk=0.35 #the contact/infection rate per day for unknown asymptomatic infecteds
beta_A_k=0.1 #the contact/infection rate per day for known symptomatic infecteds
gamma = 0.04  #the recovery rate per day
gamma_hosp=1/25
death_rate_S=0.004 #death rate for symptomatic infecteds
death_rate_hosp=0.008
nat_death=0.0002
nat_birth=0.001
E_to_I_forS=0.5 #rate at which individuals from the exposed class become symptomatic infecteds
E_to_I_forA=0.1 #rate at which as individuals from the exposed class become asymptomatic infecteds
return_rate=0.00002 #rate at which recovered people once again become susceptible
sd=1 # if social distancing is put into effect, the rate at which contact rates will decrease
v_freq=0 #frequency of people getting vaccinated
v_eff=0.98 #how effective the vaccine is 
test_rate_inc=1 #rate at which testing increases over time
hosp=0.1
health_capacity=150
hcd=1 #health capacity effecting the death rate
hcr=1 #health capacity effecting the recovery rate
# A grid of time points (in days)
t = np.linspace(0, 365, 365) #365 days
t_vac=365 #time at which vaccine is introduced

def vac_freq(t_vac, current_time): #function that gives the vaccine rate, it will be 0 before the vaccine is introduced and 0.01 after the vaccine is introduced
    vf=(.01*exp(10*(current_time-t_vac)))/(1+exp(10*(current_time-t_vac)))
    return vf

def health_cap_effect(health_capacity, Is_h):
    diff=(health_capacity-Is_h)
    hcd=1+((0.5*exp(-diff))/(1+exp(-diff)))
    hcr=0.7+((0.3*exp(diff))/(1+exp(diff)))
    return hcd, hcr    

# The SIR model differential equations.
def deriv(t, y, N, beta_A_uk, beta_A_k, beta_S_nh, beta_S_h, gamma, gamma_hosp, nat_death, death_rate_S, death_rate_hosp, E_to_I_forA, E_to_I_forS, return_rate, sd, test_rate_inc, t_vac, health_capacity):
    S, E, Ia_uk, Ia_k, Is_nh, Is_h, R, D = y
    v_freq=vac_freq(t_vac, t)
    test_rate=.001*t*test_rate_inc
    hce=health_cap_effect(health_capacity, Is_h)
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
#ret = solve_ivp(deriv, y0, t, args=(N, beta_A_uk, beta_A_k, beta_S_nh, beta_S_h,  gamma, gamma_hosp, nat_death, death_rate_S, death_rate_hosp, E_to_I_forA, E_to_I_forS, return_rate, sd, test_rate_inc, t_vac, health_capacity))
ret = solve_ivp(deriv, t_span=(0,365), y0=y0, t_eval=t, args=(N, beta_A_uk, beta_A_k, beta_S_nh, beta_S_h, gamma, gamma_hosp, nat_death, death_rate_S, death_rate_hosp, E_to_I_forA, E_to_I_forS, return_rate, sd, test_rate_inc, t_vac, health_capacity))
S, E, Ia_uk, Ia_k, Is_nh, Is_h, R, D = ret.y
#print(type(Ia_k))
all_asymp=[]
for i in range(0,365): #this combines the two types of asymptomatic infecteds (makes it easier visually to not have so many lines)
    all_asymp.append(Ia_k[i]+Ia_uk[i])
sourcePops=ColumnDataSource(data=dict(time=t, S=S, E=E, Ia_uk=Ia_uk, Ia_k=Ia_k, Is_nh=Is_nh, Is_h=Is_h, R=R, D=D, all_Ia=all_asymp, hc=([health_capacity]*365)))

#creating a graph with lines for the different classes of the model
pops=figure(title="SEIR Model Class Populations", x_axis_label="Time (in days)", y_axis_label="Proportion of people in each class", tools=TOOLS, aspect_ratio=4/3, sizing_mode='scale_both')
pops.title.text_font_size='15pt'
pops.line('time', 'S', source=sourcePops, legend_label="Susceptible", line_width=2, color=RdYlBu8[0])
pops.line('time', 'E', source=sourcePops, legend_label="Exposed", line_width=2, color=RdYlBu8[1])
pops.line('time', 'Ia_uk', source=sourcePops, legend_label="Unknown Asymptomatic Infected", line_width=2, color=RdYlBu8[2], line_dash=[4,4])
pops.line('time', 'Ia_k', source=sourcePops, legend_label="Known Asymptomatic Infected", line_width=2, color=RdYlBu8[3], line_dash=[2,2])
#pops.line('time', 'all_Ia', source=sourcePops, legend_label="All Asymptomatic Infected", line_width=2, color='olivedrab', line_dash='dotted')
pops.line('time', 'Is_nh', source=sourcePops, legend_label="Symptomatic Infected", line_width=2, color=RdYlBu8[4], line_dash=[3,3])
pops.line('time', 'Is_h', source=sourcePops, legend_label="Hospitalized Infecteds", line_width=1, color=RdYlBu8[5])
pops.line('time', 'R', source=sourcePops, legend_label="Recovered", line_width=2, color=RdYlBu8[6], line_dash=[8,2])
pops.line('time', 'D', source=sourcePops, legend_label="Dead", line_width=2, color=RdYlBu8[7])
pops.line('time', 'hc', source=sourcePops, legend_label="Health Capacity", color="black", line_alpha=0.5, line_dash='dashed')
pops.legend.click_policy="hide"
pops.legend.location='top_left'
pops.legend.background_fill_alpha=0.5

#creating a graph that only displays the 4 different types of infecteds
infecteds=figure(title="All Infected Individuals", x_axis_label="Time (in days)", y_axis_label="Proportion of Individuals in Population", tools=TOOLS, aspect_ratio=4/3, sizing_mode='scale_both', margin=(10, 10, 10, 10))
infecteds.title.text_font_size='14pt'
infecteds.line('time', 'Ia_uk', source=sourcePops, legend_label="Uknown Asymptomatic", color='blue')
infecteds.line('time', 'Ia_k', source=sourcePops, legend_label="Known Asymptomatic", line_width=2, color='darkblue', line_dash='dashed')
infecteds.line('time', 'Is_nh', source=sourcePops, legend_label="Non-Hospitalized Symptomatic", line_width=2, color='red')
infecteds.line('time', 'Is_h', source=sourcePops, legend_label="Hospitalized", line_width=2, color=RdYlBu8[5], line_dash='dashed')
infecteds.line('time', 'hc', source=sourcePops, legend_label="Health Capacity", color="black", line_alpha=0.5, line_dash='dashed')
infecteds.legend.click_policy='hide'
infecteds.legend.location='top_left'
infecteds.legend.background_fill_alpha=0.5

#creating sliders for user-adjustable values
S_infection_rate_slide=Slider(title="Infection Rate of Non-Hospitalized Symptomatic Individuals", value=beta_S_nh, start=0, end=1, step=0.01, margin=(0, 5, 0, 15))
A_infection_rate_slide=Slider(title="Infection Rate of Unknown Asymptomatic Individuals", value=beta_A_uk, start=0, end=1, step=0.01, margin=(10, 5, 0, 15))
social_distancing=Slider(title="Rate of Social Distancing", value=(1-sd), start=0, end=1, step=0.01, margin=(0, 5, 0, 15))
recovery_slider=Slider(title="Rate of Recovery", value=gamma, start=0, end=.5, step=0.01, margin=(0, 5, 0, 15))
death_rate_slide=Slider(title="Death Rate for Infection", value=death_rate_S, start=0, end=0.5, step=0.001, margin=(0, 5, 0, 15))
testing_rate=Slider(title="Rate of Increase of Testing", value=test_rate_inc, start=1, end=5, step=0.1, margin=(0, 5, 0, 15))
vaccine_slide=Slider(title="Time at Which the Vaccine is Introduced", value=t_vac, start=0, end=365, step=1, margin=(0, 5, 0, 15))
hosp_space_slide=Slider(title="Additional Hospital Beds / Ventilators", value=0, start=0, end=60, step=5, margin=(0, 5, 0, 15))
return_rate_slide=Slider(title="Rate at which Recovered Individuals Lose Their Immunity", value=return_rate, start=0, end=1, step=0.01, margin=(0, 5, 0, 15))
#latent_time_slide=Slider(title="Exposure Latency Rate", value=E_to_I_rate, start=0, end=1, step=0.05)

#creating a data table that will display all of the current values for certain parameters
rate_values=[nat_birth, nat_death, N, beta_A_uk, beta_A_k, beta_S_nh, beta_S_h, return_rate, E_to_I_forA, E_to_I_forS, "0.001*t*"+str(test_rate_inc), hosp, gamma , gamma_hosp, death_rate_S, death_rate_hosp, v_freq, 1-sd]
rate_names=["Natural Birth Rate", "Natural Death Rate", "Starting Population", "Rate at which unknown asymptomatic infecteds infect suscpetibles", "Rate at which known asymptomatic infecteds infect suscpetibles", "Rate at which non-hospitalized symptomatic infecteds infect suscpetibles", "Rate at which hospitalized symptomatic infecteds infect suscpetibles", "Rate of recovered individuals becoming susceptible again", "Rate of exposed class that become asymptomatic infected", "Rate of exposed class that become symptomatic infected", "Rate at which individuals are being tested", "Rate was which symptomatic infected become hospitalized", "Rate of recovery", "Rate of hospitalized recovery", "Death rate for non-hospitalized symptomatic infected", "Death rate for hospitalized infected", "Rate of vaccination once introduced", "Social Distancing Rate"]
data_for_table=ColumnDataSource(data=dict(names=rate_names, values=rate_values))
columnsT=[TableColumn(field='names', title="Parameter Name"), TableColumn(field='values', title="Current Value")]
data_table=DataTable(source=data_for_table, columns=columnsT, margin=(20, 10, 10, 20), sizing_mode='scale_width')

#when slider values are adjusted this function will be called and then update the data appropriately 
def update_data(attr, old, new):
    S_infect_rate=S_infection_rate_slide.value
    A_infect_rate=A_infection_rate_slide.value
    sd=(1-social_distancing.value)
    recov_rate=recovery_slider.value
    death_rate=death_rate_slide.value
    test_rate=testing_rate.value
    vaccine=vaccine_slide.value
    increase_hc=hosp_space_slide.value
    #E_to_I=latent_time_slide.value
    health_cap=health_capacity+increase_hc
    return_rate=return_rate_slide.value
    
    #ret = odeint(deriv, y0, t, args=(N, A_infect_rate, beta_A_k, S_infect_rate, beta_S_h, recov_rate, gamma_hosp, nat_death, death_rate, death_rate_hosp, E_to_I_forA, E_to_I_forS, return_rate, sd, test_rate, vaccine, health_cap))
    ret = solve_ivp(deriv, t_span=(0,365), y0=y0, t_eval=t, args=(N, A_infect_rate, beta_A_k, S_infect_rate, beta_S_h, recov_rate, gamma_hosp, nat_death, death_rate, death_rate_hosp, E_to_I_forA, E_to_I_forS, return_rate, sd, test_rate, vaccine, health_cap))
    S, E, Ia_uk, Ia_k, Is_nh, Is_h, R, D = ret.y
    all_asymp=[]
    for i in range(0,365):
        all_asymp.append(Ia_k[i]+Ia_uk[i])
    sourcePops.data=dict(time=t, S=S, E=E, Ia_uk=Ia_uk, Ia_k=Ia_k, Is_nh=Is_nh, Is_h=Is_h,  R=R, D=D, all_Ia=all_asymp, hc=([health_cap]*365))
    data_for_table.data=dict(names=rate_names, values=[nat_birth, nat_death, N, A_infect_rate, beta_A_k, S_infect_rate, beta_S_h, return_rate, E_to_I_forA, E_to_I_forS, "0.001*t*"+str(test_rate), hosp, recov_rate, gamma_hosp, death_rate, death_rate_hosp, v_freq, 1-sd])

#this calls the update_data function when slider values are adjusted
updates=[S_infection_rate_slide, social_distancing, recovery_slider, death_rate_slide, testing_rate, A_infection_rate_slide, vaccine_slide, hosp_space_slide, return_rate_slide]
for u in updates:
    u.on_change('value', update_data)

#Creating visual layout for the program 
widgets=column(A_infection_rate_slide, S_infection_rate_slide, social_distancing, recovery_slider, death_rate_slide, testing_rate, vaccine_slide, hosp_space_slide, return_rate_slide)

tabB=Panel(child=row(widgets, column(pops, infecteds), data_table), title="Adjustable SEIR Model")


#########################################################################################

#NETWORK GRAPH (will be first tab)
class_names=['Susceptible', 'Exposed', 'Unknown Asymptomatic Infected', 'Known Asymptomatic', 'Non-Hospitalized Symptomatic', 'Hospitalized Infected', 'Recovered', 'Dead']
needed_edges=[(0, 1), (0, 7), (0, 6), (1, 2), (1,4), (2, 3), (2,6), (2,7), (3, 6), (3,7), (4,5), (4,6), (4,7), (5,6), (5,7), (6,7), (6,0)] #coordinate pairs that represent an edge going from the #x node to the #y node in each pair
practice_sizes=[5, 10, 15, 20, 25, 30, 35, 40] #temporary numbers just to set up the graph

#Creating the network graph
G=nx.DiGraph()
G.add_nodes_from(range(8), name=class_names)
G.add_edges_from(needed_edges)
plot = Plot(aspect_ratio=1/1, sizing_mode='scale_both', margin=(10, 5, 5, 20),
            x_range=Range1d(-1.7,2.1), y_range=Range1d(-1.8,1.4))
plot.title.text = "Class Populations for Infectious Disease Outbreak"
plot.title.text_font_size='15pt'

graph_renderer = from_networkx(G, nx.circular_layout, scale=1, center=(0,0))

graph_renderer.node_renderer.data_source.add(RdYlBu8, 'color')
graph_renderer.node_renderer.data_source.add(practice_sizes, 'size')
graph_renderer.node_renderer.glyph = Circle(size='size', fill_color='color')
graph_renderer.node_renderer.data_source.data['name'] =['Susceptible', 'Exposed', 'Unknown Asymptomatic Infected', 'Known Asymptomatic Infected', 'Non-Hospitalized Symptomatic Infected', 'Hospitalized Symptomatic Infected', 'Recovered', 'Dead']
graph_renderer.node_renderer.selection_glyph = Circle(size=30, fill_color=Spectral4[2])
graph_renderer.node_renderer.hover_glyph = Circle(size=30, fill_color=Spectral4[1])
graph_renderer.node_renderer.data_source

graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=6)
graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=6)
graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=8)
graph_renderer.edge_renderer.data_source.data['edge_names']=["Susceptibles becoming exposed to disease", "Susceptibles dying of natural causes", "Susceptibles who have received the vaccine", "Exposed individuals becoming Asymptomatic Infecteds", "Exposed individuals becoming Symptomatic Infected", "Asymptomatic individuals get tested and and are then aware they are a carrier of the disease", "Individuals recover and are no longer infectious", "Dying of natural causes", "Individuals recover and are no longer infectious", "Dying of natural causes", "Becoming hospitalized", "Symptomatic individuals recover", "Symptomatic individuals die of disease or of natural causes", "Hospitalized patients recover", "Hospitalized patients die of the disease or of natural causes", "Recovered individuals die of natural causes", "Recovered individuals lose their immunity and become susceptible again"]


graph_renderer.selection_policy = NodesAndLinkedEdges()
graph_renderer.inspection_policy = EdgesAndLinkedNodes()


# add the labels to the nodes on the graph
xcoord = [1.15, .85, -.35, -1.2, -1.6, -1.3, 0, .85]
ycoord = [0, .85, 1.03, .85, 0.1, -1, -1.05, -.85]
label_source=ColumnDataSource(data=dict(x=xcoord, y=ycoord, names=class_names))
labels = LabelSet(x='x',y='y',text='names', text_font_size="16px",
                  source=label_source, render_mode='canvas')
plot.add_layout(labels)

node_hover_tool = HoverTool(tooltips=[("Path Movement", "@edge_names")])
plot.add_tools(node_hover_tool, TapTool(), BoxSelectTool(), ResetTool())
plot.renderers.append(graph_renderer)

ret = solve_ivp(deriv, t_span=(0,365), y0=y0, t_eval=t, args=(N, beta_A_uk, beta_A_k, beta_S_nh, beta_S_h, gamma, gamma_hosp, nat_death, death_rate_S, death_rate_hosp, E_to_I_forA, E_to_I_forS, return_rate, sd, test_rate_inc, t_vac, health_capacity))
Sb, Eb, Ia_ukb, Ia_kb, Is_nhb, Is_hb, Rb, Db = ret.y


time_slider=Slider(start=0, end=365, value=0, step=1, title="Time (in Days)", width=500, margin=(10, 10, 10, 20))
start_vals=[Sb[0]/2.3, Eb[0], Ia_ukb[0], Ia_kb[0], Is_nhb[0], Is_hb[0], Rb[0]/2.3, Db[0]]
current_source=ColumnDataSource(data=dict(sizes=start_vals))
graph_renderer.node_renderer.data_source.add(current_source.data['sizes'], 'size')
graph_renderer.node_renderer.glyph = Circle(size='size', fill_color='color')


####### Bar Graph
proportion_pops=[Sb[0]/1000, Eb[0]/1000, Ia_ukb[0]/1000, Ia_kb[0]/1000, Is_nhb[0]/1000, Is_hb[0]/1000, Rb[0]/1000, Db[0]/1000]
bar_source=ColumnDataSource(data=dict(tall=proportion_pops, names=class_names))
bargraph=figure(x_range=class_names, plot_height=1, y_range=Range1d(0, 1.04), title="Proportion of Population in Each Class", tools=TOOLS, aspect_ratio=4/3, sizing_mode='scale_both', margin=(10, 10, 10, 10))
bargraph.vbar(x='names', top='tall', source=bar_source, width=0.5)
bargraph.title.text_font_size='12pt'
bargraph.xaxis.major_label_orientation=45


def update_data_bubble(attr, old, new):
    t=time_slider.value
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
    current_source.data=dict(sizes=new_dict)
    bar_source.data=dict(tall=new_bar, names=class_names)
    graph_renderer.node_renderer.data_source.add(current_source.data['sizes'], 'size')
    graph_renderer.node_renderer.glyph = Circle(size='size', fill_color='color')
    
def animate_update(): #this function animates the graph by continually increasing the time point being looked at 
    day = time_slider.value
    if day <= 365:
        new_dict=[Sb[day]/2.3, Eb[day], Ia_ukb[day], Ia_kb[day], Is_nhb[day], Is_hb[day], Rb[day]/2.3, Db[day]]
        new_bar=[Sb[day]/1000, Eb[day]/1000, Ia_ukb[day]/1000, Ia_kb[day]/1000, Is_nhb[day]/1000, Is_hb[day]/1000, Rb[day]/1000, Db[day]/1000]
        current_source.data=dict(sizes=new_dict)
        bar_source.data=dict(tall=new_bar, names=class_names)
        graph_renderer.node_renderer.data_source.add(current_source.data['sizes'], 'size')
        graph_renderer.node_renderer.glyph = Circle(size='size', fill_color='color')
        time_slider.value = day+1
        

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
    
time_slider.on_change('value', update_data_bubble)
button = Button(label='► Play', width=120, margin=(1, 1, 1, 20))
button.on_click(animate)

note1=Div(text="Note that the size of all circles are proportional to their population size, except for the Susceptible and Recovered classes, which are shown at half capacity for ease of visualization", width=600, margin=(20, 1, 5, 20))
note2=Div(text="The outbreak modeled is based on the initial conditions of the infection rate for unknown infected being 0.25, for known infecteds being 0.1, and for hospitalized infecteds being 0.001. The recovery rate is assumed to be 0.04. The Death rate is assumed to be 0.004 for those not hospitalized and 0.008 for those hospitalized. The rate at which people lose their immunity is 0.0002. There is no vaccine in this simulation", width=600, margin=(5, 1, 5, 20))
#latout for this tab
display=column(row(plot, bargraph), time_slider, button, note1, note2)

tabA=Panel(child=display, title="General Outbreak")



##################################################################################

# Text Description of the Model 
div1=Div(text="The general SEIR model displays the populations of 8 different classes of individuals in an infectious disease outbreak; Susceptible, Exposed, Unknown Asymptomatic Infected, Known Symptomatic Infected, Non-Hospitalized Symptomatic Infected, Hospitalized Symptomatic Infected, Recovered, and Dead. The current model displays an initial population of 1,000 individuals over 160 days.", margin=(20, 20, 10, 20), width=750)
div2=Div(text="The Exposed class is for individuals who have been infected but are not yet showing symptoms and are not yet able to infect others. They then become either asymptomatic or symptomatic infected. Individuals can become Known Asymptomatic through testing. Testing is assumed to be available through a function of 0.001*t but the user can increase this rate through one of the available sliders.", margin=(10, 20, 10, 20), width=750)
div3=Div(text="Symptomatic Infecteds become hospitalized at a rate of 0.1 and hospitalized individuals have a longer recovery time and higher death rate because their cases are more severe.", margin=(10, 20, 10, 20), width=750)
div4=Div(text="Once a vaccine is introduced, individuals can move directly from the susceptible class to the recovered class. The vaccine is assumed to be 98% effective and be distributed at a rate of 0.01 once it is introduced. <br> The inclusion of social distancing will reduce the rate of infection.", margin=(10, 20, 10, 20), width=750)
div5=Div(text="By adding hospital beds and ventilators, the health capacity of the population increases. If the amount of hospitalized symptomatic individuals surpasses the health capacity then hospitalized deaths will increase and recovery rate will decrease due to the health system being overwhelmed.", margin=(10, 20, 10, 20), width=750)
div6=Div(text="There is assumed to be a natural birth rate of .001 and a natural death rate of 0.0002. This accounts for individuals entering and exiting the system, for causes unrelated to the outbreak.", margin=(10, 20, 10, 20), width=750)
text_descriptions=column(div1, div2, div3, div4, div5, div6)

tabC=Panel(child=text_descriptions, title="Model Description")


##########################
# Putting it all together
tabs=Tabs(tabs=[tabA, tabB, tabC])

curdoc().add_root(tabs)
curdoc().title="Modeling Infectious Disease Outbreaks"





