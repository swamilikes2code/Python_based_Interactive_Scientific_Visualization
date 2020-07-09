import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, Slider, Select, Paragraph, TableColumn, DataTable, Button, Panel, Tabs, LinearAxis, Range1d
from bokeh.plotting import figure, show
TOOLS = "pan,undo,redo,reset,save,box_zoom,tap"

# Total population, N.
N = 1000 #US population
# Initial number of infected and recovered individuals, I0 and R0.
I0 = 1 #initial number of infected individuals
R0 = 0 #intial number of recovered individuals
D0=0 # initial number of dead individuls
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0-D0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta=0.25 #the contact/infection rate per day
gamma = .1  #the recovery rate per day
death_rate=0.006
nat_death=0.002
# A grid of time points (in days)
t = np.linspace(0, 160, 160) #160 days

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma, nat_death, death_rate):
    S, I, R, D = y
    dSdt = (-beta * S * I / N)-(nat_death*S) 
    dIdt = (beta * S * I / N - gamma * I)-(nat_death*I)-(death_rate*I)
    dRdt = (gamma * I)-(nat_death*R)
    dDdt=nat_death*(S+I+R)
    return dSdt, dIdt, dRdt, dDdt

# Initial conditions vector
y0 = S0, I0, R0, D0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma, nat_death, death_rate))
S, I, R, D = ret.T
#print(D)
sourcePops=ColumnDataSource(data=dict(time=t, S=S/1000, I=I/1000, R=R/1000, D=D/1000))

pops=figure(title="SEIR Model Class Populations", x_axis_label="Time (in days)", y_axis_label="Proportion of people in each class", tools=TOOLS)
pops.title.text_font_size='15pt'
pops.line('time', 'S', source=sourcePops, legend_label="Susceptible", line_width=2, color='darkblue')
pops.line('time', 'I', source=sourcePops, legend_label="Infected", line_width=2, color='red', line_dash=[4,4])
pops.line('time', 'R', source=sourcePops, legend_label="Recovered", line_width=2, color='green', line_dash=[8,2])
pops.line('time', 'D', source=sourcePops, legend_label="Dead", line_width=2, color='black')

infection_rate_slide=Slider(title="Contact Rate of Infection", value=beta, start=0, end=1, step=0.01)
recovery_slider=Slider(title="Rate of Recovery", value=gamma, start=0, end=1, step=0.01)
death_rate_slide=Slider(title="Death Rate for Infection", value=death_rate, start=0, end=1, step=0.01)

def update_data(attr, old, new):
    infect_rate=infection_rate_slide.value
    recov_rate=recovery_slider.value
    death_rate=death_rate_slide.value
    
    ret = odeint(deriv, y0, t, args=(N, infect_rate, recov_rate, nat_death, death_rate))
    S, I, R, D = ret.T
    sourcePops.data=dict(time=t, S=S/1000, I=I/1000, R=R/1000, D=D/1000)

updates=[infection_rate_slide, recovery_slider, death_rate_slide]
for u in updates:
    u.on_change('value', update_data)
widgets=column(infection_rate_slide, recovery_slider, death_rate_slide)
curdoc().add_root(row(widgets, pops))
curdoc().title="Infectious Disease Model"




