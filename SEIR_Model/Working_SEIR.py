
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
beta_S_nh=0.15 #the contact/infection rate per day for non-hospitalized symptomatic infecteds
beta_A_uk=0.35 #the contact/infection rate per day for unknown asymptomatic infecteds
beta_A_k=0.01 #the contact/infection rate per day for known symptomatic infecteds
gamma = 1/14  #the recovery rate per day
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
test_rate=0.01
hosp=0.1
health_capacity=150
hcd=1 #health capacity effecting the death rate
hcr=1 #health capacity effecting the recovery rate
# A grid of time points (in days)
t = np.linspace(0, 160, 160) #160 days
t_vac=160 #time at which vaccine is introduced

def vac_freq(t_vac, total_time):
    vac_f=0
    if total_time<t_vac:
        vac_f=0
    elif total_time>=t_vac:
        vac_f=0.01
    return vac_f
def health_cap_effect(health_capacity, Is_h):
    if Is_h<health_capacity:
        hcd=1
        hcr=1
    elif Is_h>health_capacity:
        hcd=1.5
        hcr=0.7
    health_effect=[hcd, hcr]
    return health_effect
        

# The SIR model differential equations.
def deriv(y, t, N, beta_A_uk, beta_A_k, beta_S_nh, beta_S_h, gamma, gamma_hosp, nat_death, death_rate_S, death_rate_hosp, E_to_I_forA, E_to_I_forS, return_rate, sd, test_rate, t_vac, health_capacity):
    S, E, Ia_uk, Ia_k, Is_nh, Is_h, R, D = y
    v_freq=vac_freq(t_vac, t)
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
ret = odeint(deriv, y0, t, args=(N, beta_A_uk, beta_A_k, beta_S_nh, beta_S_h,  gamma, gamma_hosp, nat_death, death_rate_S, death_rate_hosp, E_to_I_forA, E_to_I_forS, return_rate, sd, test_rate, t_vac, health_capacity))
S, E, Ia_uk, Ia_k, Is_nh, Is_h, R, D = ret.T
#print(type(Ia_k))
all_asymp=[]
for i in range(0,160):
    all_asymp.append((Ia_k[i]+Ia_uk[i])/1000)
sourcePops=ColumnDataSource(data=dict(time=t, S=S/1000, E=E/1000, Ia_uk=Ia_uk/1000, Ia_k=Ia_k/1000, Is_nh=Is_nh/1000, Is_h=Is_h/1000, R=R/1000, D=D/1000, all_Ia=all_asymp, hc=([health_capacity/1000]*160)))

pops=figure(title="SEIR Model Class Populations", x_axis_label="Time (in days)", y_axis_label="Proportion of people in each class", tools=TOOLS, width=1000, height=800)
pops.title.text_font_size='15pt'
pops.line('time', 'S', source=sourcePops, legend_label="Susceptible", line_width=2, color='darkblue')
#pops.line('time', 'E', source=sourcePops, legend_label="Exposed", line_width=2, color='lightblue')
#pops.line('time', 'Ia_uk', source=sourcePops, legend_label="Unknown Asymptomatic Infected", line_width=2, color='red', line_dash=[4,4])
#pops.line('time', 'Ia_k', source=sourcePops, legend_label="Known Asymptomatic Infected", line_width=2, color='yellow', line_dash=[2,2])
pops.line('time', 'all_Ia', source=sourcePops, legend_label="All Asymptomatic Infected", line_width=2, color='olivedrab', line_dash='dotted')
pops.line('time', 'Is_nh', source=sourcePops, legend_label="Symptomatic Infected", line_width=2, color='orange', line_dash=[3,3])
pops.line('time', 'Is_h', source=sourcePops, legend_label="Hospitalized Infecteds", line_width=1, color='red')
pops.line('time', 'R', source=sourcePops, legend_label="Recovered", line_width=2, color='green', line_dash=[8,2])
pops.line('time', 'D', source=sourcePops, legend_label="Dead", line_width=2, color='black')
pops.line('time', 'hc', source=sourcePops, legend_label="Health Capacity", color="black", line_alpha=0.5, line_dash='dashed')

S_infection_rate_slide=Slider(title="Infection Rate of Non-Hospitalized Symptomatic Individuals", value=beta_S_nh, start=0, end=1, step=0.01)
A_infection_rate_slide=Slider(title="Infection Rate of Unknown Asymptomatic Individuals", value=beta_A_uk, start=0, end=1, step=0.01)
social_distancing=Slider(title="Rate of Social Distancing", value=(1-sd), start=0, end=1, step=0.01)
recovery_slider=Slider(title="Rate of Recovery", value=gamma, start=0, end=.5, step=0.01)
death_rate_slide=Slider(title="Death Rate for Infection", value=death_rate_S, start=0, end=0.5, step=0.001)
testing_rate=Slider(title="Rate at Which People Are Being Tested", value=test_rate, start=0, end=1, step=0.05)
vaccine_slide=Slider(title="Time at Which the Vaccine is Introduced", value=t_vac, start=0, end=160, step=1)
hosp_space_slide=Slider(title="Additional Hospital Beds / Ventilators", value=0, start=0, end=60, step=5)
#latent_time_slide=Slider(title="Exposure Latency Rate", value=E_to_I_rate, start=0, end=1, step=0.05)

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
    
    ret = odeint(deriv, y0, t, args=(N, A_infect_rate, beta_A_k, S_infect_rate, beta_S_h, recov_rate, gamma_hosp, nat_death, death_rate, death_rate_hosp, E_to_I_forA, E_to_I_forS, return_rate, sd, test_rate, vaccine, health_cap))
    S, E, Ia_uk, Ia_k, Is_nh, Is_h, R, D = ret.T
    all_asymp=[]
    for i in range(0,160):
        all_asymp.append((Ia_k[i]+Ia_uk[i])/1000)
    sourcePops.data=dict(time=t, S=S/1000, E=E/1000, Ia_uk=Ia_uk/1000, Ia_k=Ia_k/1000, Is_nh=Is_nh/1000, Is_h=Is_h/1000,  R=R/1000, D=D/1000, all_Ia=all_asymp, hc=([health_cap/1000]*160))

updates=[S_infection_rate_slide, social_distancing, recovery_slider, death_rate_slide, testing_rate, A_infection_rate_slide, vaccine_slide, hosp_space_slide]
for u in updates:
    u.on_change('value', update_data)
widgets=column(A_infection_rate_slide, S_infection_rate_slide, social_distancing, recovery_slider, death_rate_slide, testing_rate, vaccine_slide, hosp_space_slide)
curdoc().add_root(row(widgets, pops))
curdoc().title="Infectious Disease Model"




