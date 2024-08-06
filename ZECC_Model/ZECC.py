from bokeh.io import curdoc
import ast
from bokeh.layouts import row, column, Spacer
from bokeh.models import ColumnDataSource, Slider, Select, Paragraph, TableColumn, DataTable, Button, TabPanel, Tabs, LinearAxis, Range1d, HoverTool
from bokeh.plotting import figure
import numpy as np
import pandas as pd 
from scipy.interpolate import interp1d

time_range=list(range(0, 24)) #hourly time scale
time_range1=list(range(1,13)) #yearly time scale
initial_dims=[3, 2, 1, .3] #starting dimensions of the chamber [length, width, height, sand_thickness]
materials=["Brick", "Wood", "Terracotta", "Concrete"] #possible materials to choose from
time_ranges=["12 Months", "24 Hours"] #possible time ranges

#Creating a map of the world to show where each of the 6 possible locations are
mapp = figure(x_range=(-14000000, 7000000), y_range=(-4000000, 6060000), # range bounds supplied in web mercator coordinates
           x_axis_type="mercator", y_axis_type="mercator", height=300, width=370)
mapp.add_tile("CartoDB Positron", retina=True)
#adding each location to the map
mapp.scatter(x=-8389827.854690, y=4957234.168513, size=10, fill_color='blue', fill_alpha=0.7, legend_label="Bethlehem, PA")
mapp.scatter(x=-8931102.469623, y=2972160.043550, size=10, fill_color='darkred', fill_alpha=.7, legend_label="Miami, FL")
mapp.scatter(x=-9290844.007714, y=953484.087498, size=10, fill_color='darkgreen', fill_alpha=0.7, legend_label="Puerto Jiménez, Costa Rica")
mapp.scatter(x=-8741967.501084, y=-22993.039835, size=10, fill_color='peru', fill_alpha=0.7, legend_label="Quito, Ecuador")
mapp.scatter(x=4105174.772925, y=-145162.620135, size=10, fill_color='mediumpurple', fill_alpha=0.7, legend_label="Nairobi, Kenya")
mapp.scatter(x=3564845.194234, y=-948229.994036, size=10, fill_color='navy', fill_alpha=0.7, legend_label="Lusaka, Zambia")
mapp.legend.background_fill_alpha=0.5
#if you add more locations to the spreadsheets then you will need to manually add another point on the map here

master = True
# master = False

if master == True:
    #Creating a Pandas data frame for data stored in csv files that are found in the same folder as the program is stored
    yearly_temps_df=pd.read_csv("ZECC_Model/Yearly_Temps.csv", index_col=0, header=0) #reading in data for the monthly average temperatures for 6 different locations
    yearly_rh_df=pd.read_csv("ZECC_Model/Yearly_RH.csv", index_col=0, header=0) #reading in data for the monthly average of relative humidity for 6 different locations
    daily_temps_df=pd.read_csv("ZECC_Model/Hourly_Temps.csv", index_col=0, header=0) #reading in data for the hourly temperatures for one day in mid-June for 6 locations
    daily_rh=pd.read_csv("ZECC_Model/ZECC_Daily_rh.csv", index_col=0, header=0) #reading in data for the dailt relative humidity values for each location
else:
    yearly_temps_df=pd.read_csv("./Yearly_Temps.csv", index_col=0, header=0) #reading in data for the monthly average temperatures for 6 different locations
    yearly_rh_df=pd.read_csv("./Yearly_RH.csv", index_col=0, header=0) #reading in data for the monthly average of relative humidity for 6 different locations
    daily_temps_df=pd.read_csv("./Hourly_Temps.csv", index_col=0, header=0) #reading in data for the hourly temperatures for one day in mid-June for 6 locations
    daily_rh=pd.read_csv("./ZECC_Daily_rh.csv", index_col=0, header=0) #reading in data for the dailt relative humidity values for each location


TOOLS = "reset,save,box_zoom" #tools for the graphs
#Creating Grpah to show average temps throught the year for each location
diff_temps=figure(title="Average Temperature Throughout the Year", x_axis_label="Months", y_axis_label="Temperature in Celsius", tools=TOOLS, height=300, width=370)
# diff_temps.title.text_font_size='14pt'
diff_temps.xaxis.ticker = list(range(1, 13))
diff_temps.xaxis.major_label_overrides={1:'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 
                                        7: "July", 8:'August', 9:'September', 10: 'October', 11: 'November', 12: 'December'} #makes x-axis have name labels instead of month numbers
diff_temps.xaxis.major_label_orientation=1

#creating a graph to show the 6 locations temperatures throught one day 
hourly_temps=figure(title="Temperatures Throughout One Day in Mid-June", x_axis_label="Time in Hours", y_axis_label="Temperature in Celsius", tools=TOOLS, height=300, width=370)
# hourly_temps.title.text_font_size='14pt'

#Creating a graph to show the average humidity trends for each location throughout the year
humid=figure(title="Average Humidity Throughout The Year", x_axis_label="Months", y_axis_label="Relative Humidity", x_range=diff_temps.x_range, tools=TOOLS, height=300, width=370)
# humid.title.text_font_size='14pt'
humid.xaxis.ticker = list(range(1, 13))
humid.xaxis.major_label_overrides={1:'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 
                                        7: "July", 8:'August', 9:'September', 10: 'October', 11: 'November', 12: 'December'} #chaning x-axis to month names instead of numbers
humid.xaxis.major_label_orientation=1

colors=['blue', 'darkgreen', 'darkred', 'peru', 'mediumpurple', 'navy']
x=0
#Creating lines on each of the three graphs
for i in range(0, len(yearly_temps_df)):
    diff_temps.line(time_range1, yearly_temps_df.iloc[i], legend_label=yearly_temps_df.index[i], line_width=2, color=colors[x])
    hourly_temps.line(time_range, daily_temps_df.iloc[i], legend_label=daily_temps_df.index[i], line_width=2, color=colors[x])
    humid.line(time_range1, yearly_rh_df.iloc[i], legend_label=yearly_rh_df.index[i], line_width=2, color=colors[x])
    x=x+1
#yearly temperature graph legend attributes
diff_temps.legend.click_policy="hide"
diff_temps.legend.location='bottom_left'
diff_temps.legend.background_fill_alpha=0.7
#hourly temperature graph legend attributes
hourly_temps.legend.click_policy='hide'
hourly_temps.legend.location='bottom_left'
hourly_temps.legend.background_fill_alpha=0.7
#humidity graph legend attributes
humid.legend.click_policy="hide"
humid.legend.location='bottom_left'
humid.legend.background_fill_alpha=0.7

def calc_HC (temps, dims, conductivity, desired_temp): #function to calculate the heat conduction for yearly time scale
    k=conductivity
    Area=2*(dims[0]*dims[2])+ 2*(dims[1]*dims[2])
    Tcold=desired_temp
    d=dims[3]
    new_list=[]
    for i in temps:
        new_list.append(24*30*(k*Area)*(i-Tcold)/d)
    return new_list

def HC_hourly (temps, dims, conductivity, desired_temp): #function to calculate the heat conduction for hourly time scale
    k=conductivity
    Area=2*(dims[0]*dims[2])+ 2*(dims[1]*dims[2])
    Tcold=desired_temp
    d=dims[3]
    new_list=[]
    for i in temps:
        new_list.append((k*Area)*(i-Tcold)/d)
    return new_list

k_sand = 0.27  # thermal conductivity of dry sand W/mK
k_water = 0.6  # thermal conductivity of water W/mK
k_brick = 0.72  # thermal conductivity of brick W/mK
e_sand = 0.343  # porosity of sand

#first we calculate all values for Costa Rica yearly so we have something to initially show on our graphs before user makes adjustments
out1=calc_HC(yearly_temps_df.iloc[2], initial_dims, k_brick, 15)
source=ColumnDataSource(data=dict(time=time_range1, output=out1)) #creating a data source to show data that we currently want displayed
start1=np.min(source.data['output'])
end1=np.max(source.data['output'])

#creating a graph to show the heat conduction and evaporative cooling rate for desired ZECC
g1=figure(title="Heat per Time", x_axis_label="Time in Months", y_axis_label="Heat Conduction per Time", tools=TOOLS, height=300, width=370)
gg1=g1.line('time', 'output', source=source, color="purple", legend_label="Heat Conduction", line_dash=[4,4], line_width=3)
g1.y_range=Range1d(start1, end1)
g1.legend.click_policy="hide"
g1.legend.background_fill_alpha=0.5
# g1.title.text_font_size='14pt'
g1.legend.location='top_left'

location_options=[]
for x in range(0,len(yearly_temps_df.index)):
    location_options.append(yearly_temps_df.index[x])
    
#adding sliders for adjustable dimensions of chamber and drop down menus for location, time interval, and material specifications
slide_length=Slider(title="Length of Chamber", value=initial_dims[0], start=0, end=12, step=0.5, width=340)
slide_width=Slider(title="Width of Chamber", value=initial_dims[1], start=0, end=12, step=0.5, width=340)
slide_height=Slider(title="Height of Chamber", value=initial_dims[2], start=0, end=5, step=0.25, width=340)
slide_thick=Slider(title="Thickness of Sand Layer in Chamber Wall", value=initial_dims[3], start=0, end=1, step=0.001, width=340)
select_material=Select(title="Choice of Material for Walls of the Chamber:", value="Brick", options=materials, width=340)
slide_desired_temp=Slider(title="Desired Temperature for the Inner Chamber", value=20, start=2, end=50, step=0.5, width=340)
location_select=Select(title="Location", value="Puerto Jiménez, Costa Rica", options=location_options, width=340)
time_select=Select(title="Time Interval", value="12 Months", options=time_ranges, width=340)
calculate_button=Button(label="Calculate", button_type='success', width=340) #a button that will calculate cost and water needed when clicked

def latent_heat(temp): #function to interpolate latent heat value
    #Interpolating the values for latent heat of evaporation
    y = [45054, 44883,44627,44456,44200,43988,43774,43602,43345,43172,42911,42738,42475,42030,41579,41120] #latent heat of vaporization array
    x = [0,5,10,15,20,25,30,35,40,45,50,55,60,70,80,90] #water temperature array
    f1 = interp1d(x, y, kind= 'cubic')
    return f1(temp)

latent_out=latent_heat(yearly_temps_df.iloc[2])

def SVP(temp):
    #Interpolate the values for Saturated Vapor Pressure
    x=[.01, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
       30, 31, 32, 33, 34, 35, 36, 38, 40, 45, 50, 55, 60, 65, 70]
    y=[0.00611, 0.00813, 0.00872, 0.00935, 0.01072, 0.01228, 0.01312, 0.01402, 0.01497, 0.01598, 0.01705, 0.01818, 0.01938, 0.02064, 0.02198,
       0.02339, 0.02487, 0.02645, 0.02810, 0.02985, 0.03169, 0.03363, 0.03567, 0.03782, 0.04008, 0.04246, 0.04496, 0.04759, 0.05034, 0.05324,
       0.05628, 0.05947, 0.06632, 0.07384, 0.09593, .1235, .1576, .1994, .2503, .3119]
    vals=interp1d(x, y, kind='cubic')
    return vals(temp)

def water_needed(dims, temp, SVP, rh): #function to calculate the amount of water needed for the chamber to function properly on a yearly time scale
    theta=34.5 #(kg/(m^2*hr))
    SA=2*(dims[0]+dims[3]+.225)*dims[2] + 2*dims[2]*(dims[1]+dims[3]+.225) #surface area
    A = 18.3036 #constant value
    B = 3816.44 #constant value
    C = -46.13 #constant value
    p_star=[]
    p_air=[]
    evap_rate=[]
    for i in temp:
        p_star.append(np.exp(A - B / (C + i + 273))) 
        # Antoine equation for vapor pressure at outside air
    for j in range(0, 12):
        p_air.append(rh.iloc[j]*p_star[j])
    #for j in p_star:
     #   p_air.append(rh*j) 
        # bulk pressure of air at t bulk
    for x in range(0,12):
        yy=theta*SA*((p_star[x]-p_air[x])/760) #in L/hour
        yy=yy*(24*30) #in L/month
        evap_rate.append(yy)
    return evap_rate

def water_needed_hourly(dims, temp, SVP, rh): #function to calculate the amount of water needed for the chamber to function properly on an hourly time scale
    theta=34.5 #(kg/(m^2*hr))
    SA=2*(dims[0]+dims[3]+.225)*dims[2] + 2*dims[2]*(dims[1]+dims[3]+.225)
    A = 18.3036 #constant value
    B = 3816.44 #constant value
    C = -46.13 #constant value
    p_star=[]
    p_air=[]
    evap_rate=[]
    for i in temp:
        p_star.append(np.exp(A - B / (C + i + 273))) 
        # Antoine equation for vapor pressure at outside air
    for j in range(0, 24):
        p_air.append(rh.iloc[0]*p_star[j]) 
        # bulk pressure of air at t bulk 
    for x in range(0,24):
        yy=theta*SA*((p_star[x]-p_air[x])/760) #in L/hour
       # yy=yy*(1/1000)*(3600) #in L/hour
        evap_rate.append(yy)
    return evap_rate

vap_init=[]
for p in yearly_temps_df.iloc[2]:
    vap_init.append(SVP(p))
vap1_init=[]
for p in yearly_temps_df.iloc[2]:
    vap1_init.append(SVP(p))

#getting water for Costa Rica as that will be the initial display
water_monthly=water_needed(initial_dims, yearly_temps_df.iloc[2], vap_init, yearly_rh_df.iloc[2])
water_trial=water_needed_hourly(initial_dims, daily_temps_df.iloc[2], vap1_init, daily_rh.iloc[2])
sourceW=ColumnDataSource(data=dict(time=time_range1, temps=yearly_temps_df.iloc[2], water=water_monthly))


#Evaporative Cooling Rate Q/t=mLv/t
def evap_cool(mass, latent, time): #for yearly time scale
    cooling_rate=[]
    for w in range(0,12):
        cooling_rate.append((mass[w]*latent[w])/100)
    return cooling_rate

def evap_cool_hourly(mass, latent, time): #for hourly time scale
    cooling_rate=[]
    for w in range(0,24):
        cooling_rate.append((mass[w]*latent[w])/100)
    return cooling_rate

evap_out=evap_cool(water_monthly, latent_out, time_range1) #calculating inital for Costa Rica for initial display
source3=ColumnDataSource(data=dict(time=time_range1, evap_out=evap_out)) #data source to store info
start=int(min(source3.data['evap_out'])) #starting value of second y-axis
end=int(max(source3.data['evap_out'])) #ending value of second y-axis
g1.extra_y_ranges['second']=(Range1d(start, end)) #creating a second y-axis since evaporative cooling rate and heat conduction are on very different scales
#adding line for evaporatove cooling rate to g1 graph that has the heat conduction rate
gg2=g1.line('time', 'evap_out', source=source3, color='orange', legend_label="Evaporation Cooling Rate", line_width=2, y_range_name='second')
ax2 = LinearAxis(y_range_name="second", axis_label="Evaporative Cooling Heat per Time")
g1.add_layout(ax2, 'left') #adding the second y_axis to the graph

#Adding display of value when line hovered over at specific point
hh2=HoverTool(tooltips=[("Heat per Time", "@evap_out")], renderers=[gg2])
hh1=HoverTool(tooltips=[("Heat per Time", "@output")], renderers=[gg1])
g1.add_tools(hh1, hh2)


def cost_calc(dims, water_amount, mat): #calculating the cost to build and operate the ZECC on a yearly time scale
    #dims=[brick_length, brick_width, brick_height, sand_thickness]
    L0=dims[0] #length of inner brick chamber
    w0=dims[1] #width of inner brick chamber
    L1 = 0.1125
    L3=0.1125#thickness of brick
    L2=dims[3] #thickness of sand
    h=dims[2] #height of chamber
    w1 = w0 + 2 * L1  # width of inner brick layer
    w2 = w1 + 2 * L2  # width of sand layer
    w3 = w2 + 2 * L3  # width of outer brick layer
    A0 = L0 * w0  # area of inner chamber
    A1 = ((L0 + L1) * w1) - A0  # area of inner brick layer
    A2 = ((L0 + L1 + L2) * w2) - A1  # area of sand layer
    A3 = ((L0 + L1 + L2 + L3) * w3) - A2  # area of outer brick layer
    V0 = A0 * h  # inner chamber volume
    V1 = A1 * h  # inner brick volume
    V2 = A2 * h  # sand volume
    V3 = A3 * h  # outer brick volume
    materials_cost=0
    if mat=="Brick":
       materials_cost= 1900*0.037*V1 + 1905*.05*V2 + 1900*0.037*V3
       #Brick cost 0.037 $/Kg and density is 1900 Kg/m^3
    elif mat=="Wood":
        materials_cost=1905*0.5*V2 + (V1+V2)*(2.43*689)
        #Wood cost $2.43/Sq F and desnsity is 689 Kg/m^3
    elif mat=="Terracotta":
        materials_cost=1905*0.5*V2 + (V1+V2)*(15*2710)
        #Terracotta cost is $15 per sq inch and density is 2710 Kg/m^3
    elif mat=="Concrete":
        materials_cost=1905*0.5*V2 +(V1+V2)*(98.425)
        #Concrete cost is $98.425/m^3
    #cost of sand 0.05 $/kg
    #Density of Sand (kg/m^3): 1905
    water_cost=water_amount*0.0001
    final_cost=materials_cost+water_cost
    return final_cost

price1=cost_calc(initial_dims, sum(water_monthly), "Brick") #inital cost of ZECC in Costa Rica made with Brick using initial dimensions
sourceP=ColumnDataSource(data=dict(price=[price1])) #storing info in Price data source

#Creating a data table that will print out the volume of the chamber as well as the cost of the ZECC and the amount of water needed, both on a daily and yearly time scale
tableName=["Puerto Jiménez, Costa Rica"]
tablePriceY=["$"+str(round(price1, 2))]
tablePriceD=["$"+str(round(price1/365,2))]
tableWaterY=[str(round(sum(water_monthly), 2))+" L"]
tableWaterD=[str(round(sum(water_monthly)/365, 2)) +" L"]
tableSpace=[str(round(initial_dims[0]*initial_dims[1]*initial_dims[2], 2))+" m^3"]
tableTime=[time_ranges[0]]

#putting info into data table
sourceTable=ColumnDataSource(data=dict(name=tableName, time=tableTime, Year_Price=tablePriceY, Day_Price=tablePriceD, Year_Water=tableWaterY, Day_Water=tableWaterD, space=tableSpace))
columnsT=[TableColumn(field='name', title='Location', width=160), TableColumn(field='time', title='Time Interval', width=80), TableColumn(field='space', title='Storage Volume Capacity (in m^3)', width=140), 
          TableColumn(field='Day_Water', title='Daily Water Input (in L)', width=100), TableColumn(field='Year_Water', title='Yearly Water Input (in L)', width=140),
          TableColumn(field='Day_Price', title='Daily Cost in $', width=90), TableColumn(field='Year_Price', title='Yearly Cost in $', width=100)]
data_table=DataTable(source=sourceTable, columns=columnsT, width=350, height=200, autosize_mode="none")

def dew_point(temps, rh, time): #calculating dew point of location at speific time
    dp_out=[]
    a = 17.27
    b = 237.7
    for t in time:
        alpha = b * (((a * temps.iloc[t]) / (b + temps.iloc[t])) + np.log(rh.iloc[t]))
        gamma = a - (((a * temps.iloc[t]) / (b + temps.iloc[t])) + np.log(rh.iloc[t]))
        dp_out.append(alpha / gamma)
    return dp_out
def dew_point_hourly(temps, rh, time): #calculating dew point of loction at specific time
    dp_out=[]
    a = 17.27
    b = 237.7
    for t in time:
        alpha = b * (((a * temps.iloc[t]) / (b + temps.iloc[t])) + np.log(rh))
        gamma = a - (((a * temps.iloc[t]) / (b + temps.iloc[t])) + np.log(rh))
        dp_out.append(alpha / gamma)
    return dp_out
dp_Costa=dew_point(yearly_temps_df.iloc[2], yearly_rh_df.iloc[2], range(0,12)) #dew point for initial Costa Rica ZECC

#creating a graph that shows the ambient temp, outer wall temp, and dew point temp
g4=figure(title="Essential Temperature Values for Selected Location", x_axis_label="Time (in Months)", y_axis_label="Temperature (in Celsius)", tools=TOOLS, height=300, width=350)
g4.title.text_font_size='9pt'
sourceDP=ColumnDataSource(data=dict(time=time_range1, temps=yearly_temps_df.iloc[2], dp=dp_Costa, T1=range(0,12)))
gl1=g4.line('time', 'temps', source=sourceDP, color='orange', line_width=2, legend_label="Ambient Temperature")
gl2=g4.line('time', 'dp', source=sourceDP, color='darkblue', line_width=2, line_dash=[4,4], legend_label="Dew-Point Temperature")
g4.legend.background_fill_alpha=0.5
g4.legend.location='top_left'
g4.legend.click_policy='hide'

def T1_calc(dims, temps, wanted_temp, mat, time_range): #calculating outer wall temp
    T_bulk = temps # degrees C of air surrounding outside
    Tc = wanted_temp  # degrees C of inner chamber
    m = 907  # kg of potatoes in a metric ton
    hr = 9  # heat of respiration of potatoes in ml CO2 per kg hr
    rate = 122  # kcal per metric ton * day respiration multiplied to get rate
    k_sand = 0.27  # thermal conductivity of dry sand W/mK
    k_water = 0.6  # thermal conductivity of water W/mK
    e_sand = 0.343  # porosity of sand
    k_ws = e_sand * k_water + (1 - e_sand) * k_sand  # calculates the thermal conductivity of wet sand
    L0 = dims[0] # length of inner chamber
    L1 = .1125  # length of inner brick layer
    L2 = dims[3]  # length of sand layer
    L3 = .1125  # length of outer brick layer
    w0 = dims[1] # width of inner chamber
    h0 = dims[2]  # height of every layer in meters
    A_chamber = L0*h0*2 + w0*h0*2
    A_innerbrick = (L0+L1)*h0*2 + (w0+L1)*h0*2
    A_sand = (L0+L1+L2)*h0*2 + (w0+L1+L2)*h0*2
    h1 = 50  # convective heat transfer coefficient of inner chamber air
    h2 = 5  # convective heat transfer coefficient of outside air
    cond=0
    if mat =="Brick":
        cond=0.72
    elif mat=="Wood":
        cond=12.5 
    elif mat=='Terracotta':
        cond=7.0 
    elif mat=='Concrete':
        cond=0.8
    # calculations
    q = hr * rate * 4.18 * (1 / 24) * (1 / 3600) * m/1000 * 1000  # total respiration rate of one metric ton of potatoes - in J/sec
    T4 = -((q * (1 / (h1*A_chamber))) - Tc)
    T3 = -((q * (L1 / (cond*A_innerbrick))) - T4)
    T2 = -((q * (L2 / (k_ws*A_sand))) - T3)
    T1=[]
    for i in time_range:
        abc = (((L3 * h2 * T_bulk.iloc[i]) / k_brick) + T2) / (1 + (L3 * h2) / k_brick)
        T1.append(abc)
    #print(T1)
    return T1

Costa_T1=T1_calc(initial_dims, yearly_temps_df.iloc[2], 18, "Brick", range(0,12))
sourceDP.data=dict(time=time_range1, temps=yearly_temps_df.iloc[2], dp=dp_Costa, T1=Costa_T1)
gl3=g4.line('time', 'T1', source=sourceDP, legend_label="Outer Wall Temperature", line_width=2, line_dash=[8,2], color='purple')

#Adding display of value for when each line is hovered over at specific point
h1=HoverTool(tooltips=[("Temp","@temps" )], renderers=[gl1])
h2=HoverTool(tooltips=[("Temp", "@dp")], renderers=[gl2])
h3=HoverTool(tooltips=[("Temp", "@T1")], renderers=[gl3])
g4.add_tools(h1, h2, h3)

def update_data(attr, old, new): #when slider or drop down menu values get adjusted, this function is called and recalculates for all the values that all the graphs display
    #Get Slider Values
    length=slide_length.value
    height=slide_height.value
    width=slide_width.value
    mat=select_material.value
    thick=slide_thick.value
    want_temp=slide_desired_temp.value
    location=location_select.value
    time=time_select.value
    cond=0

    if mat =="Brick": #selectng conductivity value based off of material selected
        cond=0.72
    elif mat=="Wood":
        cond=12.5 
    elif mat=='Terracotta':
        cond=7.0 
    elif mat=='Concrete':
        cond=0.8
        
    if time=="12 Months":  #different functions used for calculations depending on if time scale is 24 hours or 12 months
        dims=[length, width, height, thick]
        out=calc_HC(yearly_temps_df.loc[location], dims, cond, want_temp)
        vap=[]
        for p in yearly_temps_df.loc[location]:
            vap.append(SVP(p))
        #recalculating values
        water=water_needed(dims, yearly_temps_df.loc[location], vap, yearly_rh_df.loc[location])
        latent=latent_heat(yearly_temps_df.loc[location])
        evap=evap_cool(water, latent, time_range1)
        dp=dew_point(yearly_temps_df.loc[location], yearly_rh_df.loc[location], range(0,12))
        T1=T1_calc(dims, yearly_temps_df.loc[location], want_temp, mat, range(0,12))
        #updating data source values for what to display
        source.data=dict(time=time_range1, output=out)
        sourceW.data=dict(time=time_range1, temps=yearly_temps_df.loc[location], water=water)
        source3.data=dict(time=time_range1, evap_out=evap)
        sourceDP.data=dict(time=time_range1, temps=yearly_temps_df.loc[location], dp=dp, T1=T1)
        g1.extra_y_ranges['second'].start=np.min(source3.data['evap_out'])-10000
        g1.extra_y_ranges['second'].end=np.max(source3.data['evap_out'])+10000
        g1.y_range.start=np.min(source.data['output'])-10000
        g1.y_range.end=np.max(source.data['output'])+10000
        g1.xaxis.axis_label="Time (in Months)"
        #g3.xaxis.axis_label="Time (in Months)"
        g4.xaxis.axis_label="Time (in Months)"
        
    elif time=="24 Hours":  #different functions used for calculations depending on if time scale is 24 hours or 12 months
        dims=[length, width, height, thick]
        out=HC_hourly(daily_temps_df.loc[location], dims, cond, want_temp)
        vap=[]
        for p in daily_temps_df.loc[location]:
            vap.append(SVP(p))
        #recalculating values
        water=water_needed_hourly(dims, daily_temps_df.loc[location], vap, daily_rh.loc[location])
        latent=latent_heat(daily_temps_df.loc[location])
        evap=evap_cool_hourly(water, latent, time_range)
        T1=T1_calc(dims, daily_temps_df.loc[location], want_temp, mat, range(0,24))
        dp=dew_point_hourly(daily_temps_df.loc[location], daily_rh.loc[location], range(0,24))
        #updating data source values for what to display
        source.data=dict(time=time_range, output=out)
        sourceW.data=dict(time=time_range, temps=daily_temps_df.loc[location], water=water)
        source3.data=dict(time=time_range, evap_out=evap)
        sourceDP.data=dict(time=time_range, temps=daily_temps_df.loc[location], dp=dp, T1=T1)
        g1.extra_y_ranges['second'].start=np.min(source3.data['evap_out'])-10
        g1.extra_y_ranges['second'].end=np.max(source3.data['evap_out'])+10
        g1.y_range.start=np.min(source.data['output'])-10
        g1.y_range.end=np.max(source.data['output'])+10
        g1.xaxis.axis_label="Time (in Hours)"
        #g3.xaxis.axis_label="Time (in Hours)"
        g4.xaxis.axis_label="Time (in Hours)"

def button_updates(): #when calculate button is pressed, this function re-calculates all the info that is spit out
    #Get Slider Values
    length=slide_length.value
    height=slide_height.value
    width=slide_width.value
    mat=select_material.value
    thick=slide_thick.value
    location=location_select.value
    interval=time_select.value
    #place=CostaRica
    dims=[length, width, height, thick]
    water=0
    price=0
    if interval=="12 Months": #different functions used for calculations depending on if time scale is 24 hours or 12 months
        vap=[]
        for p in yearly_temps_df.loc[location]:
            vap.append(SVP(p))
        #recalculating values
        water=water_needed(dims, yearly_temps_df.loc[location], vap, yearly_rh_df.loc[location])
        price=cost_calc(dims, sum(water), mat)
        tablePriceY.append("$"+str(round(price, 2)))
        tablePriceD.append("$"+str(round((price/365), 2)))
        tableWaterY.append(str(round(sum(water), 2))+" L")
        tableWaterD.append(str(round(sum(water)/365, 2))+" L")
        tableTime.append("12 Months")
        
    elif interval=="24 Hours":  #different functions used for calculations depending on if time scale is 24 hours or 12 months
        vap1=[]
        for p in daily_temps_df.loc[location]:
            vap1.append(SVP(p))
        #recalculating values
        water=water_needed_hourly(dims, daily_temps_df.loc[location], vap1, daily_rh.loc[location])
        price=cost_calc(dims, sum(water), mat)
        tablePriceD.append("$"+str(round(price/365,2)))
        tablePriceY.append("$"+str(round(price, 2)))
        tableWaterD.append(str(round(sum(water), 2))+" L")
        tableWaterY.append(str(round(sum(water)*365, 2))+ " L")
        tableTime.append("24 Hours")
    
    tableName.append(location) #changing location name in data table
    tableSpace.append(str(round((dims[0]*dims[1]*dims[2]), 2))+" m^3") #calculating chamber volume for data table
    #updating values that will be displayed in data table
    sourceTable.data=dict(name=tableName, time=tableTime, Year_Price=tablePriceY, Day_Price=tablePriceD, Day_Water=tableWaterD, Year_Water=tableWaterY, space=tableSpace)
    
    
#Information that will appear as text paragraphs 
#p_Heat=Paragraph(text="Note:    Heat per Time: Displays the heat transferred (from evaporation or conduction) per unit time. The heat conducted refers to the heat transferred from the inner chamber to the water. The evaporative cooling rate refers to the rate of heat leaving the system through evaporation. Water Used: Displays the water needed to keep the system running properly, based off the amount of water evaporating at a given time. A system at steady state needs to release the same amount of what it takes in.", 
#                 margin=(20, 10, 20, 10), width=700)
#p_ZECC=Paragraph(text="Zero Energy Cooling: By using the principles behind perspiration, there is a way to create an eco friendly chamber for storing food in harsh conditions. The two chamber cooling system consists of two nested chambers, with sand filling the space in between. Water is set to flow in the sand layer. ", 
#                 margin=(20, 10, 20, 10), width=700)
#p_HT=Paragraph(text="Heat Transfer in the ZECC: The heat transfer that occurs in the zero energy cooling chamber, is a combination of all three of the heat transfer methods. The radiation from solar energy heats the chamber and the surrounding area. The ground also radiates heat. The fluid flow and the conduction of the water is what helps to cool the chamber down.",
#               margin=(20, 10, 20, 10), width=700)
#p_LHV=Paragraph(text="Latent Heat of Vaporization: When one mole of a substance at atmospheric pressure goes from the liquid phase to the gaseous phase, there is energy required to bring the substance to a boil and make the phase change occur. Bringing a substance to its boiling point is not enough since there is still energy required to make phase change occur. This energy required is the latent heat of vaporization. Temperature changes can’t occur without phase changes.",
#                margin=(20, 10, 20, 10), width=700)
#p_dp=Paragraph(text="Note:    Dew-Point temperature is critically dependent on both the design of the chamber and inputed values. If the temperature of the outer wall of the chamber becomes too low then water will begin to condense on the surface and no evaporation will occur, halting the cooling process of the inner chamber.", 
#               margin=(20, 10, 20, 10), width=700)

left_page_spacer = Spacer(width = 20)
top_page_spacer = Spacer(height = 20)
height1 = Spacer(height=20)
height2 = Spacer(height=20)

#organizing display
widgets=column(location_select, time_select, select_material, slide_length, slide_height, slide_width, slide_thick, slide_desired_temp, calculate_button)
selecters=column(location_select, time_select, select_material)
sliders=column(slide_length, slide_height, slide_width, slide_thick, slide_desired_temp)

#organizing panels of diaply
tab2=TabPanel(child=row(left_page_spacer, column(top_page_spacer, row(diff_temps, left_page_spacer, hourly_temps), height1, row(humid, left_page_spacer, mapp))), title="Climate Data")
tab1=TabPanel(child=row(left_page_spacer, column(top_page_spacer, row(selecters, left_page_spacer, sliders), height1, row(g4, left_page_spacer, g1), height2, calculate_button, data_table)), title="Heat Transfer & Essential Temps")
tabs=Tabs(tabs=[tab1, tab2])

updates=[location_select, time_select, select_material, slide_length, slide_height, slide_width, slide_thick, slide_desired_temp]
for u in updates: #when any of the slider values are changed or drop down menu has a new selection, this will then call the update_data function which will then make appropriate adjustments to graphs and table
    u.on_change('value', update_data)
    
calculate_button.on_click(button_updates) #calls button_updates function when the calculate button is clicked

#generated output
curdoc().add_root(tabs)
curdoc().title="Zero Energy Cooling Chamber"

