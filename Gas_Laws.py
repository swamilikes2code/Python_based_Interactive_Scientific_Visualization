import numpy as np
from bokeh.io import show, curdoc
from bokeh.models import CustomJS, Dropdown, Slider, ColumnDataSource, Select, RadioButtonGroup, Spacer, HoverTool, WheelZoomTool
from bokeh.layouts import column, row
from bokeh.plotting import figure, show
import pandas as pd

#taskkill /F /IM python.exe
#python -m bokeh serve --show Gas_Laws.py


#dropdown menu to select a substance
substance_choice = Select(title="Select a Substance", value="Ammonia", 
                     options=["Ammonia", "Carbon Dioxide", "Methane", "Methanol", "Nitrogen", "Propane", "Water"])

#hover tool
hover = HoverTool(
        tooltips=[   
            ("Volume (L/mol)", "@x{0.00}"), 
            ("Pressure (MPa)", "@y{0.00}"),
        ],
        mode='vline'
    )

#creates the graph that will be selected based off of the radio button
source = ColumnDataSource(data=dict(x=[], y=[]))
p = figure(title="Gas Law Comparison", height=600, width=600, x_range = [0, 10], y_range = [0,10], tools=[hover, "pan,wheel_zoom,box_zoom,reset,save"])
p.toolbar.active_scroll = p.select_one(dict(type=WheelZoomTool))

#Specific temperature graphs
real_1_source = ColumnDataSource(data=dict(x=[], y=[]))
real_2_source = ColumnDataSource(data=dict(x=[], y=[]))
real_3_source = ColumnDataSource(data=dict(x=[], y=[]))


# Change these lines in your setup section:
p.scatter('x', 'y', source=real_1_source, name = "r1", color="red", fill_alpha=0.3, size=8, marker="circle", legend_field="Real Gas 150K")
p.scatter('x', 'y', source=real_2_source, name = "r2", color="purple", fill_alpha=0.6, size=8, marker="square", legend_field="Real Gas 300K")
p.scatter('x', 'y', source=real_3_source, name = "r3", color="green", fill_alpha=0.9, size=8, marker="triangle", legend_field="Real Gas 450K")



virial_source = ColumnDataSource(data=dict(x=[], y=[]))
p.line('x', 'y', source=source, line_width=3, color="blue", legend_label = "Ideal Gas Law")
p.line('x', 'y', source=virial_source, line_width=3, color="orange", legend_label="Virial Equation of State")

srk_source = ColumnDataSource(data=dict(x=[], y=[]))
p.line('x', 'y', source=srk_source, line_width=3, color="pink", legend_label="SRK Equation of State")

p.legend.click_policy = "hide"

#radio button group to select which graph to view
LABELS = ["Pressure vs. Volume Graph", "Pressure vs. Temperature Graph", "Volume vs. Temperature Graph"]
graph_options = RadioButtonGroup(labels=LABELS, active=0)
graph_options.js_on_change('active', CustomJS(code="""
    console.log('radio_button_group: active=' + cb_obj.active);
"""))

#sliders to adjust the volume, pressure, temperature, and number of moles
volume = Slider(start=10, end=30, value=10, step=1, title="Volume (L/mol)")
temp = Slider(start=150, end=450, value=150, step=10, title="Temperature (K)")
pressure = Slider(start=0.01, end=1, value=.01, step=.01, title="Pressure (MPa)")

#dictionary of the values to be used below
gas_constants = {
    "Nitrogen":         {"Tc": 126.2, "Pc": 3.39, "w": 0.037},
    "Methane":          {"Tc": 190.7, "Pc": 4.64, "w": 0.011},
    "Ammonia":          {"Tc": 405.5, "Pc": 11.28, "w": 0.257},
    "Methanol":         {"Tc": 513.2, "Pc": 7.95, "w": 0.565},
    "Water":            {"Tc": 647.4, "Pc": 22.12, "w": 0.344},
    "Propane":          {"Tc": 369.9, "Pc": 42.0, "w": 0.152},
    "Carbon Dioxide":   {"Tc": 304.2, "Pc": 72.9, "w": 0.225}
}


#update the sliders when adjusted
def update_data(attr, old, new):
    # Get the current slider values
    V = volume.value
    T = temp.value
    P = pressure.value
    #universal gas constant
    R = 0.008314 #J/mol*K

    selection = graph_options.active
        
    # Map selection to the desired names
    if selection == 0:
        labels = {"r1": "Real Gas 150K", "r2": "Real Gas 300K", "r3": "Real Gas 450K"}
    elif selection == 1:
        labels = {"r1": "Real Gas 10 L/mol", "r2": "Real Gas 20 L/mol", "r3": "Real Gas 30 L/mol"}
    else:
        labels = {"r1": "Real Gas 0.01 MPa", "r2": "Real Gas 0.1 MPa", "r3": "Real Gas 1.0 MPa"}

    for item in p.legend.items:
        # Get the name of the renderer associated with this legend item
        r_name = item.renderers[0].name
        if r_name in labels:
            item.label = {'value': labels[r_name]}
            #item.label.value = labels[r_name]
    
    #virial equation constants
    gas = substance_choice.value
    tc = gas_constants[gas]["Tc"]
    pc = gas_constants[gas]["Pc"]
    a = (27 * R**2 * tc**2)/(64 * pc)
    b = (R * tc)/(8 * pc)
    w = gas_constants[gas]["w"]
    m = 0.48508 + 1.55171 * w - 0.1561 * (w**2)
    b_srk = 0.08664 * (R * tc) / pc
    a_srk_const = 0.42747 * (R**2 * tc**2) / pc

    def get_srk_p(T_val, V_val):
        tr = T_val / tc
        alpha = (1 + m * (1 - np.sqrt(tr)))**2
        a_t = alpha * a_srk_const
        return (R * T_val) / (V_val - b_srk) - a_t / (V_val * (V_val + b_srk))

    if(selection == 0):
        p.xaxis.axis_label = "Volume (L/mol)"
        p.yaxis.axis_label = "Pressure (MPa)"
        hover.tooltips = [("Volume (L/mol)", "@x"), ("Pressure (MPa)", "@y")]
        pressure.visible = False
        volume.visible = False
        temp.visible = True
        p.x_range.start = 0
        p.x_range.end = 10 
        p.y_range.start = 0
        p.y_range.end = 2

        x_coords = np.linspace(0.1, 30, 100)
        y_coords = (R * T) / x_coords #pv = nrt


        df150 = pd.read_csv('Real Gas Data/' + substance_choice.value + '/' + substance_choice.value + '_150T.csv').sort_values(by='Volume (l/mol)')
        df300 = pd.read_csv('Real Gas Data/' + substance_choice.value + '/' + substance_choice.value + '_300T.csv').sort_values(by='Volume (l/mol)')
        df450 = pd.read_csv('Real Gas Data/' + substance_choice.value + '/' + substance_choice.value + '_450T.csv').sort_values(by='Volume (l/mol)')

        real_1_source.data = dict(x=df150['Volume (l/mol)'], y=df150['Pressure (MPa)'])
        real_2_source.data = dict(x=df300['Volume (l/mol)'], y=df300['Pressure (MPa)'])
        real_3_source.data = dict(x=df450['Volume (l/mol)'], y=df450['Pressure (MPa)'])
        
        y_virial = (R * T) / (x_coords - b) - (a / x_coords**2)
        y_srk = get_srk_p(T, x_coords)


    if(selection == 1):
        p.xaxis.axis_label = "Temperature (K)"
        p.yaxis.axis_label = "Pressure (MPa)"
        hover.tooltips = [("Temperature (K)", "@x"), ("Pressure (MPa)", "@y")]
        pressure.visible = False
        temp.visible = False
        volume.visible = True

        p.x_range.start = 0
        p.x_range.end = 500  
        
        p.y_range.start = 0
        p.y_range.end = 0.6

        x_coords = np.linspace(0.1, 500, 500)
        y_coords = (R / V) * x_coords #pv = nrt

        df10 = pd.read_csv('Real Gas Data/' + substance_choice.value + '/' + substance_choice.value + '_10V.csv').sort_values(by='Volume (l/mol)')
        df20 = pd.read_csv('Real Gas Data/' + substance_choice.value + '/' + substance_choice.value + '_20V.csv').sort_values(by='Volume (l/mol)')
        df30 = pd.read_csv('Real Gas Data/' + substance_choice.value + '/' + substance_choice.value + '_30V.csv').sort_values(by='Volume (l/mol)')

        real_1_source.data = dict(x=df10['Temperature (K)'], y=df10['Pressure (MPa)'])
        real_2_source.data = dict(x=df20['Temperature (K)'], y=df20['Pressure (MPa)'])
        real_3_source.data = dict(x=df30['Temperature (K)'], y=df30['Pressure (MPa)'])

        y_virial = (R * x_coords) / (V - b) - (a / V**2)
        y_srk = get_srk_p(x_coords, V)
        
    if(selection == 2):
        p.xaxis.axis_label = "Temperature (K)"
        p.yaxis.axis_label = "Volume (L)"
        hover.tooltips = [("Temperature (K)", "@x"), ("Volume (L)", "@y")]
        temp.visible = False
        volume.visible = False
        pressure.visible = True

        p.x_range.start = 0
        p.x_range.end = 450 
        p.y_range.start = 0
        p.y_range.end = 250


        x_coords = np.linspace(0.1, 450, 450)
        y_coords = (R / P) * x_coords #pv = nrt 

        df1 = pd.read_csv('Real Gas Data/' + substance_choice.value + '/' + substance_choice.value + '_0.01P.csv').sort_values(by='Volume (l/mol)')
        df2 = pd.read_csv('Real Gas Data/' + substance_choice.value + '/' + substance_choice.value + '_0.1P.csv').sort_values(by='Volume (l/mol)')
        df3 = pd.read_csv('Real Gas Data/' + substance_choice.value + '/' + substance_choice.value + '_1P.csv').sort_values(by='Volume (l/mol)')

        real_1_source.data = dict(x=df1['Temperature (K)'], y=df1['Volume (l/mol)'])
        real_2_source.data = dict(x=df2['Temperature (K)'], y=df2['Volume (l/mol)'])
        real_3_source.data = dict(x=df3['Temperature (K)'], y=df3['Volume (l/mol)'])

        y_virial = ((R * x_coords) / P) + (b - (a / (R * x_coords)))
        y_srk = ((R * x_coords) / P) + b_srk
       
    source.data = dict(x=x_coords, y=y_coords)
    virial_source.data = dict(x=x_coords, y=y_virial)
    srk_source.data = dict(x=x_coords, y=y_srk)

#updates the radio button and the graph values
graph_options.on_change('active', update_data)

for w in [substance_choice, volume, temp, pressure]:
    w.on_change('value', update_data)

#output/spacing of all of the widgets
space = Spacer(height=140)

left_layout = column(substance_choice, graph_options, p)
right_layout = column(space, volume, temp, pressure)
final_layout = row(left_layout, right_layout)
curdoc().add_root(final_layout)

update_data(None, None, None)

# temperature values: 150, 300, 450, 600, 750
# volume values: 10 1/mol, 20 l/mol, 30 l/mol
# pressure values: 0.01 mPa, 0.1 mPa, 1mPa
