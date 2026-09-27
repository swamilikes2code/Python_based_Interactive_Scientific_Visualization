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
p = figure(
    title="Gas Law Comparison", height=600, width=600,
    x_axis_type="log", y_axis_type="log",
    x_range=[0.1, 35], y_range=[0.05, 25],
    tools=[hover, "pan,wheel_zoom,box_zoom,reset,save"]
)
p.toolbar.active_scroll = p.select_one(dict(type=WheelZoomTool))

#Specific temperature graphs
real_1_source = ColumnDataSource(data=dict(x=[], y=[]))
real_2_source = ColumnDataSource(data=dict(x=[], y=[]))
real_3_source = ColumnDataSource(data=dict(x=[], y=[]))
real_4_source = ColumnDataSource(data=dict(x=[], y=[]))
real_5_source = ColumnDataSource(data=dict(x=[], y=[]))

real_sources = {1: real_1_source, 2: real_2_source, 3: real_3_source, 4: real_4_source, 5: real_5_source,}

substance_ranges = {
    "Ammonia":         {"temps": [410, 425, 475, 525, 575], "pres": [1, 5, 10, 20, 35], "vols": [0.03, 0.1, 0.5, 2, 5]},
    "Carbon Dioxide":  {"temps": [315, 345, 375, 405, 435], "pres": [0.5, 5, 10, 15, 20], "vols": [0.1, 1, 10, 20, 30]},
    "Methane":         {"temps": [195, 210, 225, 250, 275], "pres": [0.5, 2.5, 5, 10, 15], "vols": [0.05, 0.1, 0.5, 2, 4.5]},
    "Methanol":        {"temps": [515, 550, 580, 600, 620], "pres": [0.75, 2, 5, 15, 30], "vols": [0.05, 0.15, 0.8, 3, 8]},
    "Nitrogen":        {"temps": [130, 140, 150, 160, 170], "pres": [0.25, 0.5, 1, 5, 10], "vols": [0.035, 0.1, 0.5, 2, 6]},
    "Propane":         {"temps": [375, 425, 450, 475, 525], "pres": [0.3, 1, 5, 10, 15], "vols": [0.1, 0.5, 2, 7, 14.5]},
    "Water":           {"temps": [650, 700, 800, 860, 930], "pres": [2, 5, 15, 35, 70], "vols": [0.02, 0.1, 0.5, 2, 4]},
}
dfs = {}

#initializing the real gas lines
real_renderers = []
real_renderers.append(p.scatter('x', 'y', source=real_1_source, name="r1", color='#F6C7B3', fill_alpha=0.0, size=4, marker="circle", legend_label="Real Gas 1"))
real_renderers.append(p.scatter('x', 'y', source=real_2_source, name="r2", color='#F0E2C3', fill_alpha=0.0, size=4, marker="square", legend_label="Real Gas 2"))
real_renderers.append(p.scatter('x', 'y', source=real_3_source, name="r3", color='#ACC791', fill_alpha=0.0, size=4, marker="triangle", legend_label="Real Gas 3"))
real_renderers.append(p.scatter('x', 'y', source=real_4_source, name="r4", color='#82B2C0', fill_alpha=0.0, size=4, marker="diamond", legend_label="Real Gas 4"))
real_renderers.append(p.scatter('x', 'y', source=real_5_source, name="r5", color='#809BCE', fill_alpha=0.0, size=4, marker="hex", legend_label="Real Gas 5"))

p.line('x', 'y', source=source, line_width=3, color='#787b74', legend_label = "Ideal Gas Law")

virial_source = ColumnDataSource(data=dict(x=[], y=[]))
p.line('x', 'y', source=virial_source, line_width=3, color='#71351C', legend_label="Virial Equation of State")

srk_source = ColumnDataSource(data=dict(x=[], y=[]))
p.line('x', 'y', source=srk_source, line_width=3, color='#d7883f', legend_label="SRK Equation of State")

p.legend.click_policy = "hide"

#radio button group to select which graph to view
LABELS = ["Pressure vs. Volume Graph", "Pressure vs. Temperature Graph", "Volume vs. Temperature Graph"]
graph_options = RadioButtonGroup(labels=LABELS, active=0)
graph_options.js_on_change('active', CustomJS(code="""
    console.log('radio_button_group: active=' + cb_obj.active);
"""))

#sliders to adjust the volume, pressure, temperature, and number of moles
volume = Slider(start=0.03, end=5, value=0.03, step=0.005, title="Volume (L/mol)", format="0.000")
temp = Slider(start=410, end=575, value=410, step=5, title="Temperature (K)")
pressure = Slider(start=1, end=35, value=1, step=0.1, title="Pressure (MPa)")

#dictionary of the values to be used below
gas_constants = {
    "Nitrogen":         {"Tc": 126.2, "Pc": 3.39, "w": 0.037},
    "Methane":          {"Tc": 190.7, "Pc": 4.64, "w": 0.011},
    "Ammonia":          {"Tc": 405.5, "Pc": 11.28, "w": 0.257},
    "Methanol":         {"Tc": 513.2, "Pc": 7.95, "w": 0.565},
    "Water":            {"Tc": 647.4, "Pc": 22.12, "w": 0.344},
    "Propane":          {"Tc": 369.9, "Pc": 4.20, "w": 0.152},
    "Carbon Dioxide":   {"Tc": 304.2, "Pc": 7.29, "w": 0.225}
}


#update the sliders when adjusted
def update_data(attr, old, new):
    ranges = substance_ranges[substance_choice.value]
    temps = ranges["temps"]
    pres = ranges["pres"]
    vols = ranges["vols"]
    #universal gas constant
    R = 0.008314 #J/mol*K

    vol_min, vol_max = min(vols), max(vols)
    pres_min, pres_max = min(pres), max(pres)
    temp_min, temp_max = min(temps), max(temps)

    volume.start, volume.end = vol_min, vol_max
    pressure.start, pressure.end = pres_min, pres_max
    temp.start, temp.end = temp_min, temp_max

    if substance_choice.value != update_data.last_substance:
        volume.value = vol_min
        temp.value = temp_min
        pressure.value = pres_min
        update_data.last_substance = substance_choice.value

     # Get the current slider values
    V = volume.value
    T = temp.value
    P = pressure.value

    selection = graph_options.active
        
    # Map selection to the desired names
    if selection == 0:
        labels = {f"r{i+1}": f"Real Gas {temps[i]}K" for i in range(len(temps))}
        dot_size = 4
    elif selection == 1:
        labels = {f"r{i+1}": f"Real Gas {vols[i]}L/mol" for i in range(len(vols))}
        dot_size = 6
    else:
        labels = {f"r{i+1}": f"Real Gas {pres[i]}MPa" for i in range(len(pres))}
        dot_size = 8

    for r in real_renderers:
        r.glyph.size = dot_size


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

        p.x_range.start = vol_min
        p.x_range.end = vol_max
        p.y_range.start = pres_min
        p.y_range.end = pres_max
        if(substance_choice.value == "Carbon Dioxide"):
            p.x_range.start = 0.04
            p.y_range.end = 30
            p.x_range.end = 8

        x_coords = np.linspace(0.0001, vol_max, 500)
        y_coords = (R * T) / x_coords #pv = nrt

        count = 1
        for t_val in temps:
            dfs[t_val] = pd.read_csv(f'Real Gas Data/{substance_choice.value}/{substance_choice.value}_{t_val}T.csv').sort_values(by='Volume (l/mol)')
            real_sources[count].data = dict(x=dfs[t_val]['Volume (l/mol)'], y=dfs[t_val]['Pressure (MPa)'])
            count += 1
        
        y_virial = (R * T) / (x_coords - b) - (a / x_coords**2)
        y_srk = get_srk_p(T, x_coords)

    if(selection == 1):
        p.xaxis.axis_label = "Temperature (K)"
        p.yaxis.axis_label = "Pressure (MPa)"
        hover.tooltips = [("Temperature (K)", "@x"), ("Pressure (MPa)", "@y")]
        pressure.visible = False
        temp.visible = False
        volume.visible = True


        x_coords = np.linspace(0.0001, 1000, 5000)
        y_coords = (R / V) * x_coords #pv = nrt

        count = 1
        for v_val in vols:
            dfs[v_val] = pd.read_csv(f'Real Gas Data/{substance_choice.value}/{substance_choice.value}_{v_val}V.csv').sort_values(by='Temperature (K)')
            real_sources[count].data = dict(x=dfs[v_val]['Temperature (K)'], y=dfs[v_val]['Pressure (MPa)'])
            count += 1

        all_data = pd.concat([dfs[v_val] for v_val in vols])
        p.x_range.start = all_data['Temperature (K)'].min()/1.05
        p.x_range.end = all_data['Temperature (K)'].max()*1.05
        p.y_range.start = all_data['Pressure (MPa)'].min()/1.2
        p.y_range.end = all_data['Pressure (MPa)'].max()*2


        y_virial = (R * x_coords) / (V - b) - (a / V**2)
        y_srk = get_srk_p(x_coords, V)
        
    if(selection == 2):
        p.xaxis.axis_label = "Temperature (K)"
        p.yaxis.axis_label = "Volume (L)"
        hover.tooltips = [("Temperature (K)", "@x"), ("Volume (L)", "@y")]
        temp.visible = False
        volume.visible = False
        pressure.visible = True

        x_coords = np.linspace(0.0001, 1000, 5000)
        y_coords = (R / P) * x_coords #pv = nrt 

        count = 1
        for p_val in pres:
            dfs[p_val] = pd.read_csv(f'Real Gas Data/{substance_choice.value}/{substance_choice.value}_{p_val}P.csv').sort_values(by='Temperature (K)')
            real_sources[count].data = dict(x=dfs[p_val]['Temperature (K)'], y=dfs[p_val]['Volume (l/mol)'])
            count += 1

        all_data = pd.concat([dfs[p_val] for p_val in pres])
        p.x_range.start = all_data['Temperature (K)'].min()/1.05
        p.x_range.end = all_data['Temperature (K)'].max()*1.05
        p.y_range.start = all_data['Volume (l/mol)'].min()/1.2
        p.y_range.end = all_data['Volume (l/mol)'].max()*2

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

update_data.last_substance = substance_choice.value
update_data(None, None, None)

