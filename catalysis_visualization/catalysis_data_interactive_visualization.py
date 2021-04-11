import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Div, Select, Slider, TextInput
from bokeh.plotting import figure

# Import dataset
dir_path = '.\\data\\'
data_filename = 'OCM-data.csv'
df_catalysis_dataset = pd.read_csv(dir_path+data_filename)

# Determine key values for Select Tool. More details in the Notebook.

unique_temp = (df_catalysis_dataset['Temp']
 .sort_values()
 .astype(str)
 .unique()
)
sorted_unique_temp = dict(zip(unique_temp, unique_temp))

unique_ch4_to_o2 = (df_catalysis_dataset['CH4/O2']
 .sort_values()
 .astype(str)
 .unique()
)
sorted_unique_ch4_to_o2 = dict(zip(unique_ch4_to_o2, unique_ch4_to_o2))

axis_map_x = {
    "Ethane_y": "C2H6y",
    "Ethylene_y": "C2H4y",
    "CarbonDiOxide_y": "CO2y",
    "CarbonMonoOxide_y": "COy",
    "DiCarbon_s": "C2s",
    "Ethane_s": "C2H6s",
    "Ethylene_s": "C2H4s",
    "CarbonDiOxide_s": "CO2s",
    "CarbonMonoOxide_s": "COs",
}

axis_map_y = {
    "Ethane_y": "C2H6y",
    "Ethylene_y": "C2H4y",
    "CarbonDiOxide_y": "CO2y",
    "CarbonMonoOxide_y": "COy",
    "DiCarbon_s": "C2s",
    "Ethane_s": "C2H6s",
    "Ethylene_s": "C2H4s",
    "CarbonDiOxide_s": "CO2s",
    "CarbonMonoOxide_s": "COs",
}

# Create Input controls
slider_methane_conversion = Slider(title="Minimum Methane conversion value", 
                                   value=20, start=1, end=46, step=1)
slider_C2y = Slider(title="Minimum value of C2y", start=0.1, end=22.1, value=4.0, step=0.1)
slider_temp = Slider(title="Minimum value of Temperature", start=700.0, end=900.0, value=800.0, step=50.0)
select_ch4_to_o2 = Select(title="CH4 to O2", options=sorted(sorted_unique_ch4_to_o2.keys()), value="6")
select_x_axis = Select(title="X Axis", options=sorted(axis_map_x.keys()), value="Ethane_y")
select_y_axis = Select(title="Y Axis", options=sorted(axis_map_y.keys()), value="CarbonDiOxide_y")

TOOLTIPS=[
    ("M1 %", "@M1_mol%"),
    ("M2 %", "@M2_mol%"),
    ("M3 %", "@M3_mol%")
]

# Create Column Data Source that will be used by the plot
source = ColumnDataSource(data=dict(x=[], y=[]))

p = figure(height=600, width=700, title="", toolbar_location=None, tooltips=TOOLTIPS, sizing_mode="scale_both")
p.circle(x="x", y="y", source=source, size=7, color='mediumblue', line_color=None, fill_alpha=0.6)

def select_data():
    temp_val = slider_temp.value
    select_ch4_to_o2_val = select_ch4_to_o2.value
    selected = df_catalysis_dataset[
        (df_catalysis_dataset.CH4_conv >= slider_methane_conversion.value) &
        (df_catalysis_dataset.C2y >= slider_C2y.value) &
        (df_catalysis_dataset.Temp >= float(slider_temp.value)) &
        (df_catalysis_dataset['CH4/O2'] == float(select_ch4_to_o2.value))
    ]
    return selected

def update():
    df = select_data()
    x_name = axis_map_x[select_x_axis.value]
    y_name = axis_map_y[select_y_axis.value]

    p.xaxis.axis_label = select_x_axis.value
    p.yaxis.axis_label = select_y_axis.value
    p.title.text = 'Title TBD'
    source.data = dict(
        x=df[x_name],
        y=df[y_name],
    )

controls = [slider_methane_conversion, slider_C2y, slider_temp, select_ch4_to_o2, select_x_axis, select_y_axis]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())
    
inputs = column(*controls, width=320)

l = column(row(inputs, p), sizing_mode="scale_both")

update()  # initial load of the data

curdoc().add_root(l)
curdoc().title = "Catalysis Data"
