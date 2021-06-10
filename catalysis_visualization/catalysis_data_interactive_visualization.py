import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, Div, Select, Slider, TextInput, BoxSelectTool, LassoSelectTool
from bokeh.plotting import figure, curdoc

# Import dataset
df_catalysis_dataset = pd.read_csv("data/OCM-data.csv", index_col=0, header=0)

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
slider_methane_conversion = Slider(
    title="Minimum Methane conversion value", value=20, start=1, end=46, step=1)
slider_C2y = Slider(title="Minimum value of C2y",
                    start=0.06, end=21.03, value=4.0, step=0.1)
slider_temp = Slider(title="Minimum value of Temperature",
                     start=700.0, end=900.0, value=800.0, step=50.0)
select_ch4_to_o2 = Select(title="CH4 to O2", options=sorted(
    sorted_unique_ch4_to_o2.keys()), value="6")
select_x_axis = Select(title="X Axis", options=sorted(
    axis_map_x.keys()), value="Ethane_y")
select_y_axis = Select(title="Y Axis", options=sorted(
    axis_map_y.keys()), value="CarbonDiOxide_y")

TOOLTIPS = [
    ("M1 Percent", "@M1_mol_percent"),
    ("M2 Percent", "@M2_mol_percent"),
    ("M3 Percent", "@M3_mol_percent")
]

# tools in the toolbar
TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset"

# Create Column Data Source that will be used by the plot
source = ColumnDataSource(
    data=dict(x=[], y=[], M1_mol_percent=[], M2_mol_percent=[], M3_mol_percent=[]))

p = figure(height=600, width=700, title="", tools=TOOLS, toolbar_location="above",
           tooltips=TOOLTIPS, sizing_mode="scale_both")
p.select(BoxSelectTool).select_every_mousemove = False
p.select(LassoSelectTool).select_every_mousemove = False
p.circle(x="x", y="y", source=source, size=7,
         color='mediumblue', line_color=None, fill_alpha=0.6)

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

# Repositioned the todo 
# TODO: create the horizontal histogram
hhist, hedges = np.histogram(select_data()[axis_map_x[select_x_axis.value]], bins=20)
hzeros = np.zeros(len(hedges)-1)
hmax = max(hhist)*1.1

LINE_ARGS = dict(color="#3A5785", line_color=None)

ph = figure(toolbar_location=None, width=p.width, height=200, x_range=p.x_range,
            y_range=(-hmax, hmax), min_border=10, min_border_left=50, y_axis_location="right")
ph.xgrid.grid_line_color = None
ph.yaxis.major_label_orientation = np.pi/4
ph.background_fill_color = "#fafafa"

ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color="white", line_color="#3A5785")
hh1 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, alpha=0.5, **LINE_ARGS)
hh2 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, alpha=0.1, **LINE_ARGS)

# TODO: create the vertical histogram
vhist, vedges = np.histogram(select_data()[axis_map_y[select_y_axis.value]], bins=10)
vzeros = np.zeros(len(vedges)-1)
vmax = max(vhist)*1.1

pv = figure(toolbar_location=None, width=200, height=p.height, x_range=(-vmax, vmax),
            y_range=p.y_range, min_border=10, y_axis_location="right")
pv.ygrid.grid_line_color = None
pv.xaxis.major_label_orientation = np.pi/4
pv.background_fill_color = "#fafafa"

pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, color="white", line_color="#3A5785")
vh1 = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, alpha=0.5, **LINE_ARGS)
vh2 = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, alpha=0.1, **LINE_ARGS)

layout = gridplot([[p, pv], [ph, None]], merge_tools=False)

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
        M1_mol_percent=df['M1_mol_percentage'],
        M2_mol_percent=df['M2_mol_percentage'],
        M3_mol_percent=df['M3_mol_percentage'],
    )


controls = [slider_methane_conversion, slider_C2y, slider_temp,
            select_ch4_to_o2, select_x_axis, select_y_axis]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

inputs = column(*controls, width=320)

l = column(row(inputs, p), sizing_mode="scale_both")

update()  # initial load of the data

curdoc().add_root(l)
curdoc().title = "Catalysis Data"
