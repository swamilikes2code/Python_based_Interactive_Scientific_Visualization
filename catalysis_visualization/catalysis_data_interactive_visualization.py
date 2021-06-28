import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, Select, Slider, BoxSelectTool, LassoSelectTool, Tabs, Panel, LinearColorMapper, ColorBar, BasicTicker, PrintfTickFormatter, MultiSelect, DataTable, TableColumn
from bokeh.plotting import figure, curdoc
from bokeh.palettes import viridis, gray, cividis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# Import dataset
df_catalysis_dataset = pd.read_csv("catalysis_visualization/data/OCM-data.csv", index_col=0, header=0)

# Removing the Blank names from the data
df_catalysis_dataset.set_index(df_catalysis_dataset.index)
df_catalysis_dataset.drop("Blank", axis=0)

# Calculating error percentage

# Sum of columns to compare with CH4_conv
df_catalysis_dataset["Sum_y"] = df_catalysis_dataset.loc[:,
                                                         "C2H6y":"CO2y"].sum(axis=1)
df_catalysis_dataset["error_ch4_conv"] = abs((df_catalysis_dataset["Sum_y"]-df_catalysis_dataset["CH4_conv"]) /
                                             df_catalysis_dataset["CH4_conv"])*100

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
    "Argon Flow": "Ar_flow",
    "Methane Flow": "CH4_flow",
    "Oxygen Flow": "O2_flow",
    "Amount of Catalyst": "CT",
    "M2_mol": "M2_mol",
    "M3_mol": "M3_mol",
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
slider_C2y = Slider(title="Minimum value of C2y",
                    start=0.06, end=21.03, value=4.0, step=0.1)
slider_temp = Slider(title="Minimum value of Temperature",
                     start=700.0, end=900.0, value=800.0, step=50.0)
slider_error = Slider(title="Maximum Error Permitted",
                      start=0.0, end=100.0, step=0.5, value=37.0)
select_ch4_to_o2 = Select(title="CH4 to O2",
                          options=sorted(sorted_unique_ch4_to_o2.keys()),
                          value="6")
select_x_axis = Select(title="X Axis",
                       options=sorted(axis_map_x.keys()),
                       value="Argon Flow")
select_y_axis = Select(title="Y Axis",
                       options=sorted(axis_map_y.keys()),
                       value="CarbonDiOxide_y")

TOOLTIPS = [
    ("M1", "@M1"),
    ("M2", "@M2"),
    ("M3", "@M3"),
    ("Catalyst/Support", "@Name")
]

# tools in the toolbar
TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset,box_zoom,undo,redo"

# Create Column Data Source that will be used by the plot
source = ColumnDataSource(data=dict(x=[], y=[], M1=[], M2=[], M3=[], Name=[]))

p = figure(height=600, width=700, title="Data Exploration", tools=TOOLS,
           toolbar_location="above", tooltips=TOOLTIPS)
p.select(BoxSelectTool).select_every_mousemove = False
p.select(LassoSelectTool).select_every_mousemove = False
r = p.circle(x="x", y="y", source=source, size=7,
             color='mediumblue', line_color=None, fill_alpha=0.6)
# r = p.scatter(x = "x",y="y",alpha=0.3)


def select_data():
    temp_val = slider_temp.value
    select_ch4_to_o2_val = select_ch4_to_o2.value
    selected = df_catalysis_dataset[
        (df_catalysis_dataset.CH4_conv >= slider_methane_conversion.value) &
        (df_catalysis_dataset.C2y >= slider_C2y.value) &
        (df_catalysis_dataset.Temp >= float(slider_temp.value)) &
        (df_catalysis_dataset.error_ch4_conv <= float(slider_error.value)) &
        (df_catalysis_dataset['CH4/O2'] == float(select_ch4_to_o2.value))
    ]
    return selected


# the horizontal histogram
hhist, hedges = np.histogram(
    select_data()[axis_map_x[select_x_axis.value]], bins=10)
hzeros = np.zeros(len(hedges)-1)

LINE_ARGS = dict(color="#3A5785", line_color=None)

ph = figure(toolbar_location=None, width=p.width, height=100, x_range=p.x_range,
            y_range=(0, (max(hhist)*1.1)), min_border=10, min_border_left=50, y_axis_location="right")
ph.xgrid.grid_line_color = None
ph.yaxis.major_label_orientation = np.pi/4
ph.background_fill_color = "#fafafa"

# histogram to reflect the data points
hh = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:],
             top=hhist, color="white", line_color="#3A5785")
# histograms highlight on top of the original histogram
hh1 = ph.quad(
    bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, alpha=0.5, **LINE_ARGS)
hh2 = ph.quad(
    bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, alpha=0.1, **LINE_ARGS)

# the vertical histogram
vhist, vedges = np.histogram(
    select_data()[axis_map_y[select_y_axis.value]], bins=10)
vzeros = np.zeros(len(vedges)-1)

pv = figure(toolbar_location=None, width=100, height=p.height,
            x_range=(0, (max(vhist)*1.1)), y_range=p.y_range, min_border=10, y_axis_location="right")
pv.ygrid.grid_line_color = None
pv.xaxis.major_label_orientation = np.pi/4
pv.background_fill_color = "#fafafa"

# histogram to reflect the data points
vv = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:],
             right=vhist, color="white", line_color="#3A5785")
# histograms highlight on top of the original histogram
vh1 = pv.quad(
    left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, alpha=0.5, **LINE_ARGS)
vh2 = pv.quad(
    left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, alpha=0.1, **LINE_ARGS)

layout = gridplot([[p, pv], [ph, None]], merge_tools=True)


# Brought in update for the histogram selections attempt
def update():
    df = select_data()
    x_name = axis_map_x[select_x_axis.value]
    y_name = axis_map_y[select_y_axis.value]
    p.xaxis.axis_label = select_x_axis.value
    p.yaxis.axis_label = select_y_axis.value
    source.data = dict(
        x=df[x_name],
        y=df[y_name],
        M1=df['M1'],
        M2=df['M2'],
        M3=df['M3'],
        Name=df.index
    )

    # also update both histograms
    global hhist, hedges, vhist, vedges
    if len(df) == 0:
        hhist, hedges = hzeros, hzeros
        vedges, vhist = vzeros, vzeros
        hh.data_source.data["top"] = hzeros
        vv.data_source.data["right"] = vzeros
    else:
        hhist, hedges = np.histogram(
            df[axis_map_x[select_x_axis.value]], bins=10)
        vhist, vedges = np.histogram(
            df[axis_map_y[select_y_axis.value]], bins=10)
        ph.y_range.end = max(hhist)*1.1
        pv.x_range.end = max(vhist)*1.1
        hh.data_source.data["top"] = hhist
        hh.data_source.data["right"] = hedges[1:]
        hh1.data_source.data["right"] = hedges[1:]
        # hh2.data_source.data["right"] = hedges[1:]
        hh.data_source.data["left"] = hedges[:-1]
        hh1.data_source.data["left"] = hedges[:-1]
        # hh2.data_source.data["left"] = hedges[:-1]
        vv.data_source.data["right"] = vhist
        vv.data_source.data["bottom"] = vedges[:-1]
        vh1.data_source.data["bottom"] = vedges[:-1]
        # vh2.data_source.data["bottom"] = vedges[:-1]
        vv.data_source.data["top"] = vedges[1:]
        vh1.data_source.data["top"] = vedges[1:]
        # vh2.data_source.data["top"] = vedges[1:]


controls = [slider_methane_conversion, slider_C2y, slider_temp,
            slider_error, select_ch4_to_o2, select_x_axis, select_y_axis]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

inputs = column(*controls, width=320)


def update_histogram(attr, old, new):
    inds = new
    if len(inds) == 0 or len(inds) == len(select_data()[axis_map_x[select_x_axis.value]]):
        hhist1, hhist2 = hzeros, hzeros
        vhist1, vhist2 = vzeros, vzeros
    else:
        neg_inds = np.ones_like(
            select_data()[axis_map_x[select_x_axis.value]], dtype=np.bool)
        neg_inds[inds] = False
        hhist1, _ = np.histogram(
            select_data()[axis_map_x[select_x_axis.value]][inds], bins=hedges)
        vhist1, _ = np.histogram(
            select_data()[axis_map_y[select_y_axis.value]][inds], bins=vedges)
        # hhist2, _ = np.histogram(
        #     select_data()[axis_map_x[select_x_axis.value]][neg_inds], bins=hedges)
        # vhist2, _ = np.histogram(
        #     select_data()[axis_map_y[select_y_axis.value]][neg_inds], bins=vedges)

    hh1.data_source.data["top"] = hhist1
    # hh2.data_source.data["top"] = -hhist2
    vh1.data_source.data["right"] = vhist1
    # vh2.data_source.data["right"] = -vhist2


visualization_layout = column([row(inputs, layout)], sizing_mode="scale_both")


# Adding the correlation matrix
# Copy x-axis values into new df
df_corr = df_catalysis_dataset[
    ["CT", "Ar_flow", "CH4_flow", "O2_flow", "Total_flow", "Support_ID", "Temp",
     "M2_mol", "M3_mol", "M1_atom_number", "M2_atom_number", "M3_atom_number",
     "M1_mol_percentage", "M2_mol_percentage", "M3_mol_percentage"]
]
corr_matrix = df_corr.corr()

# AXIS LABELS FOR PLOT
df_corr = pd.DataFrame(corr_matrix)
df_corr = df_corr.set_index(df_corr.columns).rename_axis('parameters', axis=1)
df_corr.index.name = 'level_0'
common_axes_val = list(df_corr.index)
df_corr = pd.DataFrame(df_corr.stack(), columns=['correlation']).reset_index()
source_corr = ColumnDataSource(df_corr)

# FINDING LOWEST AND HIGHEST OF CORRELATION VALUES
low_df_corr_min = df_corr.correlation.min()
high_df_corr_min = df_corr.correlation.max()
no_of_colors = 7

# PLOT PARTICULARS
# CHOOSING DEFAULT COLORS
COLOR_SCHEME = {
    'Cividis': cividis(no_of_colors),
    'Gray': gray(no_of_colors),
    'Viridis': viridis(no_of_colors),
}

select_color = Select(title='Color Palette', value='Cividis',
                      options=list(COLOR_SCHEME.keys()), width=200, height=50)

mapper = LinearColorMapper(palette=cividis(no_of_colors),
                           low=low_df_corr_min, high=high_df_corr_min)

# SETTING UP THE PLOT
c_corr = figure(title="Correlation Matrix", x_range=common_axes_val, y_range=list((common_axes_val)), x_axis_location="below", toolbar_location=None,
                plot_width=700, plot_height=600, tooltips=[('Parameters', '@level_0 - @parameters'), ('Correlation', '@correlation')])


# SETTING UP PLOT PROPERTIES
c_corr.grid.grid_line_color = None
c_corr.axis.axis_line_color = None
c_corr.axis.major_tick_line_color = None
c_corr.axis.major_label_text_font_size = "10pt"
c_corr.xaxis.major_label_orientation = np.pi/2

# SETTING UP HEATMAP RECTANGLES
cir = c_corr.rect(x="level_0", y="parameters", width=1, height=1, source=source_corr,
                  fill_color={'field': 'correlation', 'transform': mapper}, line_color=None)

# SETTING UP COLOR BAR
color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt", ticker=BasicTicker(desired_num_ticks=10),
                     formatter=PrintfTickFormatter(format="%.1f"), label_standoff=6, border_line_color=None, location=(0, 0))
c_corr.add_layout(color_bar, 'right')


def change_color():
    mapper.palette = COLOR_SCHEME[select_color.value]
    cir.glyph.fill_color = {'field': 'correlation', 'transform': mapper}
    color_bar.color_mapper = mapper


select_color.on_change('value', lambda attr, old, new: change_color())


# REGRESSION MODEL

# Selection tools
reg_x_choices = {
    "M1 atom number": "M1_atom_number",
    "M2 atom number": "M2_atom_number",
    "M3 atom number": "M3_atom_number",
    "Support Id": "Support_ID",
    "M2 mol": "M2_mol",
    "M3 mol": "M3_mol",
    "M1 percent mol": "M1_mol_percentage",
    "M2 percent mol": "M2_mol_percentage",
    "M3 percent mol": "M3_mol_percentage",
    "Temperature": "Temp",
    "Total flow": "Total_flow",
    "Argon flow": "Ar_flow",
    "CH4 flow": "CH4_flow",
    "O2 flow": "O2_flow",
    "CT": "CT"
}
reg_y_choices = {
    "CarbonMonoOxide_y": "COy",
    "CH4_conv": "CH4_conv",
    "CO2y": "CO2y",
    "C2y": "C2y"
}
reg_select_x = MultiSelect(title="X value",
                           options=sorted(reg_x_choices.keys()),
                           size=len(reg_x_choices),
                           value=["Argon flow"])
reg_select_y = Select(title="Y value",
                      options=sorted(reg_y_choices.keys()),
                      value="CarbonMonoOxide_y")

reg_controls = [reg_select_x, reg_select_y]
for control in reg_controls:
    control.on_change("value", lambda attr, old, new: update_regression())
reg_inputs = column(*reg_controls, width=200)

# Create column data for the plot
reg_training_source = ColumnDataSource(data=dict(y_actual=[], y_predict=[]))
reg_testing_source = ColumnDataSource(data=dict(y_actual=[], y_predict=[]))
# Table to display R^2 and RMSE
reg_RMSE_source = ColumnDataSource(data=dict(
    tabs=["R^2 for Training", "R^2 for Testing",
          "RMSE for Training", "RMSE for Testing"],
    data=[None, None, None, None]))
reg_RMSE_column = [
    TableColumn(field="tabs"),
    TableColumn(field="data")
]
reg_RMSE_data_table = DataTable(
    source=reg_RMSE_source, columns=reg_RMSE_column, header_row=False, index_position=None, width=200)
# Table to display coefficients
reg_coeff_source = ColumnDataSource(data=dict(Variables=[], Coefficients=[]))
reg_coeff_column = [
    TableColumn(field="Variables", title="Variables"),
    TableColumn(field="Coefficients", title="Coefficients")
]
reg_coeff_data_table = DataTable(
    source=reg_coeff_source, columns=reg_coeff_column, index_position=None, header_row=True, width=200)

# Create figure to display the scatter plot for training set
reg_training = figure(height=500, width=600,
                      toolbar_location="above", title="Actual vs. Predicted")
reg_training.scatter(x="y_actual", y="y_predict", source=reg_training_source)
reg_training.xaxis.axis_label = "Actual"
reg_training.yaxis.axis_label = "Predicted"
# TODO: add histogram for training set
reg_training_hist, reg_training_edges = np.histogram(
    reg_training_source.data["y_actual"], bins=20)
reg_training_hori_hist = figure(toolbar_location=None, width=reg_training.width,
                                height=100, x_range=reg_training.x_range, y_range=(0, max(reg_training_hist)*1.1),
                                min_border=10, min_border_left=50, y_axis_location="right")
reg_training_hori_hist_bar = reg_training_hori_hist.quad(
    bottom=0, left=reg_training_edges[:-1], right=reg_training_edges[1:], top=reg_training_hist)

reg_training_layout = column(reg_training, reg_training_hori_hist)

# Create figure to display the scatter plot for testing set
reg_testing = figure(height=500, width=600,
                     toolbar_location="above", title="Actual vs. Predicted")
reg_testing.scatter(x="y_actual", y="y_predict", source=reg_testing_source)
reg_testing.xaxis.axis_label = "Actual"
reg_testing.yaxis.axis_label = "Predicted"
# TODO: add histogram for testing set
reg_testing_hist, reg_testing_edges = np.histogram(
    reg_testing_source.data["y_actual"], bins=20)
reg_testing_hori_hist = figure(toolbar_location=None, width=reg_testing.width,
                               height=100, x_range=reg_testing.x_range, y_range=(0, max(reg_testing_hist)*1.1),
                               min_border=10, min_border_left=50, y_axis_location="right")
reg_testing_hori_hist_bar = reg_testing_hori_hist.quad(
    bottom=0, left=reg_testing_edges[:-1], right=reg_testing_edges[1:], top=reg_testing_hist)

reg_testing_layout = column(reg_testing, reg_testing_hori_hist)

# Adding tabs for regression plots
reg_tab1 = Panel(child=reg_training_layout, title="Training Dataset")
reg_tab2 = Panel(child=reg_testing_layout, title="Testing Dataset")
reg_tabs = Tabs(tabs=[reg_tab1, reg_tab2])

regression_layout = column(
    [row(column(reg_inputs, reg_RMSE_data_table), reg_tabs, reg_coeff_data_table)], sizing_mode="scale_both")


def update_regression():
    x_name = []
    for choice in reg_select_x.value:
        x_name.append(reg_x_choices[choice])
    y_name = reg_y_choices[reg_select_y.value]
    # print("x values: ", x_name)
    # print("y value: ", y_name)
    reg_x = df_catalysis_dataset[x_name].values
    reg_y = df_catalysis_dataset[y_name].values
    # Split into training and test
    reg_x_train, reg_x_test, reg_y_train, reg_y_test = train_test_split(
        reg_x, reg_y, test_size=0.2, random_state=0)
    # Training model
    reg_ml = LinearRegression()
    reg_ml.fit(reg_x_train, reg_y_train)
    # Predict y using x test
    reg_y_train_pred = reg_ml.predict(reg_x_train)
    reg_y_test_pred = reg_ml.predict(reg_x_test)
    reg_training_source.data = dict(
        y_actual=reg_y_train, y_predict=reg_y_train_pred)
    reg_testing_source.data = dict(
        y_actual=reg_y_test, y_predict=reg_y_test_pred)
    # Update data in the table
    reg_RMSE_source.data["data"] = np.around([
        r2_score(reg_y_train, reg_y_train_pred),
        r2_score(reg_y_test, reg_y_test_pred),
        np.sqrt(mean_squared_error(reg_y_train, reg_y_train_pred)),
        np.sqrt(mean_squared_error(reg_y_test, reg_y_test_pred))
    ], decimals=6)
    reg_coeff_source.data = dict(
        Variables=x_name, Coefficients=np.around(reg_ml.coef_, decimals=6))
    # print(reg_coeff_source.data)
    # update histogram
    # global reg_training_hist, reg_training_edges, reg_testing_hist, reg_testing_edges
    reg_training_hist, reg_training_edges = np.histogram(reg_y_train, bins=20)
    reg_training_hori_hist.y_range.end = max(reg_training_hist)*1.1
    reg_training_hori_hist_bar.data_source.data["top"] = reg_training_hist
    reg_training_hori_hist_bar.data_source.data["right"] = reg_training_edges[1:]
    reg_training_hori_hist_bar.data_source.data["left"] = reg_training_edges[:-1]
    reg_testing_hist, reg_testing_edges = np.histogram(reg_y_test, bins=20)
    reg_testing_hori_hist.y_range.end = max(reg_testing_hist)*1.1
    reg_testing_hori_hist_bar.data_source.data["top"] = reg_testing_hist
    reg_testing_hori_hist_bar.data_source.data["right"] = reg_testing_edges[1:]
    reg_testing_hori_hist_bar.data_source.data["left"] = reg_testing_edges[:-1]


# organizing panels of display
tab1 = Panel(child=visualization_layout, title="Data Exploration")
tab2 = Panel(child=column(select_color, c_corr), title="Correlation Matrix")
tab3 = Panel(child=regression_layout, title="Multivariable Regression")
tabs = Tabs(tabs=[tab1, tab2, tab3])

update()  # initial load of the data
update_regression()
curdoc().add_root(tabs)
curdoc().title = "Catalysis Data"
r.data_source.selected.on_change('indices', update_histogram)
