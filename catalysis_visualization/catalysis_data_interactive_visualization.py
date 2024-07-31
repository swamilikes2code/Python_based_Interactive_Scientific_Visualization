from bokeh.models.widgets.markups import Div
import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, Select, Slider, BoxSelectTool, LassoSelectTool, Tabs, TabPanel, LinearColorMapper, ColorBar, BasicTicker, PrintfTickFormatter, MultiSelect, DataTable, TableColumn, Spacer
from bokeh.plotting import figure, curdoc
from bokeh.palettes import viridis, gray, cividis, Category20
from bokeh.transform import factor_cmap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.decomposition import PCA

######################### prepare dataset #########################
master=True
# master=False

if master:
    # Import dataset
    df_catalysis_dataset = pd.read_csv("catalysis_visualization/data/OCM-data.csv",
                                    index_col=0, header=0)
else:
    df_catalysis_dataset = pd.read_csv("./data/OCM-data.csv",
                                    index_col=0, header=0)


# Removing the Blank names from the data
df_catalysis_dataset.set_index(df_catalysis_dataset.index)
df_catalysis_dataset.drop("Blank", axis=0)

# Calculating error percentage

# Sum of columns to compare with CH4_conv
df_catalysis_dataset["Sum_y"] = df_catalysis_dataset.loc[:,
                                                         "C2H6y":"CO2y"].sum(axis=1)
df_catalysis_dataset["error_ch4_conv"] = abs((df_catalysis_dataset["Sum_y"]-df_catalysis_dataset["CH4_conv"]) /
                                             df_catalysis_dataset["CH4_conv"])*100

###########################################################################


######################### data visualization model #########################
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
# TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset,box_zoom,undo,redo"

TOOLS = "wheel_zoom,box_select,lasso_select,reset,box_zoom,undo,redo"

# Create Column Data Source that will be used by the plot
source = ColumnDataSource(data=dict(x=[], y=[], M1=[], M2=[], M3=[], Name=[]))

p = figure(height=300, width=280, title="Data Exploration", tools=TOOLS,
           toolbar_location="left", tooltips=TOOLTIPS)
p.select(BoxSelectTool).continuous = False
p.select(LassoSelectTool).continuous = False
r = p.scatter(x="x", y="y", source=source, size=7,
             color='mediumblue', line_color=None, fill_alpha=0.6)


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

ph = figure(toolbar_location=None, width=(p.width + 19), height=100, x_range=p.x_range,
            y_range=(0, (max(hhist)*1.1)), min_border_left=73, y_axis_location="right")
ph.xgrid.grid_line_color = None
ph.yaxis.major_label_orientation = np.pi/4
# ph.background_fill_color = "#fafafa"

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
            x_range=(0, (max(vhist)*1.1)), y_range=p.y_range, min_border=10, y_axis_location="left")
pv.ygrid.grid_line_color = None
pv.xaxis.major_label_orientation = np.pi/4
# pv.background_fill_color = "#fafafa"

# histogram to reflect the data points
vv = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:],
             right=vhist, color="white", line_color="#3A5785")
# histograms highlight on top of the original histogram
vh1 = pv.quad(
    left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, alpha=0.5, **LINE_ARGS)
vh2 = pv.quad(
    left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, alpha=0.1, **LINE_ARGS)

# layout = gridplot([[p, pv], [ph, None]])
layout = column(row(p, pv), ph)


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

###########################################################################


######################### correlation matrix #########################
# Copy x-axis values into new df
df_corr_0 = df_catalysis_dataset[
    ["CT", "Ar_flow", "CH4_flow", "O2_flow", "Total_flow", "Support_ID", "Temp",
     "M2_mol", "M3_mol", "M1_atom_number", "M2_atom_number", "M3_atom_number",
     "M1_mol_percentage", "M2_mol_percentage", "M3_mol_percentage"]
]

df_corr = df_corr_0.rename(columns={"M1_mol_percentage": "M1 mol %",
                                    "M2_mol_percentage": "M2 mol %",
                                    "M3_mol_percentage": "M3 mol %",
                                    "M1_atom_number": "M1 atom #",
                                    "M2_atom_number": "M2 atom #",
                                    "M3_atom_number": "M3 atom #",
                                    "M3_mol": "M3 mol",
                                    "M2_mol": "M2 mol",
                                    "Support_ID": "Support ID",
                                    "Total_flow": "Total flow",
                                    "O2_flow": "O2 flow",
                                    "CH4_flow": "CH4 flow",
                                    "Ar_flow": "Ar flow"})

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
                width=360, height=350, tools='hover', tooltips=[('Parameters', '@level_0 - @parameters'), ('Correlation', '@correlation')])


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
###########################################################################


######################### regression model #########################
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
reg_model_choices = {
    "Linear": 1,
    "Quadratic": 2,
    "Cubic": 3
}
reg_select_x = MultiSelect(title="X value",
                           options=sorted(reg_x_choices.keys()),
                           size=len(reg_x_choices),
                           value=["Argon flow"])
reg_select_y = Select(title="Y value",
                      options=sorted(reg_y_choices.keys()),
                      value="CarbonMonoOxide_y")
reg_select_model = Select(title="Models",
                          options=list(reg_model_choices.keys()),
                          value="Linear")

reg_controls = [reg_select_x, reg_select_y, reg_select_model]
for control in reg_controls:
    control.on_change("value", lambda attr, old, new: update_regression())
reg_inputs = column(*reg_controls, width=200)

# Table to display R^2 and RMSE
reg_RMSE_source = ColumnDataSource(data=dict(
    tabs=["R^2 for Training", "R^2 for Testing",
          "RMSE for Training", "RMSE for Testing"],
    data=[None, None, None, None]))
reg_RMSE_column = [
    TableColumn(field="tabs"),
    TableColumn(field="data")
]
reg_RMSE_data_table = DataTable(source=reg_RMSE_source, columns=reg_RMSE_column,
                                header_row=False, index_position=None, width=250)

# Table to display coefficients
reg_coeff_source = ColumnDataSource(
    data=dict(Variables=[], Coefficients=[]))
reg_coeff_column = [
    TableColumn(field="Variables", title="Variables"),
    TableColumn(field="Coefficients", title="Coefficients")
]
reg_coeff_data_table = DataTable(source=reg_coeff_source, columns=reg_coeff_column,
                                 index_position=None, header_row=True, width=250)

# Create figure to display the scatter plot for training set
reg_training_source = ColumnDataSource(data=dict(y_actual=[], y_predict=[]))
reg_training = figure(height=400, width=400, toolbar_location="above", tools="box_zoom,reset,undo,redo,save",
                      title="Actual vs. Predicted")
reg_training.scatter(x="y_actual", y="y_predict", source=reg_training_source)
reg_training.xaxis.axis_label = "Actual"
reg_training.yaxis.axis_label = "Predicted"

# Histogram for training set
reg_training_hist = figure(toolbar_location=None, tools='', width=reg_training.width, title="Error Histogram",
                           height=200, min_border=10, y_axis_location="right")
reg_training_hist.y_range.start = 0
reg_training_hist.xgrid.grid_line_color = None
reg_training_hist.yaxis.major_label_orientation = "horizontal"
reg_training_hist_source = ColumnDataSource(
    data=dict(top=[], left=[], right=[]))
reg_training_hist.quad(bottom=0, left="left", right="right",
                       top="top", source=reg_training_hist_source)

# training layout
reg_training_layout = column(reg_training, reg_training_hist)

# Create figure to display the scatter plot for testing set
reg_testing_source = ColumnDataSource(data=dict(y_actual=[], y_predict=[]))
reg_testing = figure(height=400, width=400, toolbar_location="above", tools="box_zoom,reset,undo,redo,save",
                     title="Actual vs. Predicted")
reg_testing.scatter(x="y_actual", y="y_predict", source=reg_testing_source)
reg_testing.xaxis.axis_label = "Actual"
reg_testing.yaxis.axis_label = "Predicted"

# Histogram for testing set
reg_testing_hist = figure(toolbar_location=None, width=reg_testing.width, title="Error Histogram",
                          height=200, min_border=10, y_axis_location="right")
reg_testing_hist.y_range.start = 0
reg_testing_hist.xgrid.grid_line_color = None
reg_testing_hist.yaxis.major_label_orientation = "horizontal"
reg_testing_hist_source = ColumnDataSource(
    data=dict(top=[], left=[], right=[]))
reg_testing_hist.quad(bottom=0, left="left", right="right",
                      top="top", source=reg_testing_hist_source)

# testing layout
reg_testing_layout = column(reg_testing, reg_testing_hist)

# Support Lines
# trend line
reg_training_trend_source = ColumnDataSource(data=dict(x=[], y=[]))
reg_training.line(x="x", y="y", source=reg_training_trend_source,
                  color="black", line_width=2, legend_label="y = x")

reg_testing_trend_source = ColumnDataSource(data=dict(x=[], y=[]))
reg_testing.line(x="x", y="y", source=reg_testing_trend_source,
                 color="black", line_width=2, legend_label="y = x")

# line of best fit
reg_training_line_source = ColumnDataSource(data=dict(x=[], y=[]))
reg_training.line(x="x", y="y", source=reg_training_line_source,
                  color="red", line_width=1.5, legend_label="Line of Best Fit")

reg_testing_line_source = ColumnDataSource(data=dict(x=[], y=[]))
reg_testing.line(x="x", y="y", source=reg_testing_line_source,
                 color="red", line_width=1.5, legend_label="Line of Best Fit")

reg_training.legend.click_policy = "hide"
reg_training.legend.location = "top_left"
reg_training.legend.background_fill_alpha = 0.5
reg_testing.legend.click_policy = "hide"
reg_testing.legend.location = "top_left"
reg_testing.legend.background_fill_alpha = 0.5

# Adding tabs for regression plots
reg_tab1 = TabPanel(child=reg_training_layout, title="Training Dataset")
reg_tab2 = TabPanel(child=reg_testing_layout, title="Testing Dataset")
reg_tabs = Tabs(tabs=[reg_tab1, reg_tab2])


def update_regression():
    # get selected values from selectors
    x_name = []  # list of attributes
    for choice in reg_select_x.value:
        x_name.append(reg_x_choices[choice])
    y_name = reg_y_choices[reg_select_y.value]
    reg_x = df_catalysis_dataset[x_name].values
    reg_y = df_catalysis_dataset[y_name].values
    # normalize data
    standardized_reg_x = StandardScaler().fit_transform(reg_x)
    # Split into training and test
    reg_x_train, reg_x_test, reg_y_train, reg_y_test = train_test_split(
        standardized_reg_x, reg_y, test_size=0.2, random_state=0)
    # Transform data into polynomial features
    reg_pre_process = PolynomialFeatures(
        degree=reg_model_choices[reg_select_model.value])
    reg_x_train = reg_pre_process.fit_transform(reg_x_train)
    reg_x_test = reg_pre_process.fit_transform(reg_x_test)
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

    # Trend lines
    # Trend line for training
    reg_training_trend_interval = np.linspace(
        start=max(min(reg_y_train), min(reg_y_train_pred)),
        stop=min(max(reg_y_train), max(reg_y_train_pred))
    )
    reg_training_trend_source.data = dict(
        x=reg_training_trend_interval,
        y=reg_training_trend_interval
    )

    # Trend line for testing
    reg_testing_trend_interval = np.linspace(
        start=max(min(reg_y_test), min(reg_y_test_pred)),
        stop=min(max(reg_y_test), max(reg_y_test_pred))
    )
    reg_testing_trend_source.data = dict(
        x=reg_testing_trend_interval,
        y=reg_testing_trend_interval
    )

    # Line of best fit
    # Regrssion line of best fit using numpy(Training dataset)
    par_training = np.polyfit(reg_y_train, reg_y_train_pred, deg=1, full=True)
    slope_training = par_training[0][0]
    intercept_training = par_training[0][1]
    y_predicted_training = [slope_training*i +
                            intercept_training for i in reg_y_train]
    reg_training_line_source.data = dict(x=reg_y_train, y=y_predicted_training)

    # Regression Line of Best Fit (Testing dataset)
    par_testing = np.polyfit(reg_y_test, reg_y_test_pred, deg=1, full=True)
    slope_testing = par_testing[0][0]
    intercept_testing = par_testing[0][1]
    y_predicted_testing = [slope_testing*i +
                           intercept_testing for i in reg_y_test]
    reg_testing_line_source.data = dict(x=reg_y_test, y=y_predicted_testing)

    # Update data in the table
    reg_RMSE_source.data["data"] = np.around([
        r2_score(reg_y_train, reg_y_train_pred),
        r2_score(reg_y_test, reg_y_test_pred),
        np.sqrt(mean_squared_error(reg_y_train, reg_y_train_pred)),
        np.sqrt(mean_squared_error(reg_y_test, reg_y_test_pred)),
    ], decimals=4)

    # Update coefficients
    # array of variable names
    x_name_coef_key = reg_pre_process.get_feature_names_out(x_name)
    x_name_coef_key = list(x_name_coef_key)
    x_name_coef_key.append("Intercept")
    reg_coeff = list(reg_ml.coef_)
    reg_coeff.append(reg_ml.intercept_)
    reg_coeff_source.data = dict(Variables=x_name_coef_key,
                                 Coefficients=np.around(reg_coeff, decimals=4))

    # Update histograms
    # training set
    reg_training_diff = []
    reg_training_zip = zip(reg_y_train_pred, reg_y_train)
    for pred, train in reg_training_zip:
        reg_training_diff.append(pred - train)

    reg_training_hhist, reg_training_hedges = np.histogram(reg_training_diff,
                                                           bins=20)
    reg_training_hist_source.data = dict(top=reg_training_hhist,
                                         right=reg_training_hedges[1:],
                                         left=reg_training_hedges[:-1])

    # testing set
    reg_testing_diff = []
    reg_testing_zip = zip(reg_y_test_pred, reg_y_test)
    for pred, train in reg_testing_zip:
        reg_testing_diff.append(pred - train)

    reg_testing_hhist, reg_testing_hedges = np.histogram(reg_testing_diff,
                                                         bins=20)
    reg_testing_hist_source.data = dict(top=reg_testing_hhist,
                                        right=reg_testing_hedges[1:],
                                        left=reg_testing_hedges[:-1])

###########################################################################


######################### unsupervised learning model #########################
# selection tools
unsuper_learn_x_choices = {
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

# standard dataset
# use to find the index
unsuper_learn_attributes = list(unsuper_learn_x_choices.values())
# the dataset without names
unsuper_learn_std_df = StandardScaler().fit_transform(
    df_catalysis_dataset[unsuper_learn_attributes].values)

# selectors
unsuper_learn_select_x = Select(title="X axis", value="Argon flow",
                                options=sorted(unsuper_learn_x_choices.keys()))
unsuper_learn_select_y = Select(title="Y axis", value="CT",
                                options=sorted(unsuper_learn_x_choices.keys()))
unsuper_learn_k_cluster_select = Slider(title="K", start=1, end=11,
                                        value=4, step=1)
unsuper_learn_PCA_select = Slider(title="# of PCA", start=2, end=15,
                                        value=4, step=1)

unsuper_learn_controls = [unsuper_learn_select_x,
                          unsuper_learn_select_y,
                          unsuper_learn_k_cluster_select,
                          unsuper_learn_PCA_select]
for control in unsuper_learn_controls:
    control.on_change("value", lambda attr, old,
                      new: update_unsuper_learning())
unsuper_learn_inputs = column(*unsuper_learn_controls, width=200)

# k clustering plot
COLORS = Category20[11]
unsuper_learn_k_cluster_source = ColumnDataSource(data=dict(x=[], y=[], c=[]))
unsuper_learn_k_cluster_model = figure(height=350, width=400, toolbar_location="right", tools="box_zoom,reset,undo,redo,save",
                                       title="Visualizing Clustering")
unsuper_learn_k_cluster_model.scatter(x="x", y="y", source=unsuper_learn_k_cluster_source,
                                     fill_alpha=0.5, line_color=None, size=8, color="c")

# elbow method plot
unsuper_learn_elbow_source = ColumnDataSource(data=dict(x=[], y=[]))
unsuper_learn_elbow_model = figure(height=350, width=400, toolbar_location="right", tools="box_zoom,reset,undo,redo,save",
                                   title="Elbow Method")
unsuper_learn_elbow_model.line(x="x", y="y", source=unsuper_learn_elbow_source)
unsuper_learn_elbow_model.xaxis.axis_label = "Number of Clusters, k"
unsuper_learn_elbow_model.yaxis.axis_label = "Error"

# PCA plot
unsuper_learn_PCA_source = ColumnDataSource(data=dict(x=[], y=[], c=[]))
unsuper_learn_PCA_model = figure(height=350, width=400, toolbar_location="right", tools="box_zoom,reset,undo,redo,save",
                                 title="Principal Component Analysis")
unsuper_learn_PCA_model.scatter(x="x", y="y", fill_alpha=0.5, line_color=None,
                               size=5, source=unsuper_learn_PCA_source, color="c")
unsuper_learn_PCA_model.xaxis.axis_label = "Principal Component 1"
unsuper_learn_PCA_model.yaxis.axis_label = "Principal Component 2"

# histogram
unsuper_learn_PCA_hist_source = ColumnDataSource(
    data=dict(top=[], left=[], right=[]))
unsuper_learn_PCA_hist_model = figure(height=350, width=400, toolbar_location="right", tools="box_zoom,reset,undo,redo,save",
                                      title="PCA Histogram")
unsuper_learn_PCA_hist_model.y_range.start = 0
unsuper_learn_PCA_hist_model.quad(top="top", left="left", right="right",
                                  bottom=0, source=unsuper_learn_PCA_hist_source)
unsuper_learn_PCA_hist_model.xaxis.axis_label = "Principal Components"
unsuper_learn_PCA_hist_model.yaxis.axis_label = "Variance %"

# loading table
unsuper_loading_source = ColumnDataSource(data=dict())
unsuper_loading_table = DataTable(source=unsuper_loading_source,
                                  header_row=True, index_position=None, width = 360, autosize_mode = "none")

# layout



def kmean_preset():
    # elbow
    Error = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(unsuper_learn_std_df)
        Error.append(kmeans.inertia_)
    temp = [j for j in range(1, 11)]
    unsuper_learn_elbow_source.data = dict(x=temp, y=Error)


def update_unsuper_learning():
    # k means
    unsuper_learn_kmeans = KMeans(n_clusters=unsuper_learn_k_cluster_select.value,
                                  random_state=0).fit_predict(unsuper_learn_std_df)
    xax = unsuper_learn_attributes.index(
        unsuper_learn_x_choices[unsuper_learn_select_x.value])
    yax = unsuper_learn_attributes.index(
        unsuper_learn_x_choices[unsuper_learn_select_y.value])

    # Coloring clusters
    groups = pd.Categorical(unsuper_learn_kmeans)
    colors_df = [COLORS[xx] for xx in groups.codes]
    unsuper_learn_k_cluster_source.data = dict(x=unsuper_learn_std_df[:, xax],
                                               y=unsuper_learn_std_df[:, yax],
                                               c=colors_df)
    unsuper_learn_k_cluster_model.xaxis.axis_label = unsuper_learn_select_x.value
    unsuper_learn_k_cluster_model.yaxis.axis_label = unsuper_learn_select_y.value

    # PCA
    pca = PCA(n_components=unsuper_learn_PCA_select.value).fit(
        unsuper_learn_std_df)
    principalComponents = pca.transform(unsuper_learn_std_df)
    unsuper_learn_PCA_source.data = dict(x=principalComponents[:, 0],
                                         y=principalComponents[:, 1],
                                         c=colors_df)
    left = []
    right = []
    for i in range(1, pca.n_components_+1):
        left.append(i-0.25)
        right.append(i+0.25)
    unsuper_learn_PCA_hist_source.data = dict(top=pca.explained_variance_ratio_,
                                              left=left,
                                              right=right)

    # loadings
    # array = [ x if x > threshold else 0.0 for x in array ]
    loadings = np.around(pca.components_.T, decimals=5)
    num_pc = pca.n_components_
    pc_list = ["PC"+str(i) for i in range(1, num_pc+1)]
    loadings_df = pd.DataFrame(loadings, columns=pc_list,
                               index=list(unsuper_learn_x_choices.keys()))
    Columns = [TableColumn(field=Ci, title=Ci) for Ci in loadings_df.columns]
    loadings_df = loadings_df.reset_index().rename(
        columns={"index": "Variables"})
    Columns.insert(0, TableColumn(field="Variables", title="Variables"))
    unsuper_loading_source.data = loadings_df
    unsuper_loading_table.columns = Columns

###########################################################################


######################### classification model #########################
svm_x_choices = {
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

df_catalysis_dataset['classifier'] = np.where(df_catalysis_dataset['C2s'] >= 40.0,
                                              True, False)
class_sample_data = df_catalysis_dataset.sample(frac=0.40, random_state=79)
svm_target = class_sample_data["classifier"].values

svm_choices = {
    "Linear": "linear",
    "Polynomial": "poly",
    "RBF": "rbf",
    "Sigmoid": "sigmoid"
}

svm_select_x = MultiSelect(title="Choose Training Features",
                           options=sorted(svm_x_choices.keys()),
                           size=len(svm_x_choices),
                           value=["Argon flow", "Temperature", "O2 flow"])

svm_select_model = Select(title="Models",
                          options=list(svm_choices.keys()),
                          value="Linear")

select_class_x_axis = Select(title="X-axis",
                             options=sorted(svm_x_choices.keys()),
                             value="O2 flow")

select_class_y_axis = Select(title="Y-axis",
                             options=sorted(svm_x_choices.keys()),
                             value="CH4 flow")

svm_controls = [svm_select_x, svm_select_model,
                select_class_x_axis, select_class_y_axis]
for control in svm_controls:
    control.on_change("value", lambda attr, old, new: update_classification())
svm_inputs = column(*svm_controls, width=200)

# Table to display coefficients
class_cm_source = ColumnDataSource(
    data=dict(x=[], y=[], z=[]))
class_cm_column = [
    TableColumn(field="z", title=""),
    TableColumn(field="x", title="Actual C2s over 40"),
    TableColumn(field="y", title="Actual C2s under 40")
]
class_cm_data_table = DataTable(source=class_cm_source, columns=class_cm_column, height=100,
                                header_row=True, width=380, index_position=None)

classification_svm_source = ColumnDataSource(data=dict(x=[], y=[], color=[]))
classification_svm_model = figure(height=350, width=400, toolbar_location="right", tools="box_zoom,reset,undo,redo,save",
                                  title="SVM")
classification_svm_model.scatter(x="x", y="y", color="color",
                                 source=classification_svm_source)

# Table to display Recall, Accuracy and others
class_scores_source = ColumnDataSource(data=dict(
    tabs=["Accuracy", "Recall", "F-Measure", "Sensitivity", "Specificity"],
    data=[None, None, None, None, None]))
class_scores_column = [
    TableColumn(field="tabs"),
    TableColumn(field="data")
]
class_scores_table = DataTable(source=class_scores_source, columns=class_scores_column,
                               header_row=False, index_position=None, width=380)


def update_classification():
    x_name = []  # list of attributes
    for choice in svm_select_x.value:
        x_name.append(svm_x_choices[choice])
    svm_x = class_sample_data[x_name]
    X_train, X_test, y_train, y_test = train_test_split(svm_x, svm_target,
                                                        train_size=0.8, random_state=0)
    svclassifier = SVC(kernel=svm_choices[svm_select_model.value], C=1,
                       decision_function_shape='ovr',
                       gamma=1, degree=2).fit(X_train, y_train)
    y_test_pred = svclassifier.predict(X_test)
    y_train_pred = svclassifier.predict(X_train)
    classification_svm_model.xaxis.axis_label = select_class_x_axis.value
    classification_svm_model.yaxis.axis_label = select_class_y_axis.value
    classification_svm_source.data = dict(x=class_sample_data[svm_x_choices[select_class_x_axis.value]],
                                          y=class_sample_data[svm_x_choices[select_class_y_axis.value]],
                                          color=np.where(class_sample_data['classifier'] == 1, "red", "blue"))
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    confusion = pd.DataFrame(data=cm,
                             columns=["Actual C2s over 40",
                                      "Actual C2s below 40"],
                             index=["Predicted C2s over 40", "Predicted C2s below 40"])
    confusion.insert(loc=0, column=" ",
                     value=["Predicted C2s over 40", "Predicted C2s below 40"])
    confusion.iloc[0, 1] = tp
    confusion.iloc[1, 1] = fn
    confusion.iloc[0, 2] = fp
    confusion.iloc[1, 2] = tn
    class_cm_source.data = dict(
        x=confusion["Actual C2s over 40"],
        y=confusion["Actual C2s below 40"],
        z=confusion.iloc[:, 0]
    )

    class_scores_source.data["data"] = np.around([
        svclassifier.score(X_test, y_test),
        recall_score(y_test, y_test_pred),
        f1_score(y_test, y_test_pred),
        (tp/(tp+fn)),
        (tn/(tn+fp))
    ], decimals=4)
    # Get support vector indices
    support_vector_indices = svclassifier.support_
    # Get number of support vectors per class
    support_vectors_per_class = svclassifier.n_support_
    support_vectors = svclassifier.support_vectors_

###########################################################################


######################### text description #########################
# div1 = Div(text="The Data Exploration section allows for one to understand the distribution of the data set being used. One will be able to play with different things such as minimum temperature, minimum methane conversion, and minimum error to see how the data changes. ", margin=(20, 20, 10, 20), width=750)
# div2 = Div(text="The Correlation Matrix shows how strong the correlation is between all the different features in the dataset. This is important because depending on the strength of correlation, one can make useful predictions about a potential regression. ", margin=(10, 20, 10, 20), width=750)
# div3 = Div(text="The Multivariable Regression section allows for one to build their own regression model. The objective of the model is to show how good certain features are in predicting an output. This is achieved by a parity plot which shows the <strong>actual</strong> on the <strong>x axis</strong> and <strong>predicted</strong> on the <strong>y axis</strong>. Furthermore, while choosing the different features to go into the model, the user will be able to see many evaluation metrics such as R^2, regression coefficients, and an error histogram. ", margin=(10, 20, 10, 20), width=750)
# div4 = Div(text="The Unsupervised Learning section will introduce two techniques. These are clustering analysis and principal component analysis. The objective of the clustering plot is to try to group similar data points within the data set. To help with the clustering plot, an elbow plot is also included to help indicate the ideal number of clusters in the plot. The objective of principal component analysis is to reduce the dimensionality of a large dataset into a few key components which still explain most of the information in the dataset. In this section, we show this through the PCA plot which plots the first two principal components, and through a histogram which explains how much information each principal component accounts for. ", margin=(10, 20, 10, 20), width=750)
# div5 = Div(text="The Classification section will show ways in which the data is partitioned into different “classes”. With the dataset being used, the classes are a good catalyst and a bad catalyst. This is achieved through a support vector machine model. Within the model, one can choose between 4 kernels and see how the data changes. Furthermore, there are evaluation metrics included in the form of a classification report and confusion matrix. ", margin=(10, 20, 10, 20), width=750)
# text_descriptions = column(div1, div2, div3, div4, div5)

###########################################################################
# Spacers and Layouts
top_page_spacer = Spacer(height = 20)
left_page_spacer = Spacer(width = 20)
large_left_page_spacer = Spacer(width = 50)

visualization_layout = row(inputs, layout)

regression_layout = column(
    [row(column(reg_inputs, reg_RMSE_data_table), large_left_page_spacer, reg_tabs, left_page_spacer, reg_coeff_data_table)])

unsuper_learn_layout = row(column(unsuper_learn_inputs, unsuper_loading_table), left_page_spacer,
                                  column(unsuper_learn_k_cluster_model,
                                         unsuper_learn_elbow_model),
                                  column(unsuper_learn_PCA_model, unsuper_learn_PCA_hist_model))


svm_layout = row(svm_inputs, classification_svm_model, left_page_spacer,
                    column(
                        Div(text="<b>Confusion Matrix</b>"),
                        class_cm_data_table,
                        Div(text="<b>Evaluation Metrics</b>"),
                        class_scores_table))

# organizing TabPanels of display
tab1 = TabPanel(child=row(left_page_spacer, column(top_page_spacer, visualization_layout)), title="Data Exploration")
tab2 = TabPanel(child=row(left_page_spacer, column(top_page_spacer, select_color, c_corr)), title="Correlation Matrix")
tab3 = TabPanel(child=row(left_page_spacer, column(top_page_spacer, regression_layout)), title="Multivariable Regression")
tab4 = TabPanel(child=row(left_page_spacer, column(top_page_spacer, unsuper_learn_layout)), title="Unsupervised Learning")
tab5 = TabPanel(child=row(left_page_spacer, column(top_page_spacer, svm_layout)), title="Classification Methods")
# tab6 = TabPanel(child=text_descriptions, title="Model Description")
# tabs = Tabs(tabs=[tab6, tab1, tab2, tab3, tab4, tab5])
tabs = Tabs(tabs=[tab1, tab2, tab3, tab4, tab5])

update()  # initial load of the data
update_regression()
update_unsuper_learning()
kmean_preset()
update_classification()
curdoc().add_root(tabs)
curdoc().title = "Catalysis Data"
r.data_source.selected.on_change('indices', update_histogram)
