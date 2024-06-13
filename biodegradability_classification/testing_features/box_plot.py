import pandas as pd

from bokeh.models import ColumnDataSource, Whisker, Button, Div
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.layouts import column

global current_list
current_list = [2, 3, 6, 8, 10]

global saved_list 
saved_list = [None, None, None, None, None]

combined_list = current_list + saved_list

d = {'kind': ['current', 'current', 'current', 'current', 'current',
              'saved', 'saved', 'saved', 'saved', 'saved'],
     'accuracy': combined_list
     }
df = pd.DataFrame(data=d)

kinds = df.kind.unique()
print(kinds)

# compute quantiles
qs = df.groupby("kind").accuracy.quantile([0.25, 0.5, 0.75])
qs = qs.unstack().reset_index()
qs.columns = ["kind", "q1", "q2", "q3"]
df = pd.merge(df, qs, on="kind", how="left")

# compute IQR outlier bounds
iqr = df.q3 - df.q1
df["upper"] = df.q3 + 1.5*iqr
df["lower"] = df.q1 - 1.5*iqr

source = ColumnDataSource(df)

p = figure(x_range=kinds, tools="", toolbar_location=None,
           title="Validation Accuracy saved vs. current",
           background_fill_color="#eaefef", y_axis_label="accuracy")

# outlier range
whisker = Whisker(base="kind", upper="upper", lower="lower", source=source)
whisker.upper_head.size = whisker.lower_head.size = 20
p.add_layout(whisker)

# quantile boxes
cmap = factor_cmap("kind", "Paired3", kinds)
p.vbar("kind", 0.7, "q2", "q3", source=source, color=cmap, line_color="black")
p.vbar("kind", 0.7, "q1", "q2", source=source, color=cmap, line_color="black")

# outliers
outliers = df[~df.accuracy.between(df.lower, df.upper)]
p.scatter("kind", "accuracy", source=outliers, size=6, color="black", alpha=0.3)

p.xgrid.grid_line_color = None
p.axis.major_label_text_font_size="14px"
p.axis.axis_label_text_font_size="12px"

# Create status message Div
status_message = Div(text='Run not saved', styles={'color': 'red', 'font-size': '16px'})

def save_accuracy():
    saved_list = current_list
    status_message.text = 'Run saved'
    status_message.styles = {'color': 'green', 'font-size': '16px'}

# Save button
save_button = Button(label="Save this run", button_type="success")

def save_accuracy():
    saved_list = current_list
    status_message.text = 'Run saved'
    status_message.styles = {'color': 'green', 'font-size': '16px'}


layout = column(p, save_button, status_message)

curdoc().add_root(layout)
