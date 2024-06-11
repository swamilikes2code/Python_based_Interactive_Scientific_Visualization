import pandas as pd
import numpy as np
from bokeh.models import ColumnDataSource, DataTable, TableColumn, CheckboxButtonGroup, Button, Div
from bokeh.io import curdoc, output_notebook
from bokeh.layouts import column

output_notebook()

# Load data from the CSV file
file_path = r'biodegrad.csv'
df = pd.read_csv(file_path)

# Remove unnecessary data
columns_to_remove = ['Class']  # columns not needed
df = df.drop(columns=columns_to_remove)  # remove those columns

# Columns that should always be shown
mandatory_columns = ['Substance Name', 'Smiles']

# Ensure mandatory columns exist in the dataframe (if not, create dummy columns)
for col in mandatory_columns:
    if col not in df.columns:
        df[col] = "N/A"

# Global saved list
global saved_list
saved_list = []

def create_table():
    # Limit the dataframe to the first 10 rows
    df_subset = df.head(10)

    df_dict = df_subset.to_dict("list")
    cols = list(df_dict.keys())

    # Separate mandatory and optional columns
    optional_columns = [col for col in cols if col not in mandatory_columns]

    # Create column datasource
    source = ColumnDataSource(data=df_subset)

    # Create figure
    columns = [TableColumn(field=col, title=col) for col in cols]
    figure = DataTable(source=source, columns=columns, width=1800)

    # Create widget excluding mandatory columns
    checkbox_button_group = CheckboxButtonGroup(labels=optional_columns, active=list(range(len(optional_columns))))

    # Create status message Div
    status_message = Div(text='Columns saved', styles={'color': 'green', 'font-size': '16px'})

    # Update columns to display
    def update_cols(display_columns):
        # Always include mandatory columns
        all_columns = mandatory_columns + display_columns
        figure.columns = [col for col in columns if col.title in all_columns]
        figure.width = np.size(all_columns) * 90

    def update(attr, old, new):
        cols_to_display = [checkbox_button_group.labels[i] for i in checkbox_button_group.active]
        update_cols(display_columns=cols_to_display)
        status_message.text = 'Columns not saved'
        status_message.styles = {'color': 'red', 'font-size': '16px'}

    checkbox_button_group.on_change('active', update)

    # Save columns to global saved list
    def save_columns():
        saved_columns = mandatory_columns + [checkbox_button_group.labels[i] for i in checkbox_button_group.active]
        saved_columns = sorted(saved_columns)
        saved_list.append(saved_columns)
        status_message.text = 'Columns saved'
        status_message.styles = {'color': 'green', 'font-size': '16px'}
        print('Columns saved')

    # Save button
    save_button = Button(label="Save Selected Columns", button_type="success")

    # Attach callback to the save button
    save_button.on_click(save_columns)

    curdoc().add_root(column(checkbox_button_group, figure, save_button, status_message))

create_table()

print(saved_list)
