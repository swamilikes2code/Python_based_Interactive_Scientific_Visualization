from bokeh.plotting import curdoc
from bokeh.layouts import row, column
from bokeh.models import Div, Spacer, Button

# Widgets
top_page_spacer = Spacer(height=20)
alg_select = Div(text="<h3>Algorithm Selection</h3>", css_classes=['grey-box'])
train_button = Button(label="Run ML algorithm", button_type="success", width=200)



# Create a column layout with grey box background
alg_select_layout = column(
    top_page_spacer,
    alg_select,
    row(train_button, Spacer(width=20)),
    css_classes=['grey-box']
)

# Combine layouts
tab2_layout = row(
    column(
        alg_select_layout,
        Spacer(height=20),  # Example spacer
        Div(text="Additional Content"),  # Example additional content
        Spacer(height=20)  # Example spacer
    )
)

# Add the layout to a tab or root layout if using tabs
tabs = []
tabs.append(tab2_layout)  # Add other tabs as needed

# Add the root layout (tabs) to the current document
curdoc().add_root(row(tabs))


