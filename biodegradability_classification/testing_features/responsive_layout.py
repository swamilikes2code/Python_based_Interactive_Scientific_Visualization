from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import Div, Tabs, TabPanel
from bokeh.io import curdoc

# Create figures
figures_tab1 = [figure(title=f"Tab 1 - Plot {i+1}") for i in range(4)]
figures_tab2 = [figure(title=f"Tab 2 - Plot {i+1}") for i in range(4)]
figures_tab3 = [figure(title=f"Tab 3 - Plot {i+1}") for i in range(4)]

# Example scatter plots for demonstration
for fig in figures_tab1 + figures_tab2 + figures_tab3:
    fig.scatter([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=20, color="navy", alpha=0.5)

# HTML structure with CSS Grid
html_template = """
<style>
    .grid-container {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
    }
    @media (max-width: 800px) {
        .grid-container {
            grid-template-columns: 1fr;
        }
    }
</style>
<div class="grid-container">
    <div id="plot1"></div>
    <div id="plot2"></div>
    <div id="plot3"></div>
    <div id="plot4"></div>
</div>
"""

# Create Divs containing the HTML template
div_tab1 = Div(text=html_template, width=800)
div_tab2 = Div(text=html_template, width=800)
div_tab3 = Div(text=html_template, width=800)

# Create TabPanels for each tab
tab1 = TabPanel(child=div_tab1, title="Tab 1")
tab2 = TabPanel(child=div_tab2, title="Tab 2")
tab3 = TabPanel(child=div_tab3, title="Tab 3")

# Combine TabPanels into Tabs
tabs = Tabs(tabs=[tab1, tab2, tab3])

# Add tabs to document
curdoc().add_root(tabs)

# Embed Bokeh plots into the HTML divs using JavaScript
js_code = """
var renderers = Bokeh.documents[0].renderers;
var plot1 = renderers[0];
var plot2 = renderers[1];
var plot3 = renderers[2];
var plot4 = renderers[3];
document.getElementById('plot1').appendChild(plot1.canvas_view.el);
document.getElementById('plot2').appendChild(plot2.canvas_view.el);
document.getElementById('plot3').appendChild(plot3.canvas_view.el);
document.getElementById('plot4').appendChild(plot4.canvas_view.el);
"""

# Execute JavaScript code to embed plots
curdoc().add_root(Div(text=f"<script>{js_code}</script>"))