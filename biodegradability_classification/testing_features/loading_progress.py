from time import sleep
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import Button, Div

d = Div(text = 'Starting', styles = {'color': 'red', 'font-size': '16px'})

b = Button()

def work():
    sleep(2)
    d.text = "Done"
    d.styles = {'color': 'green', 'font-size': '16px'}

def cb():
    d.text = "Loading..."
    d.styles = {'color': 'orange', 'font-size': '16px'}
    curdoc().add_next_tick_callback(work)

b.on_click(cb)

curdoc().add_root(column(d, b))