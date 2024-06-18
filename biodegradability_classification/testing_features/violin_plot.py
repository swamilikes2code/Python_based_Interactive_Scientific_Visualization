import numpy as np
import holoviews as hv
from holoviews import opts
from bokeh.io import curdoc
from bokeh.plotting import figure
import panel as pn
hv.extension('bokeh')

# hv.help(hv.Violin)
violin = hv.Violin([0.56, 0.64, 0.76, 0.84, 0.68, 0.76, 0.56, 0.92, 0.76, 0.84],vdims='Accuracy')
violin.opts(opts.Violin(inner='stick'))
bokeh_server = pn.Row(violin).show(port=5010)