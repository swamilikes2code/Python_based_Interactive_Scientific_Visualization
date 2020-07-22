#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 09:12:58 2020

@author: annamoragne
"""

import networkx as nx

from bokeh.io import  show, output_file
from bokeh.models import (BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool,
                          MultiLine, NodesAndLinkedEdges, Plot, Range1d, TapTool, ResetTool)
from bokeh.models import ColumnDataSource, GraphRenderer, Arrow, OpenHead
from bokeh.palettes import Spectral4, RdYlBu8
from bokeh.models.graphs import from_networkx
from bokeh.transform import transform
from bokeh.models.transforms import CustomJSTransform
from bokeh.models.annotations import LabelSet


class_names=['Susceptible', 'Exposed', 'Unknown Asymptomatic Infected', 'Known Asymptomatic Infected', 'Non-Hospitalized Symptomatic Infected', 'Hospitalized Symptomatic Infected', 'Recovered', 'Dead']
needed_edges=[(0, 1), (0, 7), (0, 6), (1, 2), (1,4), (2, 3), (2,6), (2,7), (3, 6), (3,7), (4,5), (4,6), (4,7), (5,6), (5,7), (6,7), (6,0)]

G=nx.DiGraph()
G.add_nodes_from(range(8), name=class_names)
G.add_edges_from(needed_edges)

plot = Plot(plot_width=600, plot_height=600,
            x_range=Range1d(-1.1,1.1), y_range=Range1d(-1.1,1.1))
plot.title.text = "Class Populations for Infectious Disease Outbreak"

graph_renderer = from_networkx(G, nx.circular_layout, scale=1, center=(0,0))

graph_renderer.node_renderer.data_source.add(RdYlBu8, 'color')
graph_renderer.node_renderer.glyph = Circle(size=30, fill_color='color')
graph_renderer.node_renderer.data_source.data['name'] =['Susceptible', 'Exposed', 'Unknown Asymptomatic Infected', 'Known Asymptomatic Infected', 'Non-Hospitalized Symptomatic Infected', 'Hospitalized Symptomatic Infected', 'Recovered', 'Dead']
graph_renderer.node_renderer.selection_glyph = Circle(size=30, fill_color=Spectral4[2])
graph_renderer.node_renderer.hover_glyph = Circle(size=30, fill_color=Spectral4[1])

graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=4)
graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=4)
graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=6)

graph_renderer.selection_policy = NodesAndLinkedEdges()
graph_renderer.inspection_policy = EdgesAndLinkedNodes()

#trying to add arrows to directed graph
pos = nx.layout.spring_layout(G)
nx.draw_networkx_edges(G, pos=pos, edge_color='black', alpha=0.5, arrows=True, arrowstyle='->',rrowsize=10, width=2)
# add the labels to the node renderer data source
source = graph_renderer.node_renderer.data_source
source.data['names'] = class_names

# create a transform that can extract the actual x,y positions
code = """
    var result = new Float64Array(xs.length)
    for (var i = 0; i < xs.length; i++) {
        result[i] = provider.graph_layout[xs[i]][%s]
    }
    return result
"""
xcoord = CustomJSTransform(v_func=code % "0", args=dict(provider=graph_renderer.layout_provider))
ycoord = CustomJSTransform(v_func=code % "1", args=dict(provider=graph_renderer.layout_provider))

# Use the transforms to supply coords to a LabelSet 
labels = LabelSet(x=transform('index', xcoord),
                  y=transform('index', ycoord),
                  text='names', text_font_size="12px",
                  x_offset=0, y_offset=15,
                  source=source, render_mode='canvas')

plot.add_layout(labels)


plot.add_tools(HoverTool(tooltips=None), TapTool(), BoxSelectTool(), ResetTool())
plot.renderers.append(graph_renderer)

output_file("bubble_SEIR.html")
show(plot)




