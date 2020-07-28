#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:25:06 2020

@author: annamoragne
"""

import Working_SEIR
#import working_bubble
import Description
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import Tabs, Panel

#tabA=Panel(child=working_bubble.display, title="General Outbreak")
tabB=Panel(child=row(column(Working_SEIR.widgets, Working_SEIR.data_table), column(Working_SEIR.pops, Working_SEIR.infecteds)), title="Adjustable SEIR Model")
tabC=Panel(child=Description.text_descriptions, title="Model Description")

tabs=Tabs(tabs=[tabB, tabC])

curdoc().add_root(tabs)
curdoc().title="Modeling Infectious Disease Outbreaks"


