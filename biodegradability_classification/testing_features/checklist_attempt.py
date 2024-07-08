import pandas as pd
import numpy as np
from bokeh.events import ButtonClick
from bokeh.io import curdoc, show
from bokeh.layouts import layout, column
from bokeh.models import Div
from bokeh.models.callbacks import CustomJS
from bokeh.models.dom import HTML
from numpy.random import random

from bokeh.core.enums import MarkerType
from bokeh.plotting import figure, show

#p = figure(title="Bokeh Markers", toolbar_location=None)
#p.grid.grid_line_color = None
#p.axis.visible = False
#p.y_range.flipped = True

#N = 10

#for i, marker in enumerate(MarkerType):
#    x = i % 4
#    y = (i // 4) * 4 + 1

#    p.scatter(random(N)+2*x, random(N)+y, marker=marker, size=14,
#              line_color="navy", fill_color="orange", alpha=0.5)

#    p.text(2*x+0.5, y+2.5, text=[marker],
#           text_color="firebrick", text_align="center", text_font_size="13px")

## Set the path to your static directory
# static_path = "static/check_icon.png"

## Create a relative URL for the image
# image_html = f'<img src="{static_path}" alt="Check Icon" style="width:50px;height:50px;">'

## Create a Bokeh Div with the image HTML
#image_div = Div(text=image_html, width=200, height=100)

predict_check_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.0;
            margin: 0;
            padding: 20px;
            background-color: #f4f4f4;
        }}
        .container {{
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .column_1 {{
            float: left;
            width: 25%;
        }}

        .column_3 {{
            float: left;
            width: 75%;
        }}

        .row:after {{
            content: "";
            display: table;
            clear: both;
        }}

        h1 {{
            text-align: center;
            color: #333;
        }}
        h2 {{
            color: #444;
            border-bottom: 2px solid #ddd;
            padding-bottom: 1px;
        }}
        h3 {{
            color: #555;
        }}
        p {{
            margin: 5px 0;
            max-width: 575px;
        }}
        .section {{
            margin-bottom: 20px;
            padding: 10px;
            background-color: #fafafa;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .highlight {{
            background-color: #e7f3fe;
            border-left: 5px solid #2196F3;
            padding: 2px 5px;
        }}    </style>
</head>
<body>
    <div class="container">
        <div class="row">
            <div class="column_1">
                <h1>{}</h1>
            </div>

            <div class="column_3">
                <h2>{}STEP ONE:{}</h2>
                <p>{}Select Features{}</p>
            </div>
        </div>

        <div class="row">
            <div class="column_1">
                <h1>{}</h1>
            </div>

            <div class="column_3">
                <h2>{}STEP TWO:{}</h2>
                <p>{}Split Data{}</p>
            </div>
        </div>

        <div class="row">
            <div class="column_1">
                <h1>{}</h1>
            </div>

            <div class="column_3">
                <h2>{}STEP THREE:{}</h2>
                <p>{}Select ML Algorithm{}</p>
            </div>
        </div>

        <div class="row">
            <div class="column_1">
                <h1>{}</h1>
            </div>

            <div class="column_3">
                <h2>STEP FOUR (opt.):</h2>
                <p>Tune Hyperparameters</p>
            </div>
        </div>

        <div class="row">
            <div class="column_1">
                <h1>{}</h1>
            </div>

            <div class="column_3">
                <h2>{}STEP FIVE:{}</h2>
                <p>{}Test Model{}</p>
            </div>
        </div>

        <div class="row">
            <div class="column_1">
                <h1>{}</h1>
            </div>

            <div class="column_3">
                <h2>STEP SIX (opt.):</h2>
                <p>Predict</p>
            </div>
        </div>

    </div>
</body>
</html>
"""

formatted_predict_check = predict_check_template.format('*', '', '', '', '', '*', '', '', '', '', '*', '', '', '', '', '~', '*', '', '', '', '', '~')
predict_check = Div(text=formatted_predict_check)

curdoc().add_root(predict_check)
# curdoc().add_root(p)