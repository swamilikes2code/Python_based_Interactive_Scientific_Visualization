import pandas as pd
import numpy as np
from bokeh.events import ButtonClick
from bokeh.io import curdoc, show
from bokeh.layouts import layout, column
from bokeh.models import Div
from bokeh.models.callbacks import CustomJS
from bokeh.models.dom import HTML


# Set the path to your static directory
static_path = "static/check_icon.png"

# Create a relative URL for the image
image_html = f'<img src="{static_path}" alt="Check Icon" style="width:50px;height:50px;">'

# Create a Bokeh Div with the image HTML
image_div = Div(text=image_html, width=200, height=100)

html_test_template = """
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
        .column_4 {{
            float: left;
            width: 25%;
        }}

        .column_2 {{
            float: left;
            width: 50%;
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
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="section">
            <h2>STEP ONE:</h2>
            <p>Select Features</p>
        </div>

        <div class="row">
            <h2>STEP TWO:</h2>
            <div class = "column_2">
                <img src="static/check_icon.png" alt="Check Icon" style="width:50px;height:50px;">
            <div class="column_2">
                <p>Split Data</p>
                <p>{}</p>
            </div>
        </div>
    </div>
</body>
</html>
"""

html_test = html_test_template('N/A')

curdoc().add_root(column(image_div))