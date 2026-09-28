from pathlib import Path

import numpy as np
from PIL import Image

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    Div,
    Slider,
    RadioButtonGroup,
    LabelSet,
)
from bokeh.plotting import figure



# MODE 1 — FORWARD MODEL
# ONLY m1 IS VARIABLE


def solve_no_recycle(m1):
    x1K = 0.333
    m2 = 1466.6 * (m1 / 4500)

    xsolK = 0.364
    cake_crystal_frac = 0.95

    if m1 <= 0:
        raise ValueError("m1 must be positive.")

    m3 = m1 - m2
    x3K = (m1 * x1K) / m3

    if x3K <= xsolK:
        raise ValueError("Not supersaturated → no crystals form.")

    d = (1 - cake_crystal_frac) / cake_crystal_frac

    m4 = m3 * (x3K - xsolK) / (
        1 - xsolK + d * xsolK
    )

    m5 = d * m4
    m6 = m3 - m4 - m5

    return {
        "m1_fresh_feed": m1,
        "m2_evaporated_water": m2,
        "m3_to_crystallizer": m3,
        "x3K_after_evap": x3K,
        "m4_crystals": m4,
        "m5_solution_in_cake": m5,
        "m6_filtrate": m6,
    }


def solve_with_recycle(m1):

    x1K = 0.333

    m3 = 2950 * (m1 / 4500)

    xsolK = 0.364
    cake_crystal_frac = 0.95
    x4K = 0.494

    if m1 <= 0:
        raise ValueError("m1 must be positive.")

    d = (
        1 - cake_crystal_frac
    ) / cake_crystal_frac

    m6 = 0

    m5 = (
        m1 * x1K
    ) / (
        1 + d * xsolK
    )

    m_solution_in_cake = d * m5

    m4 = (
        m5
        * (1 - xsolK)
        / (x4K - xsolK)
    )

    m2 = m4 + m3

    m7 = m2 - m1

    if m7 < 0:
        raise ValueError(
            "Negative recycle flow."
        )

    x2K = (
        m1 * x1K
        + m7 * xsolK
    ) / m2

    return {
        "m1_fresh_feed":
            m1,

        "m2_mixed_feed_to_evaporator":
            m2,

        "x2K_mixed_feed":
            x2K,

        "m3_evaporated_water":
            m3,

        "m4_to_crystallizer":
            m4,

        "x4K_after_evap":
            x4K,

        "m5_crystals":
            m5,

        "m6_filtrate":
            m6,

        "m_solution_in_cake":
            m_solution_in_cake,

        "m7_recycle":
            m7,

        "recycle_ratio":
            m7 / m1,
    }



# LOAD MODE 1 STATIC IMAGES


def load_rgba_image(path):

    img = Image.open(
        path
    ).convert("RGBA")

    arr = np.flipud(
        np.asarray(
            img,
            dtype=np.uint8
        )
    )

    view = (
        arr.view(
            dtype=np.uint32
        )
        .reshape(
            arr.shape[:2]
        )
    )

    h = arr.shape[0]
    w = arr.shape[1]

    return view, w, h


RECYCLE_IMAGE_PATH = (
    Path(__file__)
    .with_name("recycle.png")
)

NO_RECYCLE_IMAGE_PATH = (
    Path(__file__)
    .with_name("no_recycle.png")
)


if not RECYCLE_IMAGE_PATH.exists():
    raise FileNotFoundError(
        "recycle.png not found."
    )


if not NO_RECYCLE_IMAGE_PATH.exists():
    raise FileNotFoundError(
        "no_recycle.png not found."
    )


recycle_img, recycle_w, recycle_h = (
    load_rgba_image(
        str(RECYCLE_IMAGE_PATH)
    )
)


no_recycle_img, no_recycle_w, no_recycle_h = (
    load_rgba_image(
        str(NO_RECYCLE_IMAGE_PATH)
    )
)



# MODE 1 WIDGETS


mode1_system_toggle = RadioButtonGroup(
    labels=[
        "No Recycle",
        "With Recycle",
    ],
    active=0,
)


mode1_m1_slider = Slider(
    title="Fresh Feed, m1 (kg/h)",
    start=1000,
    end=8000,
    value=4500,
    step=100,
    width=360,
)


mode1_summary_div = Div(
    width=420,
    height=430,
)


mode1_error_div = Div(
    width=420
)


mode1_bar_source = ColumnDataSource(
    data=dict(
        category=[],
        value=[],
        label=[],
    )
)


mode1_flow_source = ColumnDataSource(
    data=dict(
        image=[no_recycle_img],
        x=[0],
        y=[0],
        dw=[no_recycle_w],
        dh=[no_recycle_h],
    )
)



# MODE 1 FLOW DIAGRAM


mode1_flow_fig = figure(
    width=720,
    height=360,
    x_range=(
        0,
        no_recycle_w
    ),
    y_range=(
        0,
        no_recycle_h
    ),
    toolbar_location=None,
    title="No-Recycle Process Flow Diagram",
)


mode1_flow_fig.image_rgba(
    image="image",
    x="x",
    y="y",
    dw="dw",
    dh="dh",
    source=mode1_flow_source,
)


mode1_flow_fig.axis.visible = False
mode1_flow_fig.grid.visible = False

mode1_flow_fig.outline_line_color = (
    "#d9d9d9"
)

mode1_flow_fig.title.text_font_size = (
    "13pt"
)



# MODE 1 BAR CHART

mode1_bar_fig = figure(
    width=720,
    height=390,
    x_range=[],
    title="Calculated Stream Flow Rates",
    toolbar_location=None,
)


mode1_bar_fig.vbar(
    x="category",
    top="value",
    width=0.62,
    source=mode1_bar_source,
    color="#1565C0",
)


mode1_labels = LabelSet(
    x="category",
    y="value",
    text="label",
    source=mode1_bar_source,
    y_offset=5,
    text_align="center",
    text_font_size="9pt",
)


mode1_bar_fig.add_layout(
    mode1_labels
)


mode1_bar_fig.xgrid.grid_line_color = None

mode1_bar_fig.yaxis.axis_label = (
    "Flow Rate (kg/h)"
)

mode1_bar_fig.xaxis.major_label_orientation = (
    0.75
)

mode1_bar_fig.min_border_bottom = 90

mode1_bar_fig.title.text_font_size = (
    "13pt"
)

mode1_bar_fig.outline_line_color = (
    "#d9d9d9"
)



# MODE 1 UPDATE FUNCTION

def update_mode1(attr, old, new):

    try:

        m1 = float(
            mode1_m1_slider.value
        )

        recycle = (
            mode1_system_toggle.active
            == 1
        )

        if recycle:

            s = solve_with_recycle(
                m1
            )

            categories = [
                "m2 mixed",
                "m3 evap",
                "m4 cryst",
                "m5 crystals",
                "m7 recycle",
            ]

            values = [
                s[
                    "m2_mixed_feed_to_evaporator"
                ],

                s[
                    "m3_evaporated_water"
                ],

                s[
                    "m4_to_crystallizer"
                ],

                s[
                    "m5_crystals"
                ],

                s[
                    "m7_recycle"
                ],
            ]


            mode1_flow_source.data = dict(
                image=[
                    recycle_img
                ],
                x=[0],
                y=[0],
                dw=[
                    recycle_w
                ],
                dh=[
                    recycle_h
                ],
            )


            mode1_flow_fig.x_range.end = (
                recycle_w
            )

            mode1_flow_fig.y_range.end = (
                recycle_h
            )


            mode1_flow_fig.title.text = (
                "Recycle Process Flow Diagram"
            )


            mode1_summary_div.text = f"""
            <div style="
                padding:16px;
                border:1px solid #d9d9d9;
                border-radius:10px;
                background:#fafafa;
                line-height:1.55;
            ">

            <h2 style="
                margin-top:0;
                color:#1565C0;
            ">
                Mode 1 — With Recycle
            </h2>

            <b>Variable Input:</b><br>

            m1 = {m1:.2f} kg/h

            <br><br>

            <b>Computed Streams:</b><br>

            m2 mixed feed =
            {s["m2_mixed_feed_to_evaporator"]:.2f}
            <br>

            x2K mixed feed =
            {s["x2K_mixed_feed"]:.4f}
            <br>

            m3 evaporated =
            {s["m3_evaporated_water"]:.2f}
            <br>

            m4 crystallizer =
            {s["m4_to_crystallizer"]:.2f}
            <br>

            m5 crystals =
            {s["m5_crystals"]:.2f}
            <br>

            m7 recycle =
            {s["m7_recycle"]:.2f}

            <br><br>

            <b>Recycle Ratio:</b>

            {s["recycle_ratio"]:.4f}

            </div>
            """


        else:

            s = solve_no_recycle(
                m1
            )

            categories = [
                "m2 evap",
                "m3 cryst",
                "m4 crystals",
                "m5 cake soln",
                "m6 filtrate",
            ]


            values = [
                s[
                    "m2_evaporated_water"
                ],

                s[
                    "m3_to_crystallizer"
                ],

                s[
                    "m4_crystals"
                ],

                s[
                    "m5_solution_in_cake"
                ],

                s[
                    "m6_filtrate"
                ],
            ]


            mode1_flow_source.data = dict(
                image=[
                    no_recycle_img
                ],
                x=[0],
                y=[0],
                dw=[
                    no_recycle_w
                ],
                dh=[
                    no_recycle_h
                ],
            )


            mode1_flow_fig.x_range.end = (
                no_recycle_w
            )

            mode1_flow_fig.y_range.end = (
                no_recycle_h
            )


            mode1_flow_fig.title.text = (
                "No-Recycle Process Flow Diagram"
            )


            mode1_summary_div.text = f"""
            <div style="
                padding:16px;
                border:1px solid #d9d9d9;
                border-radius:10px;
                background:#fafafa;
                line-height:1.55;
            ">

            <h2 style="
                margin-top:0;
                color:#1565C0;
            ">
                Mode 1 — No Recycle
            </h2>

            <b>Variable Input:</b><br>

            m1 = {m1:.2f} kg/h

            <br><br>

            <b>Computed Streams:</b><br>

            m2 evap =
            {s["m2_evaporated_water"]:.2f}
            <br>

            m3 crystallizer =
            {s["m3_to_crystallizer"]:.2f}
            <br>

            x3K after evap =
            {s["x3K_after_evap"]:.4f}
            <br>

            m4 crystals =
            {s["m4_crystals"]:.2f}
            <br>

            m5 cake soln =
            {s["m5_solution_in_cake"]:.2f}
            <br>

            m6 filtrate =
            {s["m6_filtrate"]:.2f}

            </div>
            """


        mode1_bar_source.data = dict(
            category=categories,
            value=values,
            label=[
                f"{v:.1f}"
                for v in values
            ],
        )


        mode1_bar_fig.x_range.factors = (
            categories
        )


        mode1_error_div.text = ""


    except Exception as e:

        mode1_error_div.text = (
            "<b style='color:red'>"
            "Error:"
            "</b> "
            f"{str(e)}"
        )



# MODE 1 CALLBACKS


mode1_m1_slider.on_change(
    "value",
    update_mode1,
)


mode1_system_toggle.on_change(
    "active",
    update_mode1,
)



# MODE 1 LAYOUT


mode1_title_div = Div(
    text="""
    <h1 style="
        margin-bottom:5px;
    ">
        Mode 1 — Forward Process Model
    </h1>

    <div style="
        color:#555;
        line-height:1.5;
    ">
        Students adjust only the fresh feed
        flow rate, m1, and observe how stream
        flow rates change. The toggle switches
        between no-recycle and recycle systems.
    </div>
    """,
    width=420,
)


mode1_controls = column(
    mode1_title_div,

    Div(
        text="<br>"
    ),

    mode1_system_toggle,

    Div(
        text="<br>"
    ),

    mode1_m1_slider,

    mode1_error_div,

    mode1_summary_div,

    width=440,
)


mode1_plots = column(
    mode1_flow_fig,
    mode1_bar_fig,
    sizing_mode="fixed",
)


mode1_layout = row(
    mode1_controls,
    mode1_plots,
)



# MODE 2 — REVERSE PROCESS DESIGN
# RECYCLE ONLY
# INPUT = desired crystal production m5


def solve_with_recycle_inverse(m5):

    x1K = 0.333
    x1W = 0.667

    x4K = 0.494
    x4W = 0.506

    x6K = 0.364
    x6W = 0.636

    x7K = 0.364
    x7W = 0.636

    cake_frac = 0.95


    if m5 <= 0:

        raise ValueError(
            "m5 must be positive."
        )


    m6 = (
        (1 - cake_frac)
        / cake_frac
    ) * m5


    m1 = (
        m5
        + m6 * x6K
    ) / x1K


    m3 = (
        m1
        - m5
        - m6
    )


    m4 = (
        m5
        * (1 - x6K)
        / (x4K - x6K)
    )


    m2 = (
        m3
        + m4
    )


    m7 = (
        m2
        - m1
    )


    if (
        m1 <= 0
        or m2 <= 0
        or m3 <= 0
        or m4 <= 0
        or m6 < 0
        or m7 < 0
    ):

        raise ValueError(
            "Calculated a nonphysical "
            "negative flow rate."
        )


    x2K = (
        m1 * x1K
        + m7 * x7K
    ) / m2


    x2W = (
        1 - x2K
    )


    return {

        "m1_fresh_feed":
            m1,

        "x1K_fresh_feed":
            x1K,

        "x1W_fresh_feed":
            x1W,

        "m2_mixed_feed_to_evaporator":
            m2,

        "x2K_mixed_feed":
            x2K,

        "x2W_mixed_feed":
            x2W,

        "m3_evaporated_water":
            m3,

        "m4_to_crystallizer":
            m4,

        "x4K_after_evap":
            x4K,

        "x4W_after_evap":
            x4W,

        "m5_crystals":
            m5,

        "m6_solution_in_cake":
            m6,

        "x6K_solution_in_cake":
            x6K,

        "x6W_solution_in_cake":
            x6W,

        "m7_recycle":
            m7,

        "x7K_recycle":
            x7K,

        "x7W_recycle":
            x7W,

        "recycle_ratio":
            m7 / m1,
    }



# MODE 2 WIDGETS


mode2_m5_slider = Slider(
    title=(
        "Desired Crystal Production, "
        "m5 (kg/h)"
    ),
    start=200,
    end=3000,
    value=1470,
    step=10,
    width=390,
)


mode2_summary_div = Div(
    width=430,
    height=470,
)


mode2_error_div = Div(
    width=430,
)


mode2_bar_source = ColumnDataSource(
    data=dict(
        category=[],
        value=[],
        label=[],
    )
)



# MODE 2 DESCRIPTION PANEL


mode2_explain_div = Div(
    text="""

    <div style="
        padding:16px;
        border:1px solid #d9d9d9;
        border-radius:10px;
        background:#fafafa;
        line-height:1.55;
        width:720px;
    ">

    <h2 style="
        margin-top:0;
        color:#1565C0;
    ">
        Mode 2 — Reverse Process Design
    </h2>

    This module determines the required
    operating stream flow rates for a
    specified crystal production target.

    <br><br>

    The user selects the desired crystal
    output rate (<b>m5</b>), and the model
    calculates:

    <ul style="margin-top:8px;">

        <li>
            Fresh feed requirement (m1)
        </li>

        <li>
            Mixed feed to the evaporator (m2)
        </li>

        <li>
            Evaporated water flow rate (m3)
        </li>

        <li>
            Crystallizer inlet flow rate (m4)
        </li>

        <li>
            Solution retained in the crystal
            cake (m6)
        </li>

        <li>
            Recycle stream flow rate (m7)
        </li>

    </ul>

    This mode demonstrates how a production
    target can be translated into required
    process operating conditions.

    </div>

    """,
    width=760,
)



# MODE 2 BAR CHART


mode2_bar_fig = figure(
    width=760,
    height=430,
    x_range=[],
    title="Required Stream Flow Rates",
    toolbar_location=None,
)


mode2_bar_fig.vbar(
    x="category",
    top="value",
    width=0.62,
    source=mode2_bar_source,
    color="#1565C0",
)


mode2_labels = LabelSet(
    x="category",
    y="value",
    text="label",
    source=mode2_bar_source,
    y_offset=5,
    text_align="center",
    text_font_size="9pt",
)


mode2_bar_fig.add_layout(
    mode2_labels
)


mode2_bar_fig.xgrid.grid_line_color = None

mode2_bar_fig.yaxis.axis_label = (
    "Flow Rate (kg/h)"
)

mode2_bar_fig.xaxis.major_label_orientation = (
    0.75
)

mode2_bar_fig.min_border_bottom = 100

mode2_bar_fig.title.text_font_size = (
    "13pt"
)

mode2_bar_fig.outline_line_color = (
    "#d9d9d9"
)



# MODE 2 UPDATE FUNCTION


def update_mode2(attr, old, new):

    try:

        m5 = float(
            mode2_m5_slider.value
        )


        s = solve_with_recycle_inverse(
            m5
        )


        categories = [
            "m1 fresh",
            "m2 mixed",
            "m3 evap",
            "m4 cryst",
            "m5 crystals",
            "m6 cake soln",
            "m7 recycle",
        ]


        values = [
            s[
                "m1_fresh_feed"
            ],

            s[
                "m2_mixed_feed_to_evaporator"
            ],

            s[
                "m3_evaporated_water"
            ],

            s[
                "m4_to_crystallizer"
            ],

            s[
                "m5_crystals"
            ],

            s[
                "m6_solution_in_cake"
            ],

            s[
                "m7_recycle"
            ],
        ]


        mode2_bar_source.data = dict(

            category=categories,

            value=values,

            label=[
                f"{v:.1f}"
                for v in values
            ],
        )


        mode2_bar_fig.x_range.factors = (
            categories
        )


        mode2_summary_div.text = f"""

        <div style="
            padding:16px;
            border:1px solid #d9d9d9;
            border-radius:10px;
            background:#fafafa;
            line-height:1.55;
        ">

        <h2 style="
            margin-top:0;
            color:#1565C0;
        ">
            Mode 2 — Reverse Process Design
        </h2>

        <b>
            Specified Production Target:
        </b>
        <br>

        m5 crystal production =
        {s["m5_crystals"]:.2f} kg/h

        <br><br>

        <b>
            Calculated Operating Requirements:
        </b>
        <br>

        m1 fresh feed =
        {s["m1_fresh_feed"]:.2f} kg/h

        <br>

        m2 mixed feed to evaporator =
        {s["m2_mixed_feed_to_evaporator"]:.2f}
        kg/h

        <br>

        m3 evaporated water =
        {s["m3_evaporated_water"]:.2f}
        kg/h

        <br>

        m4 to crystallizer =
        {s["m4_to_crystallizer"]:.2f}
        kg/h

        <br>

        m6 solution in cake =
        {s["m6_solution_in_cake"]:.2f}
        kg/h

        <br>

        m7 recycle =
        {s["m7_recycle"]:.2f}
        kg/h

        <br><br>

        <b>
            Key Compositions:
        </b>
        <br>

        x1K fresh feed =
        {s["x1K_fresh_feed"]:.3f}

        <br>

        x2K mixed feed =
        {s["x2K_mixed_feed"]:.4f}

        <br>

        x4K after evaporator =
        {s["x4K_after_evap"]:.3f}

        <br>

        x7K recycle =
        {s["x7K_recycle"]:.3f}

        <br><br>

        <b>
            Recycle Ratio:
        </b>

        {s["recycle_ratio"]:.4f}

        </div>

        """


        mode2_error_div.text = ""


    except Exception as e:

        mode2_error_div.text = (
            "<b style='color:red'>"
            "Error:"
            "</b> "
            f"{str(e)}"
        )



# MODE 2 CALLBACK


mode2_m5_slider.on_change(
    "value",
    update_mode2,
)



# MODE 2 LAYOUT


mode2_title_div = Div(
    text="""

    <h1 style="
        margin-bottom:5px;
    ">
        Mode 2 — Reverse Process Design
    </h1>

    <div style="
        color:#555;
        line-height:1.5;
    ">
        Specify a crystal production target
        and calculate the recycle-process flow
        rates required to meet it.
    </div>

    """,
    width=430,
)


mode2_controls = column(
    mode2_title_div,

    Div(
        text="<br>"
    ),

    mode2_m5_slider,

    mode2_error_div,

    mode2_summary_div,

    width=450,
)


mode2_plots = column(
    mode2_explain_div,
    mode2_bar_fig,
    sizing_mode="fixed",
)


mode2_layout = row(
    mode2_controls,
    mode2_plots,
)



# MAIN WEBSITE HEADER


page_header = Div(
    text="""

    <div style="
        text-align:center;
        padding-top:30px;
        padding-bottom:28px;
        border-bottom:1px solid #d5d5d5;
        background:#ffffff;
    ">

        <h1 style="
            margin:0;
            font-size:38px;
            font-weight:700;
            color:#111111;
        ">
            CHE 031
        </h1>

        <div style="
            margin-top:8px;
            font-size:24px;
            font-weight:500;
            color:#333333;
        ">
            Evaporative Crystallization
            Interactive Visualization
        </div>

    </div>

    """,
    width=1300,
)



# MAIN WEBSITE NAVIGATION

mode_navigation = RadioButtonGroup(

    labels=[
        "Mode 1 — Forward Model",
        "Mode 2 — Reverse Process Design",
    ],

    active=0,

    width=700,

    height=50,
)



# CONTENT HOLDER


mode_content = column(
    mode1_layout,
    width=1300,
)



# SWITCH BETWEEN MODES


def switch_mode(attr, old, new):

    if mode_navigation.active == 0:

        mode_content.children = [
            mode1_layout
        ]

    else:

        mode_content.children = [
            mode2_layout
        ]


mode_navigation.on_change(
    "active",
    switch_mode,
)



# INITIALIZE BOTH MODES


update_mode1(
    None,
    None,
    None,
)

update_mode2(
    None,
    None,
    None,
)



# NAVIGATION BAR AREA


navigation_area = row(

    mode_navigation,

    width=1300,

    margin=(
        18,
        0,
        24,
        35,
    ),
)



# COMPLETE WEBSITE


website = column(

    page_header,

    navigation_area,

    mode_content,

    width=1300,

    margin=(
        0,
        25,
        30,
        25,
    ),
)



# BOKEH DOCUMENT

curdoc().add_root(
    website
)


curdoc().title = (
    "CHE 031 Interactive Visualization"
)