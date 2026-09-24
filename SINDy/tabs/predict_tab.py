# tabs/predict_tab.py

from bokeh.models import ColumnDataSource, Slider, Button, Select, Div, TextInput, HoverTool
from bokeh.layouts import column, row
from bokeh.plotting import figure
import numpy as np


def _default_initial_condition(saved_data, n_vars):
    """Return one flat initial-condition vector from stored run data.

    Newer Train runs store ``initial_conditions`` as a list of vectors—one
    vector per uploaded trajectory. Older runs stored a single flat vector.
    Predict needs exactly one vector, so use the primary trajectory's IC while
    preserving compatibility with the legacy storage shape.
    """
    fallback = [1.0] + [0.0] * max(0, n_vars - 1)
    stored = saved_data.get('initial_conditions')
    if stored is None:
        return fallback

    values = np.asarray(stored, dtype=float)
    if values.ndim == 1:
        initial_condition = values
    elif values.ndim == 2 and values.shape[0] > 0:
        initial_condition = values[0]
    else:
        return fallback

    if len(initial_condition) != n_vars or not np.isfinite(initial_condition).all():
        return fallback
    return initial_condition.tolist()


def predict_tab_layout(engine, trained_model_storage):
    # -------------------------------------------------------------------------
    # 1. UI Components
    # -------------------------------------------------------------------------
    model_select = Select(title="SELECT MODEL (FROM HISTORY)", options=[], value="")

    # IC input — show the hint based on feature names of selected model
    ic_hint_div = Div(
        text="<i style='font-size:12px;'>Select a model to start.</i>",
        styles={'padding': '8px'}
    )

    # Sample initial condition
    ic_input = TextInput(
        title="Initial Conditions x₀",
        placeholder="e.g. 1.0, 0.0, 0.5, 0.0",
        value="",
        width=300,
    )

    horizon_s = Slider(start=10, end=500, value=100, step=10,
                       title="Prediction Horizon (seconds)")

    status_div = Div(text="", styles={'padding': '4px 0', 'font-size': '13px'})

    # Buttons — same size/style pair as Train tab's TRAIN + DELETE, so the
    # two "action tabs" (Train, Predict) feel like the same app.
    btn_predict = Button(label="PREDICT", button_type="primary", height=50, width=100, disabled=True)
    btn_clear   = Button(label="CLEAR",   button_type="danger",  height=50, width=100)

    # -------------------------------------------------------------------------
    # 2. Update IC hint when choosing a different model
    # -------------------------------------------------------------------------
    def on_model_select_change(attr, old, new):
        if not new:
            return
        try:
            run_id = int(new.replace("Run #", ""))
        except ValueError:
            return
        if run_id not in trained_model_storage:
            return

        saved_data = trained_model_storage[run_id]
        names      = saved_data.get('feature_names', [])
        n_vars     = len(names)

        # Multi-file Train runs keep one IC per trajectory. Prediction starts
        # from the first (primary) trajectory by default; users can still edit
        # the values freely before running the simulation.
        ic_list = _default_initial_condition(saved_data, n_vars)

        if names:
            # convert list to string
            ic_input.value = ", ".join([f"{v:.4f}" for v in ic_list])

            ic_hint_div.text = (
                f"<i style='color:#247008;font-size:12px;'>"
                f"Variables: <b>{', '.join(names)}</b> "
                f"— enter {n_vars} values</i>"
            )
        else:
            ic_hint_div.text = "<i style='color:red;'>⚠ Feature names missing.</i>"
        
        btn_predict.disabled = False

    model_select.on_change('value', on_model_select_change)

    # -------------------------------------------------------------------------
    # 3. Plot
    # -------------------------------------------------------------------------
    p_pred = figure(title="Future Trajectory Prediction",
                    sizing_mode="stretch_width", height=500,
                    x_axis_label="Time (s)", y_axis_label="")
    p_pred.scatter([], [], alpha=0)
    
    # Hovertool for prediction plot. same set up as train tab and test tab
    hover_pred = HoverTool(
        renderers=[],mode="vline",tooltips=[("t", "@t{0.000}")],
    )
    p_pred.add_tools(hover_pred)

    source_pred = ColumnDataSource(data={})
    renderers   = {}
    colors      = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                   "#9467bd", "#8c564b"]

    # -------------------------------------------------------------------------
    # 4. Predict callback
    # -------------------------------------------------------------------------
    def on_predict_click():
        if not model_select.value:
            status_div.text = "<span style='color:red;'>⚠ No model selected.</span>"
            return

        try:
            run_id = int(model_select.value.replace("Run #", ""))
        except ValueError:
            return

        if run_id not in trained_model_storage:
            return

        saved_data     = trained_model_storage[run_id]
        model_instance = saved_data['model_instance']
        names          = saved_data.get('feature_names', [])
        n_vars         = len(names) if names else model_instance.n_features_in_

        # Parse IC from TextInput
        ic_str = ic_input.value.strip()
        if ic_str:
            try:
                x0 = [float(v.strip()) for v in ic_str.split(',')]
                if len(x0) != n_vars:
                    status_div.text = (
                        f"<span style='color:red;'>⚠ Expected {n_vars} values, "
                        f"got {len(x0)}. Using default IC.</span>"
                    )
                    x0 = [1.0] + [0.0] * (n_vars - 1)
            except ValueError:
                status_div.text = (
                    "<span style='color:red;'>⚠ Invalid IC format. Using default.</span>"
                )
                x0 = [1.0] + [0.0] * (n_vars - 1)
        else:
            x0 = [1.0] + [0.0] * (n_vars - 1)

        t_future = np.linspace(0, horizon_s.value, 1000)

        try:
            x_future = engine.simulate_with_model(
                model_instance, x0, t_future)
        except RuntimeError as e:
            status_div.text = (
                f"<span style='color:red;'>⚠ {e} "
                "Try a different x₀, or retrain with a lower degree / "
                "higher sparsity threshold.</span>"
            )
            return

        # Clear plot
        p_pred.renderers    = []
        p_pred.legend.items = []
        renderers.clear()

        # Update data
        var_names = names if names else [f"x{i+1}" for i in range(n_vars)]
        data_dict = {'t': t_future}
        for i, name in enumerate(var_names):
            data_dict[name] = x_future[:, i]
        source_pred.data = data_dict

        # Plot lines
        for i, name in enumerate(var_names):
            renderers[name] = p_pred.line(
                't', name,
                source=source_pred,
                color=colors[i % len(colors)],
                line_width=2,
                legend_label=f"{name}"
            )

        p_pred.legend.click_policy = "hide"
        p_pred.legend.location     = "top_right"
        
        # rebuild tooltip corresponding to var_names of this run
        # Then point hover to the predicted lines just draw above 
        hover_pred.tooltips = [("t", "@t{0.000}")] + [
            (name, f"@{{{name}}}{{0.0000}}") for name in var_names
        ]
        first_renderer = next(iter(renderers.values()), None)
        hover_pred.renderers = [first_renderer] if first_renderer is not None else []
        
        p_pred.title.text = (
            f"Future Trajectory — Run #{run_id} | "
            f"x₀ = [{', '.join([f'{v:.2g}' for v in x0])}]"
        )

        ic_display = ", ".join([f"{name}={v:.2g}" for name, v in zip(var_names, x0)])
        status_div.text = (
            f"<b style='color:#27ae60;'>✅ Predicted {horizon_s.value}s "
            f"from x₀: {ic_display}</b>"
        )

    btn_predict.on_click(on_predict_click)

    # -------------------------------------------------------------------------
    # 4b. Clear callback — resets the plot and IC input back to defaults.
    # Mirrors Train tab's DELETE button visually (same red "danger" action
    # sitting next to the primary action), but here it clears the current
    # prediction view rather than deleting a leaderboard entry.
    # -------------------------------------------------------------------------
    def on_clear_click():
        p_pred.renderers    = []
        p_pred.legend.items = []
        renderers.clear()
        source_pred.data = {}
        # reset hover when clear
        hover_pred.renderers = []
        hover_pred.tooltips  = [("t", "@t{0.000}")]
        p_pred.title.text = "Future Trajectory Prediction"
        status_div.text = ""
        # Re-fill IC with the selected model's stored initial condition
        # (rather than blanking it) so the user doesn't have to re-select
        # the model just to get the hint back.
        if model_select.value:
            on_model_select_change(None, None, model_select.value)

    btn_clear.on_click(on_clear_click)

    # -------------------------------------------------------------------------
    # 5. Update model list (called on tab switch from main.py)
    # -------------------------------------------------------------------------
    def update_model_list():
        options = [f"Run #{id}" for id in sorted(trained_model_storage.keys())]
        model_select.options = options
        if options:
            if model_select.value not in options:
                model_select.value = options[-1]
                # trigger IC hint update
                on_model_select_change(None, None, model_select.value)
        else:
            btn_predict.disabled = True
            ic_hint_div.text = "<i>No trained models available. Train a model first.</i>"
            model_select.value = ""
            ic_input.value = ""

    # -------------------------------------------------------------------------
    # 6. Layout — sidebar (fixed width) + plot (stretch), matching the
    # Train tab's top_row structure so the two tabs feel like one app.
    # -------------------------------------------------------------------------
    sidebar = column(
        model_select,
        ic_hint_div,
        ic_input,
        horizon_s,
        row(btn_predict, btn_clear),
        status_div,
        width=320,
    )

    layout = column(
        row(sidebar, column(p_pred, sizing_mode="stretch_width"),
            sizing_mode="stretch_width"),
        sizing_mode="stretch_width"
    )

    return layout, update_model_list
