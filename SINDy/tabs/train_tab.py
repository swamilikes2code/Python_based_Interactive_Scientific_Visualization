# =============================================================================
# tabs/train_tab.py
#
# PURPOSE
# -------
# Renders the "Train & Validate" tab of the SINDy Expert System.
# Responsibilities:
#   1. Let the user pick a dataset (pre-set system or custom CSV upload).
#      Custom upload accepts MULTIPLE CSV files — trajectories of the SAME
#      system recorded from DIFFERENT initial conditions — and trains ONE
#      robust model pooled across all of them.
#   2. Run an automatic data-analysis heuristic ("AI Suggester") that
#      recommends starting values for polynomial degree and sparsity
#      threshold based on linearity / periodicity / noise level.
#   3. Fit a SINDy model on a random train/validation split.
#   4. Show the fitted trajectory (one simulation line per initial
#      condition), a leaderboard of all past runs, and residual diagnostics
#      (time-domain, frequency-domain, and true-vs-predicted scatter).
#   5. Allow viewing/deleting any past run from the leaderboard.
# =============================================================================

from bokeh.models import (ColumnDataSource, Slider, Div, Button,
                          Select, DataTable, TableColumn, HTMLTemplateFormatter, FileInput, TextInput, CheckboxButtonGroup, HoverTool)
from bokeh.layouts import column, row, Spacer
from bokeh.plotting import figure
import pandas as pd
import numpy as np
import os
import copy
import base64
import io
import warnings
from engine.suggester import analyze_data_linearity
from engine.check_datafile import check_upload_size, validate_dataframe, validate_trajectory_set
from bokeh.io import curdoc


def _compact_trajectory_label(filename, index, max_stem_length=20):
    """Build a compact, distinctive button label for one trajectory file."""
    stem = os.path.splitext(os.path.basename(str(filename)))[0]
    if len(stem) > max_stem_length:
        left = max_stem_length // 2
        right = max_stem_length - left - 1
        stem = f"{stem[:left]}…{stem[-right:]}"
    return f"IC {index + 1} · {stem}"


def train_tab_layout(engine, trained_model_storage):
    """
    Build the Bokeh layout for the Train & Validate tab.

    Parameters
    ----------
    engine : SINDyEngine
        Shared engine instance (holds the pySINDy model, fit/simulate/
        diagnostics methods). Same instance is passed to Test/Predict tabs
        so that trained models can be reused across tabs.
    trained_model_storage : dict
        Shared in-memory store: {run_id: {model_instance, metrics, plot_data,
        diagnostics, ...}}. Acts as the "database" for the leaderboard and
        is read by the Test/Predict/Ensemble tabs to let the user pick a
        trained run.

    Returns
    -------
    bokeh.layouts.column
        The complete tab layout, ready to be added to a Bokeh document.
    """

    def apply_suggestion(df, prefix_msg=""):
        """
        Run analyze_data_linearity() and push the result into the UI:
        updates poly_s / thr_s slider values and displays the reasoning
        text in upload_status. (library is intentionally NOT auto-applied.)
        """
        lib, deg, thr, reason = analyze_data_linearity(df)
        if reason.strip().lower().startswith("error"):
            suggestion_div.text = ""
            upload_status.text = f"<span style='color:#e74c3c;'>⚠ {reason}</span>"
            return

        poly_s.value = deg
        thr_s.value = thr
        if prefix_msg:
            upload_status.text = prefix_msg
        suggestion_div.text = f"Suggestion: {reason}"

    # =========================================================================
    # SECTION 1 — DATA SOURCE SELECTION
    # Dropdown for pre-set systems + custom CSV upload widget.
    # Custom upload accepts MULTIPLE files (multiple=True): each file is a
    # trajectory of the same system from a different initial condition.
    # =========================================================================

    system_options = [
        ("cs_train_data.csv",   "Coupled Spring-Mass (Polynomial)"),
        ("vanderpol_train.csv", "Van der Pol Oscillator (Polynomial)"),
        ("pendulum_train.csv",  "Nonlinear Pendulum (Fourier/Combined)"),
        ("timedep_train.csv",   "Forced Oscillator (Combined)"),
        ("custom_upload",       "Upload your own data")
    ]

    file_select = Select(title="SELECT SYSTEM", options=system_options,
                         value="cs_train_data.csv")
    _SYSTEM_LABELS = dict(system_options)

    # File upload widget — hidden until the user picks "Upload your own data".
    # multiple=True lets the user pick several CSVs at once (Ctrl/Shift-click).
    # Requires Bokeh >= 2.4.
    file_input = FileInput(accept=".csv", multiple=True, visible=False)

    # Cumulative store of everything the user has uploaded so far.
    # Each pick event in Bokeh only carries the NEWLY chosen files, so we
    # append them here across events. Re-uploading a name replaces it.
    _uploaded_files = []   # [{'name': str, 'b64': str}]

    upload_status = Div(
        text="", styles={'color': "#247008", 'font-size': '13px'})
    suggestion_div = Div(
        text="", styles={'color': "#2c3e50", 'font-size': '13px'})

    upload_list_div = Div(
        text="", styles={'color': '#2c3e50', 'font-size': '12px',
                         'padding': '2px 0'})
    btn_clear_files = Button(label="CLEAR FILES", button_type="warning",
                             width=120, visible=False)

    def _refresh_upload_list():
        """Re-render the uploaded-file list and toggle the clear button."""
        if _uploaded_files:
            items = "<br>".join(
                f"&bull; {f['name']}" for f in _uploaded_files)
            upload_list_div.text = (
                f"<b>{len(_uploaded_files)} file(s) ready "
                f"(different ICs of the SAME system):</b><br>{items}")
            btn_clear_files.visible = True
        else:
            upload_list_div.text = ""
            btn_clear_files.visible = False

    def _decode_upload(f):
        """Decode one stored upload into a float64 DataFrame."""
        return pd.read_csv(io.BytesIO(base64.b64decode(f['b64']))).astype(np.float64)

    def on_file_select_change(attr, old, new):
        """
        Toggle the upload widget visibility and, for pre-set systems,
        immediately load the CSV and run the AI Suggester so the sliders
        are pre-filled before the user even presses Train.
        """
        show_upload = (new == "custom_upload")
        file_input.visible = show_upload
        upload_list_div.visible = show_upload
        btn_clear_files.visible = show_upload and bool(_uploaded_files)
        if show_upload:
            upload_status.text = ("Upload one or more CSVs of the SAME system "
                                  "from different initial conditions. Columns "
                                  "must match: t, x1, x2...")
        else:
            path = os.path.join('data', new)
            if os.path.exists(path):
                df = pd.read_csv(path).astype(np.float64)
                val_err = validate_dataframe(df)
                if val_err:
                    suggestion_div.text = ""
                    upload_status.text = f"<span style='color:#e74c3c;'>⚠ {val_err}</span>"
                    return
                apply_suggestion(df, f"Selected system file: {new}")
            else:
                upload_status.text = f"⚠ Pre-set file not found at {path}"

    file_select.on_change('value', on_file_select_change)

    def upload_to_local_drive(attr, old, new):
        """
        Callback fired when FileInput receives new file(s). IMPORTANT: Bokeh
        sends 'value', 'filename', and 'mime_type' as SEPARATE ModelChanged
        events within the same PATCH-DOC message, applied one after another.
        This on_change('value', ...) callback fires as soon as the 'value'
        event is applied — BEFORE the 'filename' event later in the same
        message has been applied — so reading file_input.filename
        synchronously here raises UnsetValueError, not just an empty value.

        Fix: defer the actual processing to the next event-loop tick via
        add_next_tick_callback. By the time that runs, the whole message
        (all 3 property updates) has finished being applied, so filename is
        guaranteed to be set.
        """
        if not new:
            return
        curdoc().add_next_tick_callback(lambda: _process_uploaded_files(new))

    def _process_uploaded_files(new):
        """The actual upload-handling logic, run one tick after 'value' changed."""
        payloads = new if isinstance(new, list) else [new]

        # Safe now — filename/mime_type have been applied by this point.
        try:
            names = file_input.filename or []
        except Exception:
            names = []
        if not isinstance(names, list):
            names = [names]

        for i, b64 in enumerate(payloads):
            size_err = check_upload_size(b64)
            name = names[i] if i < len(
                names) else f"file_{len(_uploaded_files)+1}.csv"
            if size_err:
                upload_status.text = f"<span style='color:red;'>⚠ {name}: {size_err}</span>"
                continue
            _uploaded_files[:] = [f for f in _uploaded_files if f['name'] != name]
            _uploaded_files.append({'name': name, 'b64': b64})

        _refresh_upload_list()

        dfs = []
        for f in _uploaded_files:
            try:
                df_k = _decode_upload(f)
            except Exception as e:
                upload_status.text = f"⚠ Error reading {f['name']}: {e}"
                return
            val_err = validate_dataframe(df_k)
            if val_err:
                suggestion_div.text = ""
                upload_status.text = f"<span style='color:#e74c3c;'>⚠ {f['name']}: {val_err}</span>"
                return
            dfs.append(df_k)

        set_err = validate_trajectory_set(dfs)
        if set_err:
            suggestion_div.text = ""
            upload_status.text = f"<span style='color:#e74c3c;'>⚠ {set_err}</span>"
            return

        # apply_suggestion(pd.concat(dfs, ignore_index=True), f"Uploaded {len(dfs)} file(s) — suggestion uses pooled data.")

    file_input.on_change('value', upload_to_local_drive)

    def on_clear_files_click():
        """Drop every uploaded file and reset the upload UI."""
        _uploaded_files.clear()
        _refresh_upload_list()
        upload_status.text = "Uploads cleared. Add one or more CSVs."

    btn_clear_files.on_click(on_clear_files_click)

    # =========================================================================
    # SECTION 2 — MODEL CONFIGURATION CONTROLS
    # =========================================================================

    library_select = Select(title="LIBRARY",
                            options=["Polynomial", "Fourier", "Combined"],
                            value="Polynomial")

    train_s = Slider(start=10, end=90, value=60, step=5,
                     title="Train/Validation Split")

    def on_train_s_change(attr, old, new):
        train_s.title = f"SPLIT: TRAIN {new}% | VALIDATION {100 - new}%"

    train_s.on_change('value', on_train_s_change)
    on_train_s_change(None, None, train_s.value)
    train_s.show_value = False

    split_select = Select(
        title="SPLIT STRATEGY",
        value="Random Sampling",
        options=[
            "Random Sampling",
            "Time-based",
            "Random Block",
        ],
        width=150,
    )

    poly_s = Slider(start=1, end=5,     value=1,
                    step=1,     title="DEGREE / HARMONICS")
    thr_s = Slider(start=0.0, end=0.5, value=0.1,
                   step=0.005, title="SPARSITY THRESHOLD")
    thr_input = TextInput(
        value=f"{thr_s.value:.4f}", title="Or type exact threshold:", width=150)

    _thr_syncing = [False]

    def on_thr_slider_change(attr, old, new):
        if _thr_syncing[0]:
            return
        _thr_syncing[0] = True
        thr_input.value = f"{new:.4f}"
        _thr_syncing[0] = False

    def on_thr_input_change(attr, old, new):
        if _thr_syncing[0]:
            return
        try:
            val = float(new)
        except ValueError:
            return

        val = max(thr_s.start, min(thr_s.end, val))

        _thr_syncing[0] = True
        thr_s.value = val
        thr_input.value = f"{val:.4f}"
        _thr_syncing[0] = False

    thr_s.on_change('value', on_thr_slider_change)
    thr_input.on_change('value', on_thr_input_change)

    btn_train = Button(label="TRAIN", button_type="primary",
                       height=50, width=100)

    # =========================================================================
    # SECTION 3 — HISTORY TABLE (LEADERBOARD)
    # =========================================================================

    eqn_template = """
    <div style="white-space: normal; word-wrap: break-word; line-height: 1.5;
                padding: 8px 0; font-family: 'Courier New', monospace;
                font-size: 12px; color: #00000;">
        <%= value %>
    </div>
    """
    eqn_formatter = HTMLTemplateFormatter(template=eqn_template)

    metrics_template = """
    <div style="white-space: normal; line-height: 1.4; padding: 4px 0;
                font-family: 'Courier New', monospace; font-size: 11px;">
        <%= value %>
    </div>
    """
    metrics_formatter = HTMLTemplateFormatter(template=metrics_template)

    def _fmt_metrics_html(label, color, r2, rmse, mae):
        return (
            f"<b style='color:{color};'>{label}</b><br>"
            f"R²: {r2:.4f}<br>RMSE: {rmse:.6f}<br>MAE: {mae:.6f}"
        )

    source_history = ColumnDataSource(data=dict(
        run=[], system=[], split=[], lib=[], poly=[], thr=[],
        train_metrics=[], val_metrics=[],
        rmse_diff=[], equations=[]
    ))

    columns = [
        TableColumn(field="run",    title="Run #",      width=100),
        TableColumn(field="system", title="Data File",  width=400),
        TableColumn(field="split",  title="Split Type", width=300),
        TableColumn(field="lib",    title="Library",    width=200),
        TableColumn(field="poly",   title="Degree",     width=200),
        TableColumn(field="thr",    title="Threshold",  width=200),
        TableColumn(field="train_metrics", title="Train Metrics",
                    width=200, formatter=metrics_formatter),
        TableColumn(field="val_metrics",   title="Val Metrics",
                    width=200, formatter=metrics_formatter),
        TableColumn(field="rmse_diff", title="RMSE Diff", width=200),
        TableColumn(field="equations", title="Identified Equations",
                    width=1000, formatter=eqn_formatter),
    ]

    history_table = DataTable(
        source=source_history, columns=columns,
        width=1400, height=400, row_height=200,
        index_position=None, background="#ffffff",
        sortable=True, selectable=True
    )

    btn_delete = Button(label="DELETE",
                        button_type="danger", width=100, height=50)
    btn_delete.disabled = True

    def on_row_select(attr, old, new):
        btn_delete.disabled = not bool(new)
        if not new:
            return
        run_id = source_history.data['run'][new[0]]
        if run_id in trained_model_storage:
            render_plot(run_id)
            diag = trained_model_storage[run_id].get('diagnostics')
            if diag:
                _render_diag_plots(diag)

    source_history.selected.on_change('indices', on_row_select)

    # =========================================================================
    # SECTION 4 — MAIN RESULT PLOT
    # With a multi-IC run, ONE simulation line is drawn per initial
    # condition (IC1 bold solid, extra ICs thinner dashed).
    # =========================================================================

    p = figure(title="Model Result", height=500, sizing_mode="stretch_width")
    p.scatter([], [], alpha=0)

    fit_hover = HoverTool(
        renderers=[],
        mode="vline",
        tooltips=[
            ("Variable", "@name"),
            ("Trajectory", "@trajectory"),
            ("t", "@t{0.000}"),
            ("Value", "@y{0.0000}"),],
    )
    p.add_tools(fit_hover)

    # Renderers are grouped by state and then trajectory so the state, layer,
    # and trajectory controls can filter the same plot independently.
    _main_renderers = {}   # {state_idx: {'trajectories': [{train,val,fit,...}]}}

    state_toggle = CheckboxButtonGroup(
        labels=[], active=[], button_type="default")
    layer_toggle = CheckboxButtonGroup(
        labels=["Data points", "SINDy fit"], active=[0, 1], button_type="default")
    trajectory_toggle = CheckboxButtonGroup(
        labels=[], active=[], button_type="default")
    btn_show_all_trajectories = Button(
        label="SHOW ALL", button_type="default", width=88)
    btn_hide_all_trajectories = Button(
        label="HIDE ALL", button_type="default", width=88)
    trajectory_controls = row(
        trajectory_toggle,
        Spacer(width=8),
        btn_show_all_trajectories,
        btn_hide_all_trajectories,
    )
    trajectory_filter_panel = row(
        Spacer(sizing_mode="stretch_width"),
        trajectory_controls,
        Spacer(sizing_mode="stretch_width"),
        visible=False, sizing_mode="stretch_width",
    )
    # Selection order determines visual focus: the earliest still-active
    # trajectory is solid; later selections are dashed. When the focused
    # trajectory is hidden, the next active one is promoted automatically.
    _trajectory_activation_order = []

    state_key_div = Div(text="", styles={'padding': '2px 0'})

    def _update_main_visibility(attr, old, new):
        active_states = set(state_toggle.active)
        data_on = 0 in set(layer_toggle.active)
        fit_on = 1 in set(layer_toggle.active)
        active_trajectories = set(trajectory_toggle.active)
        focused_trajectory = next(
            (index for index in _trajectory_activation_order
             if index in active_trajectories),
            None,
        )

        for i, rends in _main_renderers.items():
            state_on = i in active_states
            for trajectory_index, trajectory_renders in enumerate(
                    rends['trajectories']):
                trajectory_on = trajectory_index in active_trajectories
                data_visible = state_on and data_on and trajectory_on
                fit_visible = state_on and fit_on and trajectory_on

                r_train = trajectory_renders['train']
                r_val = trajectory_renders['val']
                train_alpha = 0.35 if data_visible else 0
                val_alpha = 0.55 if data_visible else 0
                r_train.glyph.fill_alpha = train_alpha
                r_train.glyph.line_alpha = train_alpha
                r_val.glyph.fill_alpha = val_alpha
                r_val.glyph.line_alpha = val_alpha

                r_fit = trajectory_renders.get('fit')
                if r_fit is not None:
                    is_focused = trajectory_index == focused_trajectory
                    fit_alpha = 1.0 if is_focused else 0.65
                    r_fit.glyph.line_dash = (
                        "solid" if is_focused else "dashed")
                    r_fit.glyph.line_width = 2.8 if is_focused else 1.3
                    r_fit.glyph.line_alpha = fit_alpha if fit_visible else 0
                    r_fit.visible = bool(fit_visible)

    state_toggle.on_change('active', _update_main_visibility)
    layer_toggle.on_change('active', _update_main_visibility)

    def _on_trajectory_visibility_change(attr, old, new):
        active_now = set(new)
        _trajectory_activation_order[:] = [
            index for index in _trajectory_activation_order
            if index in active_now
        ]
        for index in new:
            if index not in _trajectory_activation_order:
                _trajectory_activation_order.append(index)
        _update_main_visibility(attr, old, new)

    trajectory_toggle.on_change(
        'active', _on_trajectory_visibility_change)

    def _show_all_trajectories():
        trajectory_toggle.active = list(range(len(trajectory_toggle.labels)))

    def _hide_all_trajectories():
        trajectory_toggle.active = []

    btn_show_all_trajectories.on_click(_show_all_trajectories)
    btn_hide_all_trajectories.on_click(_hide_all_trajectories)

    # =========================================================================
    # SECTION 5 — RESIDUAL DIAGNOSTIC PLOTS
    # =========================================================================

    p_resid = figure(
        title="Residual vs Time",
        sizing_mode="stretch_width",
        height=280,
        x_axis_label="Time",
        y_axis_label="Residual",
        toolbar_location=None,
    )
    p_resid.scatter([], [], alpha=0)

    p_fft = figure(
        title="Residual FFT (Frequency Content)",
        sizing_mode="stretch_width",
        height=280,
        x_axis_label="Frequency (Hz)",
        y_axis_label="Amplitude",
        toolbar_location=None,
    )
    p_fft.scatter([], [], alpha=0)

    p_scatter = figure(
        title="dX True vs dX Predicted",
        sizing_mode="stretch_width",
        height=280,
        x_axis_label="dX Predicted",
        y_axis_label="dX True",
        toolbar_location=None,
    )
    p_scatter.scatter([], [], alpha=0)

    counter = [0]

    user_warning_div = Div(
        text="",
        styles={'color': '#7f8c8d', 'font-size': '13px', 'padding': '4px 0'}
    )
    _current_view_run = [None]

    def render_plot(run_id):
        """
        Redraw the main result plot from the stored plot_data of a given run.
        Multi-IC runs: one fit line per uploaded initial condition
        (IC1 = bold solid, extra ICs = thinner dashed & fainter).
        """
        data = trained_model_storage[run_id]['plot_data']
        t, X = data['t'], data['X']
        train_idx = data['train_idx']
        val_idx = data['val_idx']
        ic_sims = data.get('ic_sims') or []
        trajectories = data.get('trajectories') or [(X, t)]
        trajectory_labels = data.get('trajectory_labels') or [
            sim.get('label', f"IC {k + 1}")
            for k, sim in enumerate(ic_sims)
        ]
        if len(trajectory_labels) < len(trajectories):
            trajectory_labels.extend(
                f"IC {k + 1}"
                for k in range(len(trajectory_labels), len(trajectories))
            )
        names = trained_model_storage[run_id].get('feature_names') or \
            [f"x{i+1}" for i in range(X.shape[1])]

        p.renderers = []
        _main_renderers.clear()

        n_vars = X.shape[1]
        n_trajectories = len(trajectories)
        color_key_parts = []

        # Map each raw trajectory to its slice in the pooled X/t arrays.
        trajectory_bounds = []
        offset = 0
        for X_k, _ in trajectories:
            end = offset + len(X_k)
            trajectory_bounds.append((offset, end))
            offset = end

        for i in range(n_vars):
            color = _DIAG_COLORS[i % len(_DIAG_COLORS)]
            label = names[i] if i < len(names) else f"x{i+1}"
            trajectory_renderers = []

            for k, (start, end) in enumerate(trajectory_bounds):
                train_k = train_idx[(train_idx >= start) & (train_idx < end)]
                val_k = val_idx[(val_idx >= start) & (val_idx < end)]
                r_train = p.scatter(
                    t[train_k], X[train_k, i], color="#1f77b4",
                    alpha=0.35, size=4, legend_label="Train points")
                r_val = p.scatter(
                    t[val_k], X[val_k, i], color="#ff7f0e",
                    alpha=0.55, size=4, legend_label="Val points")

                r_fit = None
                fit_alpha = 1.0 if k == 0 else 0.65
                if k < len(ic_sims) and ic_sims[k].get('x_sim') is not None:
                    sim_k = ic_sims[k]
                    full_trajectory_name = trajectory_labels[k]
                    fit_source = ColumnDataSource(data=dict(
                        t=sim_k['t'],
                        y=sim_k['x_sim'][:, i],
                        name=[label] * len(sim_k['t']),
                        trajectory=[full_trajectory_name] * len(sim_k['t']),
                    ))
                    r_fit = p.line(
                        't', 'y', source=fit_source, color=color,
                        line_width=2.8 if k == 0 else 1.3,
                        line_dash="solid" if k == 0 else "dashed",
                        alpha=fit_alpha,
                    )

                trajectory_renderers.append({
                    'train': r_train,
                    'val': r_val,
                    'fit': r_fit,
                    'fit_alpha': fit_alpha,
                })

            _main_renderers[i] = {'trajectories': trajectory_renderers}
            color_key_parts.append(
                f"<span style='color:{color}; font-weight:700;'>●</span> "
                f"<span style='color:#2c3e50;'>{label}</span>"
            )
            p.legend.location = "top_right"
            p.legend.click_policy = "hide"

        state_key_div.text = (
            "<div style='font-size:14px;'>" +
            "&nbsp;&nbsp;".join(color_key_parts) + "</div>"
        )

        fit_hover.renderers = [
            trajectory_renders['fit']
            for state_renders in _main_renderers.values()
            for trajectory_renders in state_renders['trajectories']
            if trajectory_renders.get('fit') is not None
        ]

        state_toggle.labels = names[:n_vars] if len(names) >= n_vars else \
            [f"x{i+1}" for i in range(n_vars)]
        state_toggle.active = list(range(n_vars))
        layer_toggle.active = [0, 1]
        trajectory_toggle.labels = [
            _compact_trajectory_label(filename, k)
            for k, filename in enumerate(trajectory_labels)
        ]
        _trajectory_activation_order.clear()
        trajectory_toggle.active = list(range(n_trajectories))
        _trajectory_activation_order[:] = list(range(n_trajectories))
        trajectory_filter_panel.visible = n_trajectories > 1
        _update_main_visibility(None, None, None)

        trajectory_suffix = (
            f" · {n_trajectories} trajectories" if n_trajectories > 1 else "")
        p.title.text = f"Model Result — Run #{run_id}{trajectory_suffix}"
        _current_view_run[0] = run_id

        warning_msg = trained_model_storage[run_id].get('warning')
        if warning_msg:
            user_warning_div.text = f"<b style='color:#d91212;'>⚠ {warning_msg}</b>"
        else:
            user_warning_div.text = "<b style='color:#27ae60;'>✅ Train complete!</b>"

    _DIAG_COLORS = ["#61e0ee", "#ebc626", "#2ca02c", "#d62728", "#9467bd"]

    def _render_diag_plots(diag):
        if not diag:
            return

        p_resid.renderers = []
        p_fft.renderers = []
        p_scatter.renderers = []
        if p_resid.legend:
            p_resid.legend.items = []
        if p_fft.legend:
            p_fft.legend.items = []
        if p_scatter.legend:
            p_scatter.legend.items = []

        var_names = list(diag['residuals'].keys())
        freqs = diag['fft_freqs']
        residual_segments = diag.get('residual_segments') or [{
            'label': 'IC1',
            't': diag['t'],
            'residuals': diag['residuals'],
        }]

        for idx, name in enumerate(var_names):
            color = _DIAG_COLORS[idx % len(_DIAG_COLORS)]

            # Each uploaded trajectory has its own time origin. Render the
            # residuals as independent lines so Bokeh never draws a fake
            # connection from the end of one IC to the start of the next.
            # Reusing legend_label groups all IC renderers for this state.
            for segment_index, segment in enumerate(residual_segments):
                p_resid.line(
                    segment['t'], segment['residuals'][name],
                    color=color,
                    line_width=1.7 if segment_index == 0 else 1.2,
                    line_dash="solid" if segment_index == 0 else "dashed",
                    alpha=0.8 if segment_index == 0 else 0.45,
                    legend_label=name,
                    muted_color=color, muted_alpha=0.08,
                )

            p_fft.line(
                freqs, diag['fft_amps'][name],
                color=color, line_width=1.5, alpha=0.8,
                legend_label=name,
                muted_color=color, muted_alpha=0.12,
            )

            p_scatter.scatter(
                diag['dX_pred'][name], diag['dX_true'][name],
                color=color, alpha=0.3, size=4,
                legend_label=name,
                muted_color=color, muted_alpha=0.06,
            )

        all_amps = np.concatenate([diag['fft_amps'][n] for n in var_names])
        max_amp = float(all_amps.max())

        if max_amp > 0:
            significant_indices = np.where(all_amps > 0.01 * max_amp)[0]

            if len(significant_indices) > 0:
                n_freqs = len(freqs)
                last_idx = int(significant_indices[-1]) % n_freqs
                f_max = float(freqs[last_idx])
                p_fft.x_range.end = f_max * 1.2
                p_fft.x_range.start = 0.0

        all_vals = np.concatenate([diag['dX_true'][n] for n in var_names])
        vmin, vmax = float(all_vals.min()), float(all_vals.max())
        p_scatter.line(
            [vmin, vmax], [vmin, vmax],
            color="#e74c3c", line_width=1.5, line_dash="dashed",
            legend_label="ideal",
        )

        for fig in [p_resid, p_fft, p_scatter]:
            fig.legend.click_policy = "mute"
            fig.legend.location = "top_right"

    # =========================================================================
    # SECTION 6 — TRAIN CALLBACK
    # Loads data (pre-set file, or ALL uploaded CSVs), fits ONE SINDy model
    # pooled across every initial condition, computes diagnostics, updates
    # the leaderboard, and re-renders all plots.
    # =========================================================================

    def on_train_click():
        # ── 1. Resolve the data source (pre-set file vs uploaded CSVs) ─────
        is_custom = (file_select.value == "custom_upload")

        if is_custom:
            if not _uploaded_files:
                user_warning_div.text = "<span style='color:red;'>⚠ Please upload at least one CSV file first!</span>"
                return
            dfs, labels = [], []
            for f in _uploaded_files:
                size_err = check_upload_size(f['b64'])
                if size_err:
                    user_warning_div.text = f"<span style='color:red;'>⚠ {f['name']}: {size_err}</span>"
                    return
                try:
                    df_k = _decode_upload(f)
                except Exception as e:
                    user_warning_div.text = f"<span style='color:red;'>⚠ {f['name']}: {e}</span>"
                    return
                val_err = validate_dataframe(df_k)
                if val_err:
                    user_warning_div.text = f"<span style='color:red;'>⚠ {f['name']}: {val_err}</span>"
                    return
                dfs.append(df_k)
                labels.append(f['name'])

            set_err = validate_trajectory_set(dfs)
            if set_err:
                user_warning_div.text = f"<span style='color:red;'>⚠ {set_err}</span>"
                return

            names = list(dfs[0].columns[1:])
            trajectories = [(df_k.iloc[:, 1:].values, df_k.iloc[:, 0].values)
                            for df_k in dfs]
            short = ", ".join(labels[:3]) + ("…" if len(labels) > 3 else "")
            data_file_label = f"Custom Upload ×{len(dfs)} ({short})"
        else:
            path = os.path.join('data', file_select.value)
            df = pd.read_csv(path).astype(np.float64)
            data_file_label = _SYSTEM_LABELS.get(
                file_select.value, file_select.value)
            names = list(df.columns[1:])
            trajectories = [(df.iloc[:, 1:].values, df.iloc[:, 0].values)]
            labels = [file_select.value]

        counter[0] += 1  # unique, ever-increasing run ID

        # ── 2. Parse data ───────────────────────────────────────────────────
        t = trajectories[0][1]          # primary trajectory time (display)
        X = trajectories[0][0]          # primary trajectory states (display)
        train_frac = train_s.value / 100.0

        # ── 3. Fit SINDy on a train/validation split of the POOLED (X, dX)
        #        pairs from ALL trajectories ────────────────────────────────
        fit_warning_msg = None
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                (model, train_idx, val_idx,
                 m_train, m_val, X_pool, t_pool) = \
                    engine.fit_model(
                        trajectories[0][0], trajectories[0][1],
                        poly_degree=poly_s.value,
                        threshold=thr_s.value,
                        names=names,
                        lib_type=library_select.value,
                        train_frac=train_frac,
                        random_seed=counter[0] * 7,  # unique seed per run
                        split_method=split_select.value.lower(),
                        extra_trajectories=trajectories[1:] or None,
                    )
                if caught:
                    fit_warning_msg = str(caught[-1].message)
        except Exception as e:
            user_warning_div.text = f"<span style='color:red;'>⚠ Fit error: {e}</span>"
            return

        # ── 4. Compute residual diagnostics (aggregated across all ICs) ─────
        diag = engine.compute_diagnostics_multi(trajectories)

        t_r2, t_rmse, t_mae = m_train['r2'], m_train['rmse'], m_train['mae']
        v_r2, v_rmse, v_mae = m_val['r2'],   m_val['rmse'],   m_val['mae']
        rmse_diff = float(np.abs(t_rmse - v_rmse))

        # ── 5. Forward-simulate the discovered equations from EVERY initial
        #        condition over its own time range, for visualization ───────
        ic_sims = []
        for k, (X_k, t_k) in enumerate(trajectories):
            try:
                sim_k = engine.simulate(np.asarray(X_k)[0], np.asarray(t_k))
            except Exception as e:
                sim_k = None
                extra_warn = f"Sim from IC#{k+1} ({labels[k]}) failed: {e}"
                fit_warning_msg = (f"{fit_warning_msg} | {extra_warn}"
                                   if fit_warning_msg else extra_warn)
            ic_sims.append({
                't': np.asarray(t_k),
                'x_sim': sim_k,
                'label': labels[k],
            })

        x_sim_full = ic_sims[0]['x_sim']
        if x_sim_full is None:
            user_warning_div.text = f"<span style='color:red;'>⚠ Simulation error: {fit_warning_msg}</span>"
            return

        # ── 6. Format the discovered equations for display ─────────────────
        raw_eqs = engine.get_equations()
        formatted_eqs_html = "".join(
            [f"<b style='color:#e74c3c;'>({i+1})</b> {eq}<br>" for i,
             eq in enumerate(raw_eqs)]
        )

        # ── 7. Append a new row to the leaderboard ──────────────────────────
        new_entry = {
            'run':        [counter[0]],
            'system':     [data_file_label],
            'split':      [split_select.value],
            'lib':        [library_select.value],
            'poly':       [poly_s.value],
            'thr':        [thr_s.value],
            'train_metrics': [_fmt_metrics_html("TRAIN", "#1f77b4", t_r2, t_rmse, t_mae)],
            'val_metrics':   [_fmt_metrics_html("VAL", "#ff7f0e", v_r2, v_rmse, v_mae)],
            'rmse_diff':  [f"{rmse_diff:.6f}"],
            'equations':  [formatted_eqs_html],
        }
        source_history.stream(new_entry)

        # ── 8. Persist everything needed to reconstruct this run later ─────
        # 'trajectories' stores the RAW, unpooled (X_i, t_i) list — this is
        # what lets the Ensemble tab differentiate each initial condition on
        # its own continuous time axis instead of a seamed pooled array.
        trained_model_storage[counter[0]] = {
            'run_id':             counter[0],
            'system_name':        file_select.value,
            'split_strategy':     split_select.value,
            'model_instance':     copy.deepcopy(engine.model),
            'lib_type':           library_select.value,
            'poly_degree':        poly_s.value,
            'threshold':          thr_s.value,
            'feature_names':      names,
            'initial_conditions': [np.asarray(tr[0])[0].tolist() for tr in trajectories],
            'n_ic':               len(trajectories),
            'metrics': {
                'train_rmse': t_rmse,
                'val_rmse':   v_rmse,
                'rmse_diff':  rmse_diff,
                'val_r2':     v_r2,
            },
            'equations':  raw_eqs,
            'warning':    fit_warning_msg,
            'plot_data': {
                't':            t_pool,       # pooled time (scatter x-axis)
                'X':            X_pool,       # pooled states (scatter y-axis)
                'trajectories': trajectories, # raw list [(X_1,t_1), (X_2,t_2), ...]
                'trajectory_labels': labels,  # full filenames for filters/hover
                'train_idx':    train_idx,    # indices INTO the pooled arrays
                'val_idx':      val_idx,
                'x_sim':        x_sim_full,   # primary-IC simulation
                'ic_sims':      ic_sims,      # one sim per IC (may contain None)
            },
            'diagnostics': diag,
        }

        # ── 9. Select the new history row automatically ───────────────────
        # This triggers on_row_select(), which renders the saved run and
        # enables DELETE immediately. A user who notices a bad parameter
        # choice can therefore remove the run without scrolling to and
        # manually selecting it in the history table first.
        newest_row = len(source_history.data['run']) - 1
        source_history.selected.indices = [newest_row]

    def on_delete_click():
        """
        NOTE: Run IDs (`counter`) are intentionally NOT reset/renumbered
        after a deletion — every run ID stays permanently unique so past
        references (e.g. from the Test/Predict/Ensemble tabs) never become
        ambiguous.
        """
        selected = source_history.selected.indices
        if not selected:
            return

        idx = selected[0]
        run_id = source_history.data['run'][idx]

        if run_id in trained_model_storage:
            del trained_model_storage[run_id]

        new_data = {k: [v for i, v in enumerate(vals) if i != idx]
                    for k, vals in source_history.data.items()}
        source_history.data = new_data
        source_history.selected.indices = []

        if _current_view_run[0] == run_id:
            p.renderers = []
            p.title.text = "Model Result"
            user_warning_div.text = ""
            _current_view_run[0] = None
            _main_renderers.clear()
            state_toggle.labels = []
            state_toggle.active = []
            trajectory_toggle.labels = []
            _trajectory_activation_order.clear()
            trajectory_toggle.active = []
            trajectory_filter_panel.visible = False
            state_key_div.text = ""

        for figs in [p_resid, p_fft, p_scatter]:
            figs.renderers = []
            if figs.legend and len(figs.legend) > 0:
                figs.legend[0].items = []

        p_fft.x_range.start = 0.0
        p_fft.x_range.end = 1.0

    btn_delete.on_click(on_delete_click)

    # ── Run the Suggester once on page load for the default pre-set
    #     system, so the sliders aren't left at arbitrary defaults. ──────
    initial_path = os.path.join('data', file_select.value)
    if os.path.exists(initial_path):
        try:
            df_init = pd.read_csv(initial_path).astype(np.float64)
            val_err = validate_dataframe(df_init)
            if val_err:
                upload_status.text = f"<span style='color:#e74c3c;'>⚠ {val_err}</span>"
            else:
                apply_suggestion(
                    df_init, f"Loaded default pre-set system: {file_select.value}")
        except Exception:
            pass  # non-fatal — user can still configure manually

    btn_train.on_click(on_train_click)

    # =========================================================================
    # SECTION 7 — LAYOUT ASSEMBLY
    # =========================================================================

    top_row = row(
        column(file_select, file_input, upload_list_div, btn_clear_files,
               upload_status, train_s, split_select, library_select,
               poly_s, thr_s, thr_input, row(btn_train, btn_delete), user_warning_div, width=320),
        column(p,
               row(Spacer(sizing_mode="stretch_width"), state_key_div, Spacer(
                   sizing_mode="stretch_width"), sizing_mode="stretch_width"),
               row(Spacer(sizing_mode="stretch_width"), row(state_toggle, layer_toggle), Spacer(
                   sizing_mode="stretch_width"), sizing_mode="stretch_width"),
               trajectory_filter_panel,
               sizing_mode="stretch_width"),
        sizing_mode="stretch_width"
    )

    return column(
        top_row,
        Div(text="<b>RESIDUAL ANALYSIS</b>"),
        row(p_resid, p_fft, p_scatter, sizing_mode="stretch_width"),
        Div(text="<b>TRAINING HISTORY — Metrics on dx/dt</b>"),
        history_table,
        sizing_mode="stretch_width"
    )
