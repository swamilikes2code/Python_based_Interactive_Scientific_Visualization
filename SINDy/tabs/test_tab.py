# =============================================================================
# tabs/test_tab.py
#
# PURPOSE
# -------
# Renders the "Test" tab of the SINDy Expert System.
#
# This tab answers the question the Train tab CANNOT answer on its own:
# "Do the discovered equations actually generalize to a trajectory the
# model has never seen?" SINDy's loss is computed on dX/dt residuals, so
# a model can look excellent during training yet still diverge badly when
# forward-simulated from a new initial condition — this tab is the
# ground-truth check for that failure mode.
#
# Workflow:
#   1. User picks a previously-trained model (from trained_model_storage,
#      populated by the Train tab).
#   2. The tab auto-suggests matching held-out test CSVs for pre-set
#      systems, or lets the user upload their own test trajectory if the
#      model was trained on custom data.
#   3. For each test trajectory, the discovered ODE system is forward-
#      integrated with scipy's solve_ivp() starting from the test file's
#      initial condition, and the result is compared against the true
#      trajectory (RMSE / R² per state variable + overlay plot).
# =============================================================================

from bokeh.models import ColumnDataSource, Button, Select, Div, DataTable, TableColumn, FileInput, HoverTool
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.palettes import Category10
import numpy as np
import pandas as pd
import os
import base64
import io
from scipy.integrate import solve_ivp
from engine.check_datafile import check_upload_size, validate_dataframe


def test_tab_layout(engine, trained_model_storage):
    """
    Build the Bokeh layout for the Test tab.

    Parameters
    ----------
    engine : SINDyEngine
        Shared engine instance. Not called directly in this tab — test
        simulation calls model_instance.predict() straight from the stored
        model — but kept in the signature for API consistency with the
        other tabs.
    trained_model_storage : dict
        Shared store populated by the Train tab: {run_id: {model_instance,
        system_name, feature_names, ...}}. Read-only from this tab.

    Returns
    -------
    tuple(bokeh.layouts.column, callable)
        (layout, update_model_list) — update_model_list() is called by
        main.py whenever the user switches to this tab, so the model
        dropdown always reflects the latest training history.
    """

    # =========================================================================
    # SECTION 1 — MODEL SELECTION & DATA SOURCE UI
    # Dropdown to pick a trained model, plus the (pre-set or custom) test
    # data source for up to 2 independent test trajectories.
    # =========================================================================

    csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]

    model_select = Select(
        title="SELECT MODEL (FROM HISTORY)", options=[], value="")
    file_select_1 = Select(title="Test Run 1 — CSV File",
                           options=csv_files, value="")
    file_select_2 = Select(title="Test Run 2 — CSV File (optional)",
                           options=["(none)"] + csv_files, value="(none)")

    # Upload widgets — only shown when the selected model was trained on a
    # custom-uploaded dataset (no matching pre-set test files exist for it).
    file_input_test1 = FileInput(
        accept=".csv", title="Upload Test Data 1", visible=False)
    file_input_test2 = FileInput(
        accept=".csv", title="Upload Test Data 2", visible=False)

    # ── In-memory upload buffer ──────────────────────────────────────────
    # Bokeh's FileInput.value only fires a change event; it does NOT persist
    # the file content anywhere else. We cache the raw base64 string here so
    # on_test_click() can re-decode it later without depending on widget
    # state timing (avoids a race where the widget value changes between
    # upload and button click).
    _upload_buffer = {
        'test1': None,
        'test2': None,
    }

    # Single unified status line — replaces the earlier per-column
    # status_div1 / status_div2 pair, which felt visually redundant since
    # both usually showed the same "select a model" / "test complete"
    # message at the same time. All feedback (errors for Run 1, Run 2, or
    # the final success message) now goes through this one Div.
    status_div = Div(
        text="<i>Select a model to start.</i>",
        styles={'padding': '8px'}
    )

    def get_test_filenames_list(model_run_id):
        """
        Map a trained run's source system to its matching pair of held-out
        test CSVs (generated with different initial conditions than the
        training file — see project data-generation scripts). Returns
        ["", "(none)"] if the run isn't found or has no known test pair.
        """
        if model_run_id not in trained_model_storage:
            return ["", "(none)"]
        train_file = trained_model_storage[model_run_id].get('system_name', '')
        if train_file == "cs_train_data.csv":
            return ["cs_test_data_1.csv", "cs_test_data_2.csv"]
        if train_file == "vanderpol_train.csv":
            return ["vanderpol_test_1.csv", "vanderpol_test_2.csv"]
        if train_file == "pendulum_train.csv":
            return ["pendulum_test_1.csv", "pendulum_test_2.csv"]
        if train_file == "timedep_train.csv":
            return ["timedep_test_1.csv", "timedep_test_2.csv"]
        return ["", "(none)"]

    def update_ui_on_model_select(attr, old, new):
        """
        Fired when the user picks a model from the dropdown. Decides
        whether to show the pre-set test-file dropdowns or the custom
        upload widgets, and auto-fills the pre-set test filenames when
        applicable. Also resets the upload buffer so a stale file from a
        previously-selected custom model can't accidentally be reused.
        """
        if not new:
            return
        try:
            run_id = int(new.replace("Run #", ""))
            train_file = trained_model_storage[run_id].get('system_name', '')
            is_custom = "custom" in train_file or train_file == "custom_upload"

            # Toggle visibility: pre-set dropdowns XOR custom upload widgets.
            file_select_1.visible = not is_custom
            file_select_2.visible = not is_custom
            file_input_test1.visible = is_custom
            file_input_test2.visible = is_custom

            # FileInput.value is read-only from the server side, so we
            # cannot force-clear it to guarantee the next upload fires a
            # fresh change event. More importantly: clearing the buffer on
            # every model switch was the wrong design to begin with — a
            # user commonly reuses the SAME test file across multiple
            # training runs (Run 1, Run 2, ...). We now KEEP whatever is
            # cached in _upload_buffer across model switches instead of
            # wiping it. Any genuine incompatibility (e.g. this model
            # expects a different number of state variables) is still
            # caught safely later by _run_single_test()'s variable-count
            # check, so reusing a stale buffer is never unsafe — at worst
            # it produces a clear error message instead of a crash.
            if is_custom:
                if _upload_buffer["test1"]:
                    status_div.text = "<b style='color:#27ae60;'>✅ Reusing previously uploaded Test file 1. Click TEST.</b>"
                    btn_test.disabled = False
                else:
                    status_div.text = "<i>Please upload a test file.</i>"
                    btn_test.disabled = True
            else:
                targets = get_test_filenames_list(run_id)
                file_select_1.value, file_select_2.value = targets[0], targets[1]
                # Preset model: enable only if a valid Test Run 1 file was found.
                # get_test_filenames_list returns "" for unmatched system_name,
                # which must NOT enable the button.
                btn_test.disabled = not bool(targets[0])
        except Exception as e:
            # Non-fatal — log to server console, leave UI in its current state
            # rather than crashing the callback.
            print(f"UI Update Error: {e}")

    model_select.on_change('value', update_ui_on_model_select)

    btn_test = Button(label="TEST", button_type="primary",
                      height=50, width=100, disabled=True)

    # =========================================================================
    # SECTION 2 — FILE UPLOAD HANDLING
    # Caches uploaded test CSVs (as base64) so they can be decoded later
    # inside on_test_click(), independent of widget event timing.
    # =========================================================================

    def on_upload_test1(attr, old, new):
        """Cache Test Run 1's uploaded file content (base64) on change."""
        if not new:
            return
        _upload_buffer['test1'] = new
        status_div.text = "<b style='color:#27ae60;'>✅ Test file 1 ready. Click TEST.</b>"
        btn_test.disabled = False

    def on_upload_test2(attr, old, new):
        """Cache Test Run 2's uploaded file content (base64) on change."""
        if not new:
            return
        _upload_buffer['test2'] = new
        status_div.text = "<b style='color:#27ae60;'>✅ Test file 2 ready. Click TEST.</b>"
        # btn_test.disabled = False

    file_input_test1.on_change('value', on_upload_test1)
    file_input_test2.on_change('value', on_upload_test2)

    # =========================================================================
    # SECTION 3 — RESULTS DISPLAY (Metrics Table & Plots)
    # Per-variable RMSE/R² table plus one overlay plot (true vs SINDy) for
    # each of the two possible test trajectories.
    # =========================================================================

    source_metrics = ColumnDataSource(data=dict(
        run=[], model_run=[], variable=[], rmse=[], mae=[], r2=[]
    ))
    metrics_table = DataTable(source=source_metrics, columns=[
        TableColumn(field="run",       title="Test Run", width=100),
        TableColumn(field="model_run", title="Model",    width=100),
        TableColumn(field="variable",  title="Variable",      width=100),
        TableColumn(field="rmse",      title="RMSE on x(t)",     width=120),
        TableColumn(field="r2",        title="R²",       width=120),
    ], sizing_mode="stretch_width", height=200)

    # p2 starts invisible — only shown once Test Run 2 actually produces
    # results, so an unused second plot doesn't clutter the layout.
    p1 = figure(title="Test Run 1", width=900, height=350,
                x_axis_label="Time (s)", sizing_mode="stretch_width")
    p2 = figure(title="Test Run 2", width=900, height=350,
                x_axis_label="Time (s)", visible=False, sizing_mode="stretch_width")

    # Hover for each plot, view on SINDy line mode='vline'
    # view 3 stats at once: sindy, true, and diff = true - sindy for all the states
    _test_tooltips = [
        ("Variable", "@name"),
        ("t", "@t{0.000}"),
        ("True", "@true{0.0000}"),
        ("SINDy", "@pred{0.0000}"),
        ("Diff", "@diff{0.0000}"),
    ]
    hover_p1 = HoverTool(renderers=[], mode='vline', tooltips=_test_tooltips)
    hover_p2 = HoverTool(renderers=[], mode='vline', tooltips=_test_tooltips)
    p1.add_tools(hover_p1)
    p2.add_tools(hover_p2)
    # {fig.id: hover_tool} — để _run_single_test tra ra đúng hover tool
    # ứng với plot (p1 hoặc p2) mà nó đang vẽ, mà không cần đổi chữ ký hàm.
    _hover_by_fig = {p1: hover_p1, p2: hover_p2}

    # ── ROBUSTNESS FIX ──────────────────────────────────────────────────
    # Category10[10] only has 10 distinct colors. The original indexing
    # scheme (colors[i*2], colors[i*2+1]) silently assumed at most 5 state
    # variables — a 6th variable (i=5) would index colors[10], which is out
    # of range and raises IndexError, crashing the whole test callback.
    # We fix this by wrapping every index with `% len(colors)` so the
    # palette cycles instead of crashing on systems with more variables
    # (e.g. a 3-mass coupled system has 6 states: x1,v1,x2,v2,x3,v3).
    _TEST_COLORS = Category10[10]

    # =========================================================================
    # SECTION 4 — DATA LOADING HELPERS
    # Two symmetric loaders: one for pre-set files on disk, one for
    # user-uploaded base64 payloads. Both return (DataFrame, error_message)
    # and both validate the data for NaN/Inf before handing it back, so a
    # malformed CSV fails fast with a clear message instead of silently
    # propagating into solve_ivp and producing a cryptic numerical error.
    # =========================================================================

    def _load_df_from_select(sel_value):
        """Load a DataFrame from a pre-set CSV file living in data/."""
        path = os.path.join('data', sel_value)
        if not os.path.exists(path):
            return None, f"File {sel_value} missing"
        try:
            df = pd.read_csv(path).astype(np.float64)
        except Exception as e:
            return None, f"Could not parse CSV: {e}"
        err = validate_dataframe(df)
        if err:
            return None, err
        return df, None

    def _load_df_from_buffer(b64_data):
        """Decode a cached base64 upload payload into a DataFrame."""
        err = check_upload_size(b64_data)
        if err:
            return None, err
        try:
            decoded = base64.b64decode(b64_data)
            df = pd.read_csv(io.BytesIO(decoded)).astype(np.float64)
        except Exception as e:
            return None, f"Could not parse uploaded CSV: {e}"
        err = validate_dataframe(df)
        if err:
            return None, err
        return df, None

    # =========================================================================
    # SECTION 5 — CORE TEST RUNNER
    # Forward-simulates the discovered SINDy equations from the test file's
    # initial condition using scipy's solve_ivp, then computes per-variable
    # error metrics against the true (measured) trajectory.
    # =========================================================================

    def _run_single_test(fig, df, label):
        """
        Run one test trajectory through the currently-selected model and
        plot the result on `fig`.

        Key idea: this is a genuine out-of-sample check — the equations
        were fit on TRAINING data, but here they are numerically integrated
        (not just evaluated pointwise) starting only from the test file's
        x(0), so any error in the discovered coefficients compounds over
        time exactly the way it would in a real deployment scenario.

        Parameters
        ----------
        fig : bokeh.plotting.figure
            Target plot (p1 or p2) to draw the true-vs-predicted overlay on.
        df : pandas.DataFrame
            Test trajectory — first column is time, remaining columns are
            state variables in the same order used during training.
        label : str
            Human-readable label for this run (currently only used for
            potential future logging — not rendered directly).

        Returns
        -------
        tuple(list[dict] | None, str | None)
            (rows, error_message) — rows is a list of per-variable metric
            dicts on success; error_message is set (rows=None) on failure
            (e.g. dimension mismatch, solve_ivp not converging, or an
            exception raised inside the model's predict() call).
        """
        t = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values
        n_test_vars = X.shape[1]

        # ── ROBUSTNESS FIX ──────────────────────────────────────────────
        # If the test CSV has a different number of state variables than
        # the model was trained on, model_instance.predict() will raise a
        # low-level sklearn/numpy shape-mismatch error deep inside
        # solve_ivp's internal loop, which is confusing to a non-technical
        # user. We check this up front and fail with a clear message.
        n_model_vars = getattr(model_instance, "n_features_in_", n_test_vars)
        if n_test_vars != n_model_vars:
            return None, (f"Variable count mismatch: test file has "
                          f"{n_test_vars} state variable(s), but the "
                          f"selected model was trained on {n_model_vars}.")

        def rhs(t_val, x):
            # model_instance.predict expects a 2D array of shape (1, n_features);
            # solve_ivp passes/expects flat 1D state vectors, hence the reshape/flatten.
            return model_instance.predict(np.array(x).reshape(1, -1)).flatten()

        # ── ROBUSTNESS FIX ──────────────────────────────────────────────
        # solve_ivp does not catch exceptions raised inside the RHS
        # function — if predict() throws (e.g. due to a still-mismatched
        # shape, or a NaN produced during integration), the exception
        # propagates uncaught and crashes the entire Bokeh callback,
        # taking down the whole Test tab. We wrap the call so any failure
        # becomes a normal, user-visible error message instead.
        try:
            sol = solve_ivp(rhs, (t[0], t[-1]), X[0, :],
                            t_eval=t, method='RK45')
        except Exception as e:
            return None, f"Simulation failed: {e}"

        if not sol.success:
            return None, sol.message

        # Clear any previous overlay before drawing the new one.
        fig.renderers = []
        if fig.legend:
            fig.legend.items = []

        rows = []
        pred_renderers = []  # to set to hover_p1/hover_p2 after the loop
        for i, vname in enumerate(df.columns[1:]):
            # ── ROBUSTNESS FIX ──
            # Wrap indices with modulo so the palette cycles instead of
            # raising IndexError for systems with more than 5 variables.
            c_true = _TEST_COLORS[(i * 2) % len(_TEST_COLORS)]
            c_pred = _TEST_COLORS[(i * 2 + 1) % len(_TEST_COLORS)]

            diff = X[:, i] - sol.y[i]  # diff = true - sindy
            # Source enough field for HoverTool: t, true, pred, diff, name.
            var_source = ColumnDataSource(data=dict(
                t=t, true=X[:, i], pred=sol.y[i], diff=diff,
                name=[vname] * len(t)
            ))

            # True (measured) trajectory as scattered points.
            fig.scatter('t', 'true', source=var_source, color=c_true,
                        alpha=0.3, legend_label=f"{vname} (True)")
            # SINDy-simulated trajectory as a solid line — this is the
            # renderer the hover tool attaches to.
            r_pred = fig.line('t', 'pred', source=var_source, color=c_pred,
                              line_width=2, legend_label=f"{vname} (SINDy)")
            pred_renderers.append(r_pred)

            # X[:,i] = x(t) from test data, sol.y[i] = x(t) from forward-integrate
            res = X[:, i] - sol.y[i]
            ss_tot = np.sum((X[:, i] - np.mean(X[:, i])) ** 2)
            # Guard against a degenerate constant true-trajectory (ss_tot=0),
            # which would otherwise produce a division-by-zero R² of inf/NaN.
            r2 = 1 - (np.sum(res ** 2) /
                      ss_tot) if ss_tot > 0 else float('nan')
            rows.append({
                'variable': vname,
                'rmse':     f"{np.sqrt(np.mean(res ** 2)):.6f}",
                'r2':       f"{r2:.4f}",
            })

        fig.legend.click_policy = "hide"
        fig.legend.location = "top_right"

        # update hover tool of this fig to point to the corresponding SINDy-line
        # avoid pointing to the wrong sindy line.
        hover_tool = _hover_by_fig.get(fig)
        if hover_tool is not None:
            hover_tool.renderers = pred_renderers

        return rows, None

    # =========================================================================
    # SECTION 6 — RUN TEST CALLBACK
    # Orchestrates loading both test trajectories (pre-set or uploaded),
    # running _run_single_test on each, and aggregating results into the
    # shared metrics table.
    # =========================================================================

    # Placeholder — populated inside on_test_click() and read via `nonlocal`
    # from _run_single_test's closure (rhs()) during the solve_ivp call.
    model_instance = None

    def on_test_click():
        if not model_select.value:
            status_div.text = "<b style='color:red;'>⚠️ Please select a model first.</b>"
            return

        run_id = int(model_select.value.replace("Run #", ""))
        model_data = trained_model_storage.get(run_id)

        # ── ROBUSTNESS FIX ──────────────────────────────────────────────
        # Defensive check in case the selected run was deleted from the
        # leaderboard (Train tab) between selecting it here and pressing
        # TEST — avoids a raw KeyError/AttributeError crash.
        if model_data is None or model_data.get('model_instance') is None:
            status_div.text = "<b style='color:red;'>⚠️ Selected model is no longer available. Please pick another.</b>"
            return

        # Bind the model instance for this test run so _run_single_test's
        # inner rhs() closure can access it.
        nonlocal model_instance
        model_instance = model_data['model_instance']

        run_id_str = f"#{run_id}"
        res_data = {'run': [], 'model_run': [],
                    'variable': [], 'rmse': [], 'r2': []}
        messages = []  # collects per-run error/success text for status_div

        # ── Test Run 1 (required) ───────────────────────────────────────
        is_custom = file_input_test1.visible
        if is_custom:
            if not _upload_buffer['test1']:
                status_div.text = "<b style='color:red;'>⚠️ Please upload Test file 1 first.</b>"
                return
            df, err = _load_df_from_buffer(_upload_buffer['test1'])
        else:
            df, err = _load_df_from_select(file_select_1.value)

        if err:
            messages.append(f"⚠️ Run 1 error: {err}")
        elif df is not None:
            rows, err = _run_single_test(p1, df, "Run 1")
            if err:
                messages.append(f"⚠️ Run 1 error: {err}")
            elif rows:
                p1.visible = True
                for r in rows:
                    res_data['variable'].append(r['variable'])
                    res_data['rmse'].append(r['rmse'])
                    res_data['r2'].append(r['r2'])
                    res_data['run'].append("Run 1")
                    res_data['model_run'].append(run_id_str)

        # ── Test Run 2 (optional) ───────────────────────────────────────
        if is_custom:
            if _upload_buffer['test2']:
                df2, err2 = _load_df_from_buffer(_upload_buffer['test2'])
            else:
                df2, err2 = None, None  # user chose not to provide a 2nd test file
                p2.visible = False
        else:
            df2, err2 = _load_df_from_select(file_select_2.value) \
                if file_select_2.value != "(none)" else (None, None)

        if err2:
            messages.append(f"⚠️ Run 2 error: {err2}")
        elif df2 is not None:
            rows2, err2 = _run_single_test(p2, df2, "Run 2")
            if err2:
                messages.append(f"⚠️ Run 2 error: {err2}")
            elif rows2:
                p2.visible = True
                for r in rows2:
                    res_data['variable'].append(r['variable'])
                    res_data['rmse'].append(r['rmse'])
                    res_data['r2'].append(r['r2'])
                    res_data['run'].append("Run 2")
                    res_data['model_run'].append(run_id_str)

        source_metrics.data = res_data

        # ── Build the final status message ───────────────────────────────
        if messages:
            status_div.text = "<b style='color:red;'>" + \
                "<br>".join(messages) + "</b>"
        else:
            status_div.text = "<b style='color:#27ae60;'>✅ Test complete!</b>"

    btn_test.on_click(on_test_click)

    # =========================================================================
    # SECTION 7 — EXTERNAL HOOK (called by main.py on tab switch)
    # Keeps the model dropdown and pre-set file dropdowns fresh whenever
    # the user navigates to this tab, since new runs/files may have been
    # added since the tab was last visited.
    # =========================================================================

    def update_model_list():
        """
        Refresh model_select options from trained_model_storage, and
        refresh the pre-set CSV dropdowns from the data/ directory.
        Called externally (from main.py) on every switch to this tab —
        NOT wired to a Bokeh event, since there is no "tab became active"
        signal for a plain layout composition.
        """
        opts = [f"Run #{i}" for i in sorted(trained_model_storage.keys())]
        model_select.options = opts
        
        # The old check `if opts and not model_select.value` only catches
        # the empty-value case. It misses the common case where the
        # currently-selected run was deleted from trained_model_storage
        # (e.g. removed from the Train tab leaderboard) — model_select.value
        # still holds a stale string like "Run #1" that no longer exists in
        # `opts`. Since Bokeh only fires on_change when the value actually
        # changes, that stale value never triggers update_ui_on_model_select,
        # so file input visibility / TEST button state stay frozen on
        # whatever the deleted run last left behind — which, combined with
        # the persistent upload buffer, can leave TEST looking enabled while
        # actually pointing at a run_id that no longer exists.
        if opts:
            if model_select.value not in opts:
                model_select.value = opts[-1]
            status_div.text = "<i>Ready to test this model.</i>"
        else:
            # No runs left at all.
            model_select.value = ""
            file_input_test1.visible = False
            file_input_test2.visible = False
            file_select_1.visible = True
            file_select_2.visible = True
            btn_test.disabled = True
            status_div.text = "<i>No trained models available. Train a model first.</i>"

        f_list = [f for f in os.listdir('data') if f.endswith('.csv')]
        file_select_1.options = f_list
        file_select_2.options = ["(none)"] + f_list

    # =========================================================================
    # SECTION 8 — LAYOUT ASSEMBLY
    # =========================================================================

    layout = column(
        row(
            model_select,
            column(file_select_1, file_input_test1),
            column(file_select_2, file_input_test2),
            btn_test,
        ),
        status_div,
        metrics_table, p1, p2,
        sizing_mode="stretch_width"
    )
    return layout, update_model_list
