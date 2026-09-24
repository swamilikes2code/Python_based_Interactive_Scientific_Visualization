# =============================================================================
# tabs/ensemble_tab.py
#
# PURPOSE
# -------
# Experimental tab : runs block-bootstrap ensemble
# fitting on an already-trained model from the Train tab, reporting how
# often each candidate term survives thresholding (inclusion frequency)
# and how stable its coefficient is across resamples.
#
# Deliberately synchronous — this is a local-dev experiment, not a
# deployed feature, so no background-thread/executor machinery here.

# Step-by-Step ConceptOriginal Data:
# You have $m$ original data points ($m$ rows in the CSV).
# Generate a Simulated Dataset: Randomly draw $m$ points—allowing duplicates—from the original $m$ points.
# For instance, if the original data is $[1, 2, 3, 4, 5]$, a single draw might yield $[2, 2, 4, 1, 5]$ (the number 2 appears twice, while 3 vanishes).
# It still contains 5 elements, but its composition has changed.
# Fit the Model: Fit SINDy on this simulated dataset to produce a new set of coefficients
# (which may differ slightly from the original set because the input composition changed).
# Repeat: Repeat steps 2 and 3 many times (e.g., 50–100 times) to obtain 50–100 different sets of coefficients.
# Analyze Dispersion: Look at the spread of those 50–100 sets:Low variation:
# If they are all nearly identical, the coefficients are reliable and stable.
# If they fluctuate wildly—or if certain terms appear and disappear—those coefficients/terms are uncertain
# and heavily influenced by random noise in that specific measurement run.
# This directly aligns with what was mentioned earlier: "It's like I manually ran it 50 times and counted."
# Exactly—except that instead of taking 50 actual new measurements (which is impossible with only one file),
# bootstrap simulates 50 "similar" versions of the original dataset via resampling.
#
# Why Resample by BLOCK Instead of Individual Points?
# The data consists of time series where points like $t = 5.00$ and $t = 5.01$ are nearly identical (a smooth trajectory).
# Point-wise resampling: If you resample individual points independently, you risk accidentally picking many near-identical points.
# The simulated dataset might look varied on paper, but the actual information inside changes very little,
# leading you to underestimate true uncertainty.
# Block resampling: Resampling continuous blocks (Block Bootstrap) preserves the smooth temporal structure within each block
# while shuffling/repeating the blocks themselves. This far better reflects the true uncertainty of trajectory data.
# =============================================================================

from bokeh.models import ColumnDataSource, Button, Select, Div, Slider, DataTable, TableColumn
from bokeh.layouts import column, row
from bokeh.plotting import figure


def _ensemble_label(run_id, ensemble_number, n_bootstrap):
    """Build the canonical label used by the ensemble history dropdown."""
    return (f"Run #{run_id} - Ensemble #{ensemble_number} "
            f"- n = {n_bootstrap}")


def ensemble_tab_layout(engine, trained_model_storage):
    """
    Read-only against trained_model_storage: never creates a new run,
    only attaches an 'ensemble' result dict onto an existing run_id.
    """
    # Button select, slider, and data table
    model_select = Select(
        title="SELECT MODEL (FROM HISTORY)", options=[], value="")
    n_bootstrap_s = Slider(start=100, end=100000, value=100, step=100,
                           title="Bootstrap Samples")
    btn_run = Button(label="ENSEMBLE", button_type="primary",
                     width=100, height=50, disabled=True)
    progress_div = Div(text="<i>No trained models available. Train a model first.</i>",
                       styles={'padding': '8px'})
    ensemble_view_run = Select(
        title="VIEWING RUN", options=[], value="", visible=False)

    source_incl = ColumnDataSource(data=dict(
        state=[], term=[], incl_pct=[], coef_mean=[], coef_std=[], n_samples=[]))
    incl_table = DataTable(source=source_incl, columns=[
        TableColumn(field="state",     title="State",        width=80),
        TableColumn(field="term",      title="Term",          width=150),
        TableColumn(field="incl_pct",  title="Inclusion %",   width=100),
        TableColumn(field="coef_mean", title="Coef (mean)",   width=120),
        TableColumn(field="coef_std",  title="Coef (std)",    width=120),
        TableColumn(field="n_samples", title="# Samples",     width=100),
    ], sizing_mode="stretch_width", height=400)

    p_incl = figure(x_range=[], title="Term Inclusion Frequency",
                    sizing_mode="stretch_width", height=300,
                    y_axis_label="% of bootstrap runs", toolbar_location=None)
    # Prevent Bokeh's MISSING_RENDERERS warning before an ensemble is run.
    p_incl.scatter([], [], alpha=0)

    # Rebuild whenever options change — required because the label alone cannot
    # safely infer (run_id) via simple string splitting.

    # Here's the original trained_model_storage structure
    #     trained_model_storage = {
    #     1: {                          # ← key = run_id (int)
    #         'model_instance': ...,    
    #         'plot_data': {...},
    #         'ensemble_runs': [        # ← key = "ensemble_runs" - value = one list
    #             {'n_bootstrap': 50, 'per_state': {...}, 'feature_names': [...]},   # index(i) 0 = "Ensemble #1 (i+1)"
    #             {'n_bootstrap': 70, 'per_state': {...}, 'feature_names': [...]},   # index 1 = "Ensemble #2"
    #         ]
    #     },
    #     2: {
    #         'ensemble_runs': [
    #             {'n_bootstrap': 50, ...}   # Run 2: 1 ensemble
    #         ]
    #     },
    #     3: {
    #         'ensemble_runs': []    # Run 3: 0 ensemble — empty list
    #     }
    #   }
    # We want to make the _ensemble_option_map to look like this:
    #     _ensemble_option_map = {
    #     "Run 1 - Ensemble #1 - n=50": (1, 0),   # (run_id, idx in list)
    #     "Run 1 - Ensemble #2 - n=70": (1, 1),
    #     "Run 2 - Ensemble #1 - n=50": (2, 0),
    # }

    _ensemble_option_map = {}

    def _build_global_ensemble_options():
        """
    Flattens ensemble_runs from ALL trained runs (not just the currently 
    selected run in model_select) into a single list. Each label ALWAYS 
    contains run_id so string duplicates never occur between different runs 
    — this is essential for Bokeh to trigger on_change reliably (see the bug 
    fixed previously: setting .value to its current value causes Bokeh to 
    ignore it and skip calling the callback).
    """
        opts = []
        _ensemble_option_map.clear()
        for run_id in sorted(trained_model_storage.keys()): # sorted the runs: make sure Run 1 first, then Run2,... no matter which 
                                                            # Run gets to ensembled first
            run_data = trained_model_storage[run_id] # take the run's data
            for i, r in enumerate(run_data.get('ensemble_runs', [])):
                # enumerate when loop through a list will return a pair (index,element)
                # for example: run_id = 1 (Run 1), the first inside loop will end at
                # [
                #     {'n_bootstrap': 50, ...},   # i=0, r=this dict
                #     {'n_bootstrap': 70, ...},   # i=1, r=this dict
                # ]
                label = _ensemble_label(run_id, i + 1, r['n_bootstrap']) # Label string for UI - Viewing
                # r['n_bootstrap'] is to get the number of bootstrap conducted - for user to differentiate between each ensembles
                opts.append(label)
                _ensemble_option_map[label] = (run_id, i)
        return opts

    def on_model_select_change(attr, old, new):
        btn_run.disabled = not bool(new)
        progress_div.text = "<i>Ready to run ensemble on this model.</i>" if new else "<i>Select a model to start.</i>"

    model_select.on_change('value', on_model_select_change)
    # def _print_progress(i, total):
    #     # print in terminal when run bootstrap
    #     print(f"[Ensemble] bootstrap {i}/{total}")

    def on_ensemble_view_run_change(attr, old, new):
        """Show 1 saved ensemble of any run - just rerender, no recompute"""
        # 'new' is the newly selected label string from the dropdown (e.g. "Run 1 - Ensemble #2 - n=70")
        # Guard: empty selection, or a stale/unknown label not present in the current map -> do nothing
        if not new or new not in _ensemble_option_map:
            return

        # Reverse-lookup: label -> (run_id, index in that run's ensemble_runs list)
        run_id, idx = _ensemble_option_map[new]

        # Fetch the already-computed ensemble result dict directly from storage.
        # No re-fitting here - this is purely a "replay a saved result" action.
        result = trained_model_storage[run_id]['ensemble_runs'][idx]

        # Give the user visual confirmation of which ensemble is displayed,
        # including fit failures that affect the effective sample count.
        n_requested = result.get('n_bootstrap', 0)
        n_success = result.get('n_successful_bootstrap', n_requested)
        n_failed = result.get('n_failed_bootstrap', n_requested - n_success)
        detail = (f" — {n_success}/{n_requested} fits succeeded"
                  if n_failed else "")
        progress_div.text = (
            f"<b style='color:#27ae60;'>✅ Showing {new}{detail}</b>")

        # Re-render plots/tables using the selected saved result
        _render_results(result)

    # Register the callback: whenever the dropdown's 'value' property changes, call the handler above
    ensemble_view_run.on_change('value', on_ensemble_view_run_change)


    def on_run_click():
        # Guard: no run currently selected in the model_select widget ->
        # nothing to do.
        if not model_select.value:
            return

        # model_select.value is a display string like "Run #1" -> extract
        # the integer run_id.
        run_id = int(model_select.value.replace("Run #", ""))

        # Look up the stored data for this run (model, plot_data,
        # feature_names, ensemble_runs, ...).
        run_data = trained_model_storage.get(run_id)

        # Defensive check: run_id might no longer exist (e.g. deleted from
        # storage between page load and clicking Run) -> abort with warning.
        if run_data is None:
            progress_div.text = "<span style='color:red;'>⚠ Model no longer available.</span>"
            return

        # Disable the "Run" button to prevent duplicate clicks while
        # computation is in progress (fit_ensemble runs synchronously and
        # can take a while — see the module docstring).
        btn_run.disabled = True
        n_boot = n_bootstrap_s.value  # number of bootstrap samples from the slider
        progress_div.text = f"<i>Running {n_boot} bootstrap samples…</i>"

        # Pull out the model's original training configuration (library
        # type, polynomial degree, threshold) and the data it was fit on.
        plot_data = run_data['plot_data']
        names = run_data['feature_names']
        lib_type = run_data['lib_type']
        poly_degree = run_data['poly_degree']
        threshold = run_data['threshold']

        # --- Recover the ORIGINAL (unpooled) trajectory list, if available ---
        # train_tab.py now stores the raw list of (X_i, t_i) pairs — one per
        # uploaded initial condition — under plot_data['trajectories'], in
        # addition to the already-pooled plot_data['X']/plot_data['t'] used
        # for plotting. fit_ensemble needs the UNPOOLED list so it can
        # differentiate each trajectory on its own continuous time axis
        # (exactly like fit_model does) instead of differentiating across
        # a seam where t restarts at 0 for the next file.
        #
        # Backward compatibility: runs trained before this field existed
        # only have the pooled X/t. In that case, fall back to treating the
        # pooled data as a single trajectory — this reproduces the OLD
        # (single-trajectory) behavior exactly, so old runs don't break,
        # they just don't get the multi-IC-aware bootstrap improvement.
        trajectories = plot_data.get('trajectories') or \
            [(plot_data['X'], plot_data['t'])]

        # Split into "primary" trajectory (positional args expected by
        # fit_ensemble) and "extra" trajectories (optional kwarg). This
        # mirrors exactly how train_tab.py calls engine.fit_model().
        X0, t0 = trajectories[0]
        extra = trajectories[1:] or None

        try:
            # Actually compute the ensemble (this is the expensive/slow
            # part — n_bootstrap independent SINDy fits, run synchronously).
            result = engine.fit_ensemble(
                X0, t0, poly_degree, threshold, names,
                lib_type=lib_type, n_bootstrap=n_boot,
                # Deterministic-but-varied seed: unique per run_id AND per
                # ensemble attempt within that run, so re-running ensembles
                # on the same run doesn't silently reuse the same seed.
                random_seed=run_id * 13 + len(run_data.get('ensemble_runs', [])) * 7,
                # Pass along any additional initial-condition trajectories
                # so the block bootstrap respects trajectory boundaries
                # (see _make_blocks_multi) and never differentiates across
                # a file seam.
                extra_trajectories=extra,
            )

            # Persist this new ensemble result into storage. setdefault
            # ensures 'ensemble_runs' key exists (creates [] if missing)
            # before appending, so this works even for runs that started
            # with no ensemble_runs key at all.
            run_data.setdefault('ensemble_runs', []).append(result)

            # Rebuild the ENTIRE option list (across all runs, not just the
            # current run_id) — the dropdown is global, showing ensembles
            # from every trained run so the user can compare across runs.
            opts = _build_global_ensemble_options()
            ensemble_view_run.options = opts
            ensemble_view_run.visible = True  # enable dropdown now that at least one option exists

            # Build the label for the ensemble we just computed, so we can
            # select it in the dropdown. len(run_data['ensemble_runs']) is
            # used (not n_boot) because it reflects the 1-indexed position
            # of this new result within THIS run's list, matching how
            # _build_global_ensemble_options() generates labels
            # ("Ensemble #{i+1}").
            new_label = _ensemble_label(
                run_id, len(run_data['ensemble_runs']), n_boot)

            # Setting .value to this new label triggers
            # on_ensemble_view_run_change via on_change, because this label
            # is guaranteed unique/new (never existed before) — Bokeh WILL
            # fire the callback. (Contrast with the earlier bug: setting
            # .value to something IDENTICAL to the current value causes
            # Bokeh to skip on_change entirely.)
            ensemble_view_run.value = new_label

            # Defensive/explicit call: even though on_change is expected to
            # fire and call _render_results itself, call it directly here
            # too, guaranteeing the UI updates regardless of any Bokeh
            # timing/event-order edge case.
            _render_results(result)

            n_success = result.get('n_successful_bootstrap', n_boot)
            n_failed = result.get('n_failed_bootstrap', n_boot - n_success)
            if n_failed:
                progress_div.text = (
                    "<b style='color:#d97706;'>⚠ Ensemble complete: "
                    f"{n_success}/{n_boot} fits succeeded; {n_failed} failed. "
                    "Inclusion frequencies use successful fits only.</b>"
                )
            else:
                progress_div.text = (
                    f"<b style='color:#27ae60;'>✅ Ensemble complete: "
                    f"{n_success}/{n_boot} fits succeeded.</b>"
                )
        except Exception as e:
            # Catch-all: any failure during fit_ensemble (bad data,
            # numerical error, etc.) is shown to the user instead of
            # crashing the callback silently.
            progress_div.text = f"<span style='color:red;'>⚠ Ensemble error: {e}</span>"
        finally:
            # Always re-enable the button, whether the run succeeded or failed.
            btn_run.disabled = False
            
    def _render_results(result):
        rows = dict(state=[], term=[], incl_pct=[],
                    coef_mean=[], coef_std=[], n_samples=[])
        bar_labels, bar_vals = [], []
        for state_name, stats in result['per_state'].items():
            for term in result['feature_names']:
                pct = stats['inclusion_pct'][term]
                rows['state'].append(state_name)
                rows['term'].append(term)
                rows['incl_pct'].append(f"{pct*100:.1f}%")
                mean = stats['coef_mean'][term]
                std = stats['coef_std'][term]
                rows['coef_mean'].append(
                    f"{mean:.4f}" if mean is not None else "—")
                rows['coef_std'].append(
                    f"{std:.4f}" if std is not None else "—")
                rows['n_samples'].append(stats['n_samples'][term])
                bar_labels.append(f"{state_name}: {term}")
                bar_vals.append(pct * 100)
        source_incl.data = rows

        p_incl.renderers = []
        p_incl.x_range.factors = bar_labels
        p_incl.vbar(x=bar_labels, top=bar_vals, width=0.7, color="#3498db")
        p_incl.xaxis.major_label_orientation = 1.0

    btn_run.on_click(on_run_click)

    def update_model_list():
        """Get the runs from train_tab through trained_model_storage (shared for all tabs)"""
        opts = [f"Run #{i}" for i in sorted(trained_model_storage.keys())]
        model_select.options = opts

        if not opts:
            progress_div.text = "<i>No trained models available. Train a model first.</i>"
        elif not model_select.value:
            progress_div.text = "<i>Select a model to start.</i>"
        # ← ADDED: also refresh the global ensemble list, preserving the current
        # selection if it is still valid (prevents unnecessarily resetting to empty
        # every time you switch tabs back and forth)
        ens_opts = _build_global_ensemble_options()
        current = ensemble_view_run.value
        ensemble_view_run.options = ens_opts
        if current in ens_opts:
            ensemble_view_run.value = current
        elif ens_opts:
            ensemble_view_run.value = ens_opts[-1]
            _render_results(trained_model_storage[_ensemble_option_map[ens_opts[-1]][0]]
                            ['ensemble_runs'][_ensemble_option_map[ens_opts[-1]][1]])
        else:
            ensemble_view_run.visible = False

    layout = column(
        row(model_select, n_bootstrap_s, btn_run),
        progress_div,
        ensemble_view_run,
        p_incl,
        incl_table,
        sizing_mode="stretch_width"
    )
    return layout, update_model_list
