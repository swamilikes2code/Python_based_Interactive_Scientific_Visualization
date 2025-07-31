import numpy as np
import threading
import warnings
from functools import partial
import csv
import io
import base64

from skopt import Optimizer
from skopt.space import Real
from skopt.learning import (
    GaussianProcessRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)

from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    Select,
    Div,
    Spinner,
    TextInput,
    RadioButtonGroup,
    DataTable,
    TableColumn,
    NumberFormatter,
    ColumnDataSource,
    FileInput,
    CustomJS,
)

# --- Suppress scikit-optimize warnings ---
warnings.filterwarnings("ignore", category=UserWarning)

# --- Global State ---
optimizer = None
param_names = []
dimensions = []
maximize_objective = True
experiment_history = []
suggested_x = None
TOLERANCE = 1e-4    # Stopping threshold

# --- Bokeh Application Setup ---
doc = curdoc()
doc.title = "Interactive Optimizer"

# --- Constants ---
MAX_PARAMS = 20
MAX_INITIAL_POINTS = 200

# --- Data Sources ---
experiments_source = ColumnDataSource(data=dict(Iteration=[], Objective=[]))
convergence_source = ColumnDataSource(data=dict(iter=[], best_value=[]))

# --- UI and Workflow Functions ---
def set_ui_state(phase='setup', lock_all=False):
    """Manages which UI elements are enabled or disabled."""
    if lock_all:
        for widget in all_buttons + setup_widgets + optimization_widgets:
            widget.disabled = True
        return

    is_setup = (phase == 'setup')
    for widget in setup_widgets:
        widget.disabled = not is_setup
    for widget in optimization_widgets:
        widget.disabled = is_setup

    reset_button.disabled = is_setup
    submit_result_button.disabled = True
    csv_file_input.disabled = not is_setup


def update_direction_indicator():
    """Updates the direction indicator text based on current setting"""
    direction = "Maximizing" if maximize_objective else "Minimizing"
    direction_indicator.text = f"<i>({direction})</i>"
    direction_indicator.styles = {
        'color': 'green' if maximize_objective else 'blue',
        'font-weight': 'bold'
    }

def on_num_params_change(attr, old, new):
    """Shows/hides parameter definition rows and updates initial data rows."""
    num_params = int(new)
    for i in range(MAX_PARAMS):
        param_rows[i].visible = (i < num_params)
        initial_data_headers[i].visible = (i < num_params)
        if i < num_params:
            on_param_range_change(i, None, None, None)
    
    # Update initial data UI
    num_initial_points = initial_data_spinner.value or 0
    on_initial_data_change(None, None, num_initial_points)
    update_initial_random_points_info()

def on_initial_data_change(attr, old, new):
    """Shows/hides rows for entering initial data."""
    num_to_show = new or 0
    initial_data_header_row.visible = (num_to_show > 0)
    active_param_indices = [i for i in range(num_params_spinner.value)]
    
    for i in range(MAX_INITIAL_POINTS):
        row_visible = i < num_to_show
        initial_data_rows[i].visible = row_visible
        if row_visible:
            for j in range(MAX_PARAMS):
                initial_param_inputs[i][j].visible = (j in active_param_indices)
            
    update_initial_points_warning(new)
    update_initial_random_points_info()

def on_objective_name_change(attr, old, new):
    """Updates the initial data header for the objective column in real-time."""
    objective_header.text = f"<b>{new or 'Objective'}</b>"
    update_direction_indicator()
    # When objective name changes, it might affect the column header in the DataTable,
    # so we need to trigger a DataTable columns update.
    # This will be handled by re-initializing the optimizer when setup is locked in.

def on_param_name_change(index, attr, old, new):
    """Updates a specific parameter header in the initial data section in real-time."""
    initial_data_headers[index].text = f"<b>{new or f'Param {index + 1}'}</b>"

def on_surrogate_model_change(attr, old, new):
    """Updates acquisition function options based on the selected model."""
    if new == "GP":
        acq_func_select.options = ["gp_hedge", "EI", "PI", "LCB"]
    else:
        acq_func_select.options = ["EI", "PI", "LCB"]
        if acq_func_select.value == "gp_hedge":
            acq_func_select.value = "EI"

def on_param_range_change(index, attr, old, new):
    """Updates the min/max of the initial data spinners when a parameter's range is changed."""
    low_val = param_low_spinners[index].value
    high_val = param_high_spinners[index].value

    if low_val is None or high_val is None:
        return

    for k in range(MAX_INITIAL_POINTS):
        spinner_to_update = initial_param_inputs[k][index]
        spinner_to_update.low = low_val
        spinner_to_update.high = high_val
        if spinner_to_update.value is not None:
            spinner_to_update.value = max(low_val, min(high_val, spinner_to_update.value))

def update_initial_points_warning(num_points=0):
    """Shows a warning if the number of points is between 1 and 4."""
    num_points = num_points or 0
    initial_points_warning_div.visible = 0 < num_points < 5

def update_initial_random_points_info():
    """Updates the info div about the number of initial random points."""
    num_params = num_params_spinner.value
    num_initial_data = initial_data_spinner.value if initial_data_spinner.value is not None else 0 
    
    # Calculate the effective n_initial_points that skopt will use
    effective_initial_random_points = max(5, 2 * num_params, num_initial_data)
    
    info_text = f"""
    <p style="font-size: 0.9em; color: #555;">
    The optimizer requires at least <b>{effective_initial_random_points}</b> total initial experiments
    to build its first reliable model before suggesting optimized points.
    <br><br>
    This count includes any existing data you provide. If you provide fewer than this number,
    the optimizer will automatically suggest additional random experiments
    until this total is reached. This initial phase helps explore the search space effectively.
    </p>
    """
    initial_random_points_info_div.text = info_text

def lock_in_setup():
    """Reads all setup widgets, creates the Optimizer, and transitions the UI."""
    update_status("🔄 Initializing...")
    doc.add_next_tick_callback(lambda: set_ui_state(lock_all=True))

    try:
        num_params_val = num_params_spinner.value
        num_initial_data_val = initial_data_spinner.value or 0
        
        config = {
            "num_params": num_params_val,
            "objective_name": objective_name_input.value or "Objective",
            "maximize": objective_type_select.active == 0,
            "surrogate_model": surrogate_select.value,
            "acq_func": acq_func_select.value,
            "params": [],
            "initial_data": []
        }

        # Make n_initial_points more robust
        config["initial_random_points"] = max(5, 2 * config["num_params"], num_initial_data_val)

        p_names_check = set()
        for i in range(config["num_params"]):
            name = param_name_inputs[i].value.strip()
            if not name or name in p_names_check:
                raise ValueError(f"Invalid or duplicate name for Parameter {i+1}")
            p_names_check.add(name)

            low = param_low_spinners[i].value
            high = param_high_spinners[i].value
            if low is None or high is None or low >= high:
                raise ValueError(f"Invalid range for '{name}'.")

            config["params"].append({"name": name, "low": low, "high": high})

        if not config["params"]:
            raise ValueError("At least one parameter must be defined.")

        # Collect initial data from UI
        for i in range(num_initial_data_val):
            obj_val = initial_objective_inputs[i].value
            if obj_val == None:
                continue
                
            x_vals = []
            for j in range(num_params_val):
                val = initial_param_inputs[i][j].value
                if val is None:
                    break
                x_vals.append(val)
            else:
                if len(x_vals) == num_params_val:
                    config["initial_data"].append({"x": x_vals, "y": obj_val})

    except Exception as e:
        update_status(f"❌ Error: {e}", is_error=True)
        set_ui_state(phase='setup')
        return

    def worker(config):
        global optimizer, dimensions, param_names, maximize_objective, experiment_history
        try:
            doc.add_next_tick_callback(lambda: update_status("🔄 Step 1/4: Validating parameters..."))

            dims = [Real(p['low'], p['high'], name=p['name']) for p in config['params']]
            p_names = [p['name'] for p in config['params']]

            # FIX 1: Construct table_cols with Objective as the last column
            table_cols = [
                TableColumn(field="Iteration", title="Iteration"),
            ]
            for name in p_names:
                table_cols.append(TableColumn(field=name, title=name, formatter=NumberFormatter(format="0.0000")))
            # Add objective LAST, using the actual objective name from config
            table_cols.append(TableColumn(field="Objective", title=config['objective_name'], formatter=NumberFormatter(format="0.0000")))

            doc.add_next_tick_callback(lambda: update_status("🔄 Step 2/4: Initializing optimizer model..."))

            param_names, dimensions, maximize_objective = p_names, dims, config['maximize']
            
            # Update direction indicator
            doc.add_next_tick_callback(update_direction_indicator)

            def update_table_cols():
                data_table.columns = table_cols
            doc.add_next_tick_callback(update_table_cols)

            model_map = {
                "GP": GaussianProcessRegressor(normalize_y=True), 
                "RF": RandomForestRegressor(n_estimators=100), 
                "ET": ExtraTreesRegressor(n_estimators=100)
            }

            optimizer = Optimizer(
                dimensions=dimensions,
                base_estimator=model_map[config['surrogate_model']],
                acq_func=config['acq_func'],
                n_initial_points=config['initial_random_points']
            )

            doc.add_next_tick_callback(lambda: update_status("🔄 Step 3/4: Processing initial data..."))

            experiment_history = []
            initial_xs, initial_ys = [], []
            for point in config["initial_data"]:
                x, y = point["x"], point["y"]
                initial_xs.append(x)
                internal_y = -y if maximize_objective else y
                initial_ys.append(internal_y)
                experiment_history.append((x, internal_y))

            if initial_xs:
                optimizer.tell(initial_xs, initial_ys)

            doc.add_next_tick_callback(lambda: update_status("🔄 Step 4/4: Finalizing setup..."))

            def final_callback():
                # Initialize new_cols with all dynamic parameter names
                # It's important to set up all keys in the source initially
                initial_source_data = {'Iteration': [], 'Objective': []}
                for name in param_names:
                    initial_source_data[name] = []
                experiments_source.data = initial_source_data # Reset with all potential keys
                
                # If there's initial data, populate it
                if initial_xs:
                    new_data = {"Iteration": list(range(1, len(initial_xs) + 1))}
                    new_data["Objective"] = [point['y'] for point in config["initial_data"]]
                    for i, name in enumerate(param_names):
                        new_data[name] = [x[i] for x in initial_xs]
                    experiments_source.data = new_data # Update with actual data

                process_and_plot_latest_results()
                update_initial_points_warning(len(experiment_history))
                update_status("🟢 Ready. Click 'Suggest Next Experiment' to begin.")
                set_ui_state(phase='optimization')

            doc.add_next_tick_callback(final_callback)

        except Exception as e:
            doc.add_next_tick_callback(partial(error_callback, e))

    threading.Thread(target=worker, args=(config,)).start()

def error_callback(e):
    """Updates the status div with an error message."""
    update_status(f"❌ Error during worker execution: {e}", is_error=True)
    set_ui_state(phase='setup')

def suggest_next_experiment():
    """Asks the optimizer for the next point and displays it with predictions."""
    global suggested_x

    try:
        if optimizer is None:
            update_status("❌ Error: Optimizer is not initialized. Please reset and complete setup.", is_error=True)
            return

        update_status("🤔 Thinking of the next best experiment...")

        suggested_x = optimizer.ask()

        expected_internal, uncertainty_val = 0.0, 1.0

        if optimizer.models:
            model = optimizer.models[-1]
            X_transformed = optimizer.space.transform([suggested_x])

            if isinstance(model, GaussianProcessRegressor):
                mean, std = model.predict(X_transformed, return_std=True)
                expected_internal, uncertainty_val = mean[0], std[0]
            elif isinstance(model, (RandomForestRegressor, ExtraTreesRegressor)):
                predictions = [tree.predict(X_transformed)[0] for tree in model.estimators_]
                expected_internal, uncertainty_val = np.mean(predictions), np.std(predictions)

        elif experiment_history:
            ys = [item[1] for item in experiment_history]
            expected_internal = np.mean(ys)
            uncertainty_val = np.std(ys) if len(ys) > 1 else 1.0

        expected_display_val = -expected_internal if maximize_objective else expected_internal
        direction_label = "Maximum" if maximize_objective else "Minimum"
        
        suggestion_html = f"<h5>💡 Suggested Experiment (Iteration {len(experiment_history)+1}):</h5>"
        suggestion_html += "<ul>"
        for name, val in zip(param_names, suggested_x):
            suggestion_html += f"<li><b>{name}:</b> {val:.4f}</li>"
        suggestion_html += "</ul>"

        suggestion_html += f"<b>Predicted {direction_label}:</b> {expected_display_val:.4f}<br>"
        suggestion_html += f"<b>Model Uncertainty:</b> {uncertainty_val:.4f}"

        suggestion_div.text = suggestion_html
        actual_result_input.disabled = False
        submit_result_button.disabled = False
        update_status("💡 Suggestion received. Please provide the result.")

    except Exception as e:
        update_status(f"❌ An error occurred while suggesting an experiment: {e}", is_error=True)

def submit_result():
    """Submits the experimental result to the optimizer and updates the display."""
    global suggested_x
    result_value = actual_result_input.value
    if result_value is None or suggested_x is None: 
        return

    update_status("🔄 Updating model with new result...")

    internal_result = -result_value if maximize_objective else result_value
    optimizer.tell(suggested_x, internal_result)
    experiment_history.append((suggested_x, internal_result))

    # Prepare new_data for streaming, ensuring all parameter columns are included
    new_data = {"Iteration": [len(experiment_history)], "Objective": [result_value]}
    for i, name in enumerate(param_names):
        new_data[name] = [suggested_x[i]]

    experiments_source.stream(new_data)

    process_and_plot_latest_results()
    update_initial_points_warning(len(experiment_history))

    suggestion_div.text = ""
    actual_result_input.value = None
    suggested_x = None
    actual_result_input.disabled = True
    submit_result_button.disabled = True
    update_status("✅ Model updated. Ready for the next suggestion.")

def process_and_plot_latest_results():
    """Finds the best result from history and updates plots and summary stats."""
    if not experiment_history:
        best_result_div.text = ""
        return

    # Find the best result
    best_idx = 0
    best_y_internal = float('inf')
    # Use actual objective value for comparison, taking into account maximization
    current_best_objective = -float('inf') if maximize_objective else float('inf')

    for idx, (x_val, y_internal) in enumerate(experiment_history):
        y_display = -y_internal if maximize_objective else y_internal
        if (maximize_objective and y_display > current_best_objective) or \
           (not maximize_objective and y_display < current_best_objective):
            current_best_objective = y_display
            best_idx = idx

    best_x = experiment_history[best_idx][0]
    best_y_display = -experiment_history[best_idx][1] if maximize_objective else experiment_history[best_idx][1]

    results_html = f"<h3>Current Best: {best_y_display:.6f}</h3><ul>"
    for name, value in zip(param_names, best_x):
        results_html += f"<li><b>{name}:</b> {value:.6f}</li>"
    results_html += "</ul>"
        
    best_result_div.text = results_html

    # Convergence tracking
    iters = list(range(1, len(experiment_history) + 1))
    best_values_so_far = []
    
    # Re-calculate current_best based on display values for the plot
    current_best_plot = -float('inf') if maximize_objective else float('inf')
    for _, y_val_internal in experiment_history:
        y_val_display = -y_val_internal if maximize_objective else y_val_internal
        if (maximize_objective and y_val_display > current_best_plot) or \
           (not maximize_objective and y_val_display < current_best_plot):
            current_best_plot = y_val_display
        best_values_so_far.append(current_best_plot) # Append current best for plot

    convergence_source.data = {'iter': iters, 'best_value': best_values_so_far}

def reset_all():
    """Resets the entire application state."""
    global optimizer, param_names, dimensions, experiment_history, suggested_x, csv_file_input
    optimizer, param_names, dimensions, experiment_history, suggested_x = None, [], [], [], None

    # Reset data sources
    experiments_source.data = dict(Iteration=[], Objective=[]) # Initial minimal columns
    data_table.columns = [ # Reset DataTable columns to default
        TableColumn(field="Iteration", title="Iteration"),
        TableColumn(field="Objective", title="Objective")
    ]
    convergence_source.data = dict(iter=[], best_value=[])

    suggestion_div.text, best_result_div.text = "", ""
    actual_result_input.value = None
    
    update_initial_points_warning(0)
    update_direction_indicator()

    # Reset visual headers based on current input values (before user might change them)
    objective_header.text = f"<b>{objective_name_input.value or 'Objective'}</b>"
    for i in range(MAX_PARAMS):
        initial_data_headers[i].text = f"<b>{param_name_inputs[i].value or f'Param {i+1}'}</b>"

    # Reset all initial data input spinners
    initial_data_spinner.value = 0
    for i in range(MAX_INITIAL_POINTS):
        initial_objective_inputs[i].value = None
        for j in range(MAX_PARAMS):
            initial_param_inputs[i][j].value = None

    # THE FIX for FileInput: Replace the FileInput widget instance to clear its state
    new_csv_file_input = FileInput(
        accept=".csv",
        multiple=False,
        title="Upload CSV Data",
        width=200
    )
    # Re-attach its callback
    new_csv_file_input.on_change('value', handle_csv_upload)
    
    # Update the global reference
    csv_file_input = new_csv_file_input

    # Find the row that contains the csv_file_input and replace its child
    # Based on your layout, it's the second child (index 1) of initial_data_layout
    initial_data_layout.children[1] = row(initial_data_spinner, csv_file_input)


    update_status("🟢 Ready. Define optimization problem.")
    set_ui_state(phase='setup')
    update_initial_random_points_info()

def update_status(message, is_error=False):
    """Updates the status div with a message and optional error styling."""
    status_div.text = message
    if is_error:
        status_div.styles = {
            'background-color': '#F8D7DA',
            'color': '#721C24',
            'border': '1px solid #F5C6CB',
            'padding': '10px',
            'border-radius': '5px'
        }
    else:
        status_div.styles = {
            'background-color': 'transparent',
            'color': 'black',
            'border': 'none'
        }

def handle_csv_upload(attr, old, new):
    """
    Handles the uploaded CSV file, parses it, and populates the initial data fields.
    """
    if not new:
        return

    try:
        # Decode base64 content
        decoded_csv_content = base64.b64decode(new).decode('utf-8-sig')
        csv_data = io.StringIO(decoded_csv_content)
        reader = csv.reader(csv_data)
        
        header = next(reader, None)
        if not header:
            raise ValueError("CSV file is empty or missing header row.")

        # CHANGE: Last column is objective, rest are parameters
        obj_col_name = header[-1].strip()  # Last column is objective
        csv_param_names = [name.strip() for name in header[:-1]]  # All except last are parameters

        # Validate against current setup
        current_num_params = num_params_spinner.value
        current_param_names = [param_name_inputs[i].value.strip() for i in range(current_num_params)]
        current_obj_name = objective_name_input.value.strip()

        if obj_col_name.lower() != current_obj_name.lower():
            update_status(f"CSV header objective '{obj_col_name}' does not match '{current_obj_name}'.", is_error=True)
            return
        
        if len(csv_param_names) != current_num_params:
            update_status(f"CSV has {len(csv_param_names)} parameters, but current setup expects {current_num_params}.", is_error=True)
            return

        # Check parameter names
        for p_csv, p_setup in zip(csv_param_names, current_param_names):
            if p_csv.lower() != p_setup.lower():
                update_status("CSV parameter names do not match current setup.", is_error=True)
                return

        parsed_data = []
        for i, row_data in enumerate(reader):
            if not row_data:
                continue
            if len(row_data) != len(header):
                update_status(f"Row {i+2} has incorrect number of columns.", is_error=True)
                return

            try:
                # CHANGE: Last value is objective, rest are parameters
                obj_value = float(row_data[-1])
                param_values = [float(v) for v in row_data[:-1]]
                parsed_data.append({"x": param_values, "y": obj_value})
            except ValueError as ve:
                update_status(f"Row {i+2} contains non-numeric data: {ve}", is_error=True)
                return
        
        if not parsed_data:
            update_status("No valid data rows found in CSV.", is_error=True)
            return

        if len(parsed_data) > MAX_INITIAL_POINTS:
            update_status(f"CSV contains {len(parsed_data)} data points, only first {MAX_INITIAL_POINTS} loaded.", is_error=False)
            parsed_data = parsed_data[:MAX_INITIAL_POINTS]

        # Reset previous data
        for i in range(MAX_INITIAL_POINTS):
            initial_objective_inputs[i].value = None
            for j in range(MAX_PARAMS):
                initial_param_inputs[i][j].value = None

        # Populate UI with parsed data
        for i, data_point in enumerate(parsed_data):
            initial_objective_inputs[i].value = data_point["y"]
            for j, val in enumerate(data_point["x"]):
                initial_param_inputs[i][j].value = val
        
        # Set spinner value and update UI
        initial_data_spinner.value = len(parsed_data)
        doc.add_next_tick_callback(partial(on_initial_data_change, 'value', None, len(parsed_data)))

        update_status(f"✅ Successfully loaded {len(parsed_data)} data points from CSV.", is_error=False)
        
    except Exception as e:
        update_status(f"❌ Error processing CSV: {e}", is_error=True)

# --- UI Widget Definitions ---
setup_title = Div(text="<h2>1. Define Optimization Problem</h2>")
num_params_spinner = Spinner(title="Number of Input Parameters", low=1, high=20, step=1, value=2, width=200)
objective_name_input = TextInput(title="Objective Name (e.g., Yield, Purity):", value="Objective")
objective_type_select = RadioButtonGroup(labels=["Maximize", "Minimize"], active=0)

# Direction indicator
direction_indicator = Div(text="(Maximizing)", styles={
    'color': 'green',
    'font-style': 'italic',
    'font-weight': 'bold',
    'margin-left': '10px'
})

param_rows, param_name_inputs, param_low_spinners, param_high_spinners = [], [], [], []
initial_data_headers = []

for i in range(MAX_PARAMS):
    name_input = TextInput(value=f"Parameter {i+1}", width=150)
    low_spinner = Spinner(title="Min", value=10, step=1, low=None, high=None, width=100)
    high_spinner = Spinner(title="Max", value=50, step=1, low=None, high=None, width=100)

    param_name_inputs.append(name_input)
    param_low_spinners.append(low_spinner)
    param_high_spinners.append(high_spinner)
    param_rows.append(row(name_input, low_spinner, high_spinner, sizing_mode="stretch_width", visible=(i < num_params_spinner.value)))

    header_div = Div(text=f"<b>Parameter {i+1}</b>", width=80, visible=(i < num_params_spinner.value), styles={'text-align': 'center'})
    initial_data_headers.append(header_div)

    name_input.on_change('value', partial(on_param_name_change, i))
    low_spinner.on_change('value', partial(on_param_range_change, i))
    high_spinner.on_change('value', partial(on_param_range_change, i))

initial_data_title = Div(text="<h4>Enter Existing Experimental Data (Recommended)</h4>")
initial_data_spinner = Spinner(title="Number of existing data points:", low=0, high=MAX_INITIAL_POINTS, step=1, value=0, width=200)

# CSV File Input (Initial Definition)
csv_file_input = FileInput(
    accept=".csv",
    multiple=False,
    title="Upload CSV Data",
    width=200
)

# NEW: Div for CSV format instruction
csv_format_instruction = Div(text="""
    <p style="font-size: 0.9em; color: #555;">
    <b>CSV Format:</b> Your CSV should have input parameters starting from the first column and first row as headers,
    and the optimized/objective variable must be in the last column of the file. Each row represents a single experiment.
    </p>
""", styles={'margin-top': '5px', 'margin-bottom': '10px'})


warning_text = """
<p style="color: #856404; background-color: #FFF3CD; border: 1px solid #FFEEBA; padding: 10px; border-radius: 5px;">
⚠️ <b>Note:</b> It is recommended to have at least 2 * number of variables as entered data points. Model performance may be poor with fewer points.
</p>
"""
initial_points_warning_div = Div(text=warning_text, visible=False)

# Div to inform about initial random points strategy
initial_random_points_info_div = Div(text="", styles={'margin-top': '10px'})

objective_header = Div(text=f"<b>{objective_name_input.value}</b>", width=80, styles={'text-align': 'center'})

# Initial definition of header row for data entry, Parameters first, objective last
initial_data_header_row = row(*initial_data_headers, objective_header, sizing_mode="stretch_width", visible=False)

initial_data_rows, initial_param_inputs, initial_objective_inputs = [], [], []
for i in range(MAX_INITIAL_POINTS):
    default_low = param_low_spinners[0].value
    default_high = param_high_spinners[0].value
    param_inputs = [Spinner(width=80, step=0.01, value=None, low=default_low, high=default_high, visible=(j < num_params_spinner.value)) for j in range(MAX_PARAMS)]
    obj_input = Spinner(width=80, step=0.01, value=None)
    initial_param_inputs.append(param_inputs)
    initial_objective_inputs.append(obj_input)
    # CHANGE: Parameters first, objective last in each row
    initial_data_rows.append(row(*param_inputs, obj_input, sizing_mode="stretch_width", visible=False))

model_title = Div(text="<h4>Model Configuration</h4>")
surrogate_select = Select(title="Surrogate Model:", value="GP", options=["GP", "RF", "ET"])
acq_func_select = Select(title="Acquisition Function:", value="gp_hedge", options=["gp_hedge", "EI", "PI", "LCB"])
lock_setup_button = Button(label="Lock Setup & Start Optimization", button_type="primary", width=400)

workflow_title = Div(text="<h2>2. Run Optimization Workflow</h2>")
suggest_button = Button(label="Suggest Next Experiment", button_type="success", width=400)
suggestion_div = Div()
actual_result_input = Spinner(title="Enter Measured Objective Value:", value=None, step=0.01)
submit_result_button = Button(label="Submit Result & Update Model", button_type="warning", width=400)

status_div = Div(text="🟢 Ready. Define optimization problem.")
best_result_div = Div()
# DataTable columns are dynamically set in lock_in_setup, so initial definition can be minimal
data_table = DataTable(source=experiments_source, columns=[TableColumn(field="Iteration", title="Iteration"), TableColumn(field="Objective", title="Objective")], width=600, height=200, editable=False)
p_conv = figure(height=300, width=600, title="Convergence Plot", x_axis_label="Iteration", y_axis_label="Best Objective Value")
p_conv.line(x='iter', y='best_value', source=convergence_source, line_width=2)
reset_button = Button(label="Reset Experiment", button_type="danger", width=400)

# Add a new button for CSV download
download_csv_button = Button(label="Download Experiment Data (CSV)", button_type="default", width=250)

# Layout for initial data, needs to be re-created if csv_file_input is replaced
initial_data_layout = column(
    initial_data_title,
    row(initial_data_spinner, csv_file_input),
    csv_format_instruction, # <--- ADDED THIS LINE
    initial_points_warning_div,
    initial_random_points_info_div,
    initial_data_header_row,
    *initial_data_rows
)

setup_widgets = [
    num_params_spinner, objective_name_input, objective_type_select,
    surrogate_select, acq_func_select, lock_setup_button, initial_data_spinner,
] + param_name_inputs + param_low_spinners + param_high_spinners + [
    item for sublist in initial_param_inputs for item in sublist
] + initial_objective_inputs

# Add the row containing csv_file_input (which is the second child of initial_data_layout)
# and the new csv_format_instruction to setup_widgets for correct disabling
setup_widgets.append(initial_data_layout.children[1]) # row(initial_data_spinner, csv_file_input)
setup_widgets.append(csv_format_instruction) # The new instruction div


optimization_widgets = [suggest_button, actual_result_input, submit_result_button]
all_buttons = [lock_setup_button, suggest_button, submit_result_button, reset_button]

# --- Attach Callbacks ---
objective_name_input.on_change('value', on_objective_name_change)
num_params_spinner.on_change('value', on_num_params_change)
initial_data_spinner.on_change('value', on_initial_data_change)
surrogate_select.on_change('value', on_surrogate_model_change)
# Initial attachment of the callback for csv_file_input
csv_file_input.on_change('value', handle_csv_upload)

def on_objective_type_change(attr, old, new):
    global maximize_objective
    maximize_objective = (new == 0)
    update_direction_indicator()
objective_type_select.on_change('active', on_objective_type_change)

lock_setup_button.on_click(lock_in_setup)
suggest_button.on_click(suggest_next_experiment)
submit_result_button.on_click(submit_result)
reset_button.on_click(reset_all)

# CustomJS for CSV download button
download_csv_button.js_on_click(CustomJS(args=dict(source=experiments_source, maximize=maximize_objective, obj_name_input=objective_name_input), code="""
    const data = source.data;
    const file_name = 'experiment_data.csv';
    const is_maximizing = maximize;
    const objective_actual_name = obj_name_input.value || 'Objective'; // Get actual objective name

    let csv_content = "";

    // Dynamically get all unique keys (column fields) from the data source
    let all_fields = Object.keys(data);
    
    // Remove "Iteration" and "Objective" (which is now dynamic) from the list temporarily
    const iteration_field = "Iteration";
    const objective_field = "Objective"; // This field is fixed in the ColumnDataSource

    const parameter_fields = all_fields.filter(field => 
        field !== iteration_field && field !== objective_field
    );

    // Construct header titles in desired order: Iteration, Parameters, Objective
    const header_titles = [iteration_field, ...parameter_fields, objective_actual_name];
    // Construct header fields (which are the actual keys in source.data)
    const header_fields_ordered = [iteration_field, ...parameter_fields, objective_field];

    csv_content += header_titles.join(",") + "\\n";

    // Prepare rows for sorting
    const all_rows = [];
    const num_rows = data[iteration_field] ? data[iteration_field].length : 0; 

    for (let i = 0; i < num_rows; i++) {
        const row_values_obj = {}; // Stores raw values for sorting
        const row_display_values = []; // Stores formatted values for CSV string

        for (let j = 0; j < header_fields_ordered.length; j++) {
            const col_field = header_fields_ordered[j];
            let value = data[col_field][i];
            
            // Store raw value for sorting (using the fixed 'Objective' field name)
            row_values_obj[col_field] = value;

            // Format for CSV output
            if (value === null || typeof value === 'undefined') {
                value = ''; // Handle null or undefined values
            } else if (typeof value === 'string' && (value.includes(',') || value.includes('"') || value.includes('\\n'))) {
                // Enclose strings with special characters in double quotes and escape existing double quotes
                value = '"' + value.replace(/"/g, '""') + '"';
            } else if (typeof value === 'number') {
                // Format numbers to a reasonable precision, adjust as needed
                value = value.toFixed(6); 
            }
            row_display_values.push(value);
        }
        all_rows.push({
            raw: row_values_obj,
            display: row_display_values.join(",")
        });
    }

    // Sort rows by Objective
    if (all_rows.length > 0) { 
        all_rows.sort((a, b) => {
            const objA = a.raw[objective_field]; // Use the fixed 'Objective' field for sorting
            const objB = b.raw[objective_field];
            if (is_maximizing) {
                return objB - objA; // Descending for maximization
            } else {
                return objA - objB; // Ascending for minimization
            }
        });
    }

    // Append sorted rows to CSV content
    for (let i = 0; i < all_rows.length; i++) {
        csv_content += all_rows[i].display + "\\n";
    }

    const blob = new Blob([csv_content], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a");
    if (link.download !== undefined) { // feature detection
        const url = URL.createObjectURL(blob);
        link.setAttribute("href", url);
        link.setAttribute("download", file_name);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
"""))


# Combine objective name and direction indicator
objective_row = row(objective_name_input, direction_indicator)

controls_col = column(
    setup_title, num_params_spinner, *param_rows,
    Div(text="<b>Objective Configuration:</b>"),
    objective_row, objective_type_select,
    initial_data_layout, # This layout contains the csv_file_input row
    model_title, row(surrogate_select, acq_func_select),
    lock_setup_button, workflow_title,
    suggest_button, suggestion_div, actual_result_input, submit_result_button,
    reset_button, status_div,
    width=500
)

# Update the results_col layout to include the download button
results_col = column(best_result_div, data_table, download_csv_button, p_conv)

main_layout = row(controls_col, results_col)

doc.add_root(main_layout)

# --- Document Ready Handler ---
def on_doc_ready(event):
    set_ui_state(phase='setup')
    update_status("🟢 Ready. Define optimization problem.")
    for i in range(num_params_spinner.value):
        on_param_range_change(i, None, None, None)
    update_initial_random_points_info()
    update_direction_indicator()
    # Ensure initial data layout is hidden on initial load
    initial_data_spinner.value = 0
    initial_data_header_row.visible = False
    for row_widget in initial_data_rows:
        row_widget.visible = False

doc.on_event("document_ready", on_doc_ready)
