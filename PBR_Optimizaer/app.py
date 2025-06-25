#This code is only for Lutein Concentration optimization
import numpy as np
from scipy.integrate import solve_ivp
from skopt import Optimizer
from skopt.space import Real
from skopt.sampler import Lhs, Sobol
import threading

from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    Button,
    Select,
    Div,
    Spinner,
    Paragraph,
    DataTable,
    TableColumn,
    NumberFormatter,
    RangeSlider,
    LinearAxis,
    DataRange1d,
    NumeralTickFormatter,
    FixedTicker,
)

# --- 1. Define Model Parameters (Constants of the Lutein system) ---
U_M = 0.152
U_D = 5.95e-3
K_N = 30.0e-3
Y_NX = 0.305
K_M = 0.350e-3 * 2
K_D = 3.71 * 0.05 / 90
K_NL = 10.0e-3
K_S = 142.8
K_I = 214.2
K_SL = 320.6
K_IL = 480.9
TAU = 0.120
KA = 0.0

# --- Global Variables for ODE solver ---
C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = 0.5, 1.0, 8e-3, 10.0, 150.0

# --- Global state for the interactive experiment ---
optimization_history = [] # Stores all evaluated points [x_params, y_value]
optimizer = None # The scikit-optimize Optimizer object
previewed_point = None # Stores the point from the "Preview" action

# --- 2. Define the Photobioreactor ODE Model ---
def pbr(t, C):
    """Defines the system of Ordinary Differential Equations for the photobioreactor."""
    C_X, C_N, C_L = C
    if C_X < 1e-9: C_X = 1e-9
    if C_N < 1e-9: C_N = 1e-9
    if C_L < 1e-9: C_L = 1e-9

    I = 2 * I0_model * (np.exp(-(TAU * 0.01 * 1000 * C_X)))
    Iscaling_u = I / (I + K_S + I ** 2 / K_I)
    Iscaling_k = I / (I + K_SL + I ** 2 / K_IL)
    u0 = U_M * Iscaling_u
    k0 = K_M * Iscaling_k

    dCxdt = u0 * C_N * C_X / (C_N + K_N) - U_D * C_X
    dCndt = -Y_NX * u0 * C_N * C_X / (C_N + K_N) + F_in_model * C_N_in_model
    dCldt = k0 * C_N * C_X / (C_N + K_NL) - K_D * C_L * C_X
    return np.array([dCxdt, dCndt, dCldt])

# --- 3. Helper function to evaluate the model and objective ---
def _evaluate_lutein_model_objective(*args):
    """Sets up and runs a single simulation to find the final lutein concentration."""
    global C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model
    C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = args
    
    sol = solve_ivp(pbr, [0, 150], [C_x0_model, C_N0_model, 0.0], t_eval=[150], method="RK45")
    final_lutein = sol.y[2, -1]
    if not np.isfinite(final_lutein) or final_lutein <= 0:
        return 1e6  # Penalty for non-physical results
    return -final_lutein # Return negative because optimizer minimizes

# --- 4. Bokeh Application Setup ---
doc = curdoc()
doc.title = "Lutein Production Optimizer"

# --- Data Sources ---
convergence_source = ColumnDataSource(data=dict(iter=[], best_lutein=[]))
simulation_source = ColumnDataSource(data=dict(time=[], C_X=[], C_N=[], C_L=[], C_L_scaled=[]))
experiments_source = ColumnDataSource(data=dict(C_x0=[], C_N0=[], F_in=[], C_N_in=[], I0=[], Lutein=[]))

# --- UI and Workflow Functions ---
def set_ui_state(lock_all=False):
    """Central function to manage the enabled/disabled state of all buttons."""
    if lock_all:
        for w in all_buttons: w.disabled = True
        return

    has_points = len(experiments_source.data['C_x0']) > 0
    has_uncalculated_points = has_points and any(np.isnan(v) for v in experiments_source.data['Lutein'])
    has_calculated_points = has_points and not has_uncalculated_points
    is_preview_pending = previewed_point is not None

    for widget in param_and_settings_widgets: widget.disabled = has_points
    
    generate_button.disabled = has_points
    reset_button.disabled = not has_points
    calculate_button.disabled = not has_uncalculated_points
    suggest_button.disabled = not has_calculated_points or is_preview_pending
    run_suggestion_button.disabled = not is_preview_pending


def get_current_dimensions():
    """Reads parameter ranges from UI and creates skopt dimension objects."""
    try:
        return [
            Real(cx0_range.value[0], cx0_range.value[1], name="C_x0"),
            Real(cn0_range.value[0], cn0_range.value[1], name="C_N0"),
            Real(fin_range.value[0], fin_range.value[1], name="F_in", prior='log-uniform'),
            Real(cnin_range.value[0], cnin_range.value[1], name="C_N_in"),
            Real(i0_range.value[0], i0_range.value[1], name="I0"),
        ]
    except Exception as e:
        update_status(f"❌ Error creating dimensions: {e}")
        return None

def reset_experiment():
    """Resets the entire application state to the beginning."""
    global optimization_history, optimizer, previewed_point
    optimization_history.clear()
    optimizer = None
    previewed_point = None

    experiments_source.data = {k: [] for k in experiments_source.data}
    convergence_source.data = {k: [] for k in convergence_source.data}
    simulation_source.data = {k: [] for k in simulation_source.data}
    
    suggestion_div.text = ""
    results_div.text = ""
    update_status("🟢 Ready. Define parameters and generate initial points.")
    set_ui_state()

def generate_initial_points():
    """Generates initial experimental points based on UI settings."""
    update_status("🔄 Generating initial points...")
    set_ui_state(lock_all=True)
    dims = get_current_dimensions()
    if dims is None: set_ui_state(); return

    n_initial = n_initial_input.value
    sampler_choice = sampler_select.value
    
    try:
        if sampler_choice == 'LHS':
            sampler = Lhs(lhs_type="centered", criterion="maximin")
            x0 = sampler.generate(dims, n_samples=n_initial)
        elif sampler_choice == 'Sobol':
            sampler = Sobol()
            x0 = sampler.generate(dims, n_samples=n_initial, random_state=np.random.randint(1000))
        else: # Random
            x0 = [ [d.rvs(1)[0] for d in dims] for _ in range(n_initial) ]

        new_data = {name.name: [point[i] for point in x0] for i, name in enumerate(dims)}
        new_data['Lutein'] = [np.nan] * n_initial
        experiments_source.data = new_data
        
        update_status("🟢 Generated initial points. Ready to calculate.")
    except Exception as e:
        update_status(f"❌ Error generating points: {e}")
    finally:
        set_ui_state()

def calculate_lutein_for_table():
    """Runs simulation for the points in the table."""
    update_status("🔄 Calculating Lutein for initial points...")
    set_ui_state(lock_all=True)

    def worker():
        try:
            points_to_calc, nan_indices = [], [i for i, v in enumerate(experiments_source.data['Lutein']) if np.isnan(v)]
            if not nan_indices:
                doc.add_next_tick_callback(lambda: update_status("🟢 All points already calculated."))
                doc.add_next_tick_callback(set_ui_state); return
                
            for i in nan_indices: points_to_calc.append([experiments_source.data[name][i] for name in ['C_x0', 'C_N0', 'F_in', 'C_N_in', 'I0']])

            results = []
            for i, p in enumerate(points_to_calc):
                doc.add_next_tick_callback(lambda i=i: update_status(f"🔄 Calculating point {i+1}/{len(points_to_calc)}..."))
                obj_val = _evaluate_lutein_model_objective(*p)
                results.append(-obj_val) # Store positive lutein value
                if not any(np.array_equal(p, item[0]) for item in optimization_history):
                    optimization_history.append([p, obj_val])

            def callback():
                current_lutein_col = experiments_source.data['Lutein']
                for i, res_idx in enumerate(nan_indices): current_lutein_col[res_idx] = results[i]
                experiments_source.patch({'Lutein': [(slice(len(current_lutein_col)), current_lutein_col)]})
                update_status("✅ Calculation complete. Ready to get a suggestion.")
                set_ui_state()
            doc.add_next_tick_callback(callback)
        except Exception as e:
            error_message = f"❌ Error during calculation: {e}"
            doc.add_next_tick_callback(lambda: update_status(error_message))
            doc.add_next_tick_callback(set_ui_state)

    threading.Thread(target=worker).start()

def _ensure_optimizer_is_ready():
    """Internal helper to create and prime the optimizer if it doesn't exist."""
    global optimizer
    if optimizer is None:
        dims = get_current_dimensions()
        x_history = [item[0] for item in optimization_history]
        y_history = [item[1] for item in optimization_history]
        
        optimizer = Optimizer(
            dimensions=dims,
            base_estimator=surrogate_select.value,
            acq_func=acq_func_select.value,
            n_initial_points=len(x_history), 
            random_state=np.random.randint(1000)
        )
        
        if x_history:
            optimizer.tell(x_history, y_history)

def suggest_next_experiment():
    """Asks the optimizer for the next best point to sample, without running it."""
    global previewed_point
    update_status("🔄 Getting next suggestion preview...")
    set_ui_state(lock_all=True)

    def worker():
        global previewed_point
        try:
            _ensure_optimizer_is_ready()
            next_point = optimizer.ask()
            previewed_point = next_point # Store for the 'Run' button
            mean, std = optimizer.models[-1].predict([next_point], return_std=True)
            predicted_lutein = -mean[0]
            uncertainty = std[0]

            def callback():
                names = [d.name for d in get_current_dimensions()]
                suggestion_html = "<h5>Suggested Next Experiment:</h5>"
                suggestion_html += f"<b>Predicted Lutein: {predicted_lutein:.4f} ± {uncertainty:.4f} g/L</b><ul>"
                for name, val in zip(names, next_point):
                    suggestion_html += f"<li><b>{name}:</b> {val:.4f}</li>"
                suggestion_html += "</ul>"
                suggestion_div.text = suggestion_html
                update_status("💡 Suggestion received. You can now run this specific experiment.")
                set_ui_state()
            doc.add_next_tick_callback(callback)
        except Exception as e:
            error_message = f"❌ Error getting preview: {e}"
            doc.add_next_tick_callback(lambda: update_status(error_message))
            doc.add_next_tick_callback(set_ui_state)
    threading.Thread(target=worker).start()
    
def run_suggestion():
    """Runs the specific experiment that was previewed."""
    if previewed_point is None: return
    update_status("🔄 Running suggested experiment...")
    set_ui_state(lock_all=True)

    def worker():
        global previewed_point
        try:
            point_to_run = previewed_point
            obj_val = _evaluate_lutein_model_objective(*point_to_run)
            lutein_val = -obj_val
            optimizer.tell(point_to_run, obj_val)
            optimization_history.append([point_to_run, obj_val])
            
            def callback():
                global previewed_point
                new_point_data = {'C_x0': [point_to_run[0]],'C_N0': [point_to_run[1]],'F_in': [point_to_run[2]],'C_N_in': [point_to_run[3]],'I0': [point_to_run[4]],'Lutein': [lutein_val]}
                experiments_source.stream(new_point_data)
                opt_step_number = len(optimization_history) - n_initial_input.value
                update_status(f"✅ Ran suggested experiment as Step {opt_step_number}. Lutein: {lutein_val:.4f} g/L")
                previewed_point = None
                suggestion_div.text = ""
                process_and_plot_latest_results()
                set_ui_state()
            doc.add_next_tick_callback(callback)
        except Exception as e:
            error_message = f"❌ Error running suggestion: {e}"
            doc.add_next_tick_callback(lambda: update_status(error_message))
            doc.add_next_tick_callback(set_ui_state)

    threading.Thread(target=worker).start()


def process_and_plot_latest_results():
    """Finds the best result from the history and updates plots."""
    if not optimization_history: return
    
    best_item = min(optimization_history, key=lambda item: item[1])
    best_params, best_obj_val = best_item[0], best_item[1]
    
    max_lutein = -best_obj_val
    optimal_params = {dim.name: val for dim, val in zip(get_current_dimensions(), best_params)}

    results_html = f"<h3>Overall Best Result So Far</h3>"
    results_html += f"<b>Maximum Lutein Found:</b> {max_lutein:.4f} g/L<br/>"
    results_html += "<b>Corresponding Parameters:</b><ul>"
    for param, value in optimal_params.items(): results_html += f"<li><b>{param}:</b> {value:.4f}</li>"
    results_html += "</ul>"
    results_div.text = results_html
    
    update_convergence_plot_from_history()
    run_final_simulation(best_params)

def update_convergence_plot_from_history():
    """Recalculates and updates the entire convergence plot from the history."""
    num_initial = n_initial_input.value
    opt_history = optimization_history[num_initial:]
    if not opt_history:
        convergence_source.data = dict(iter=[], best_lutein=[])
        return
        
    iters = list(range(1, len(opt_history) + 1))
    best_lutein_so_far = []
    
    initial_points_history = optimization_history[:num_initial]
    current_best = -min(p[1] for p in initial_points_history) if initial_points_history else -np.inf

    for _, y_val in opt_history:
        lutein_val = -y_val
        if lutein_val > current_best: current_best = lutein_val
        best_lutein_so_far.append(current_best)
        
    convergence_source.data = {'iter': iters, 'best_lutein': best_lutein_so_far}
    # FIX: Explicitly set the ticks on the x-axis to be integers
    p_conv.xaxis.ticker = FixedTicker(ticks=iters)


def run_final_simulation(best_params):
    """Runs and plots a full simulation using the provided parameter set."""
    global C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model
    C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = best_params
    
    t_eval = np.linspace(0, 150, 300)
    initial_conditions = [best_params[0], best_params[1], 0.0]
    sol = solve_ivp(pbr, [0, 150], initial_conditions, t_eval=t_eval, method="RK45")
    
    simulation_source.data = {"time": sol.t, "C_X": np.maximum(0, sol.y[0]), "C_N": np.maximum(0, sol.y[1]), "C_L": np.maximum(0, sol.y[2]), "C_L_scaled": np.maximum(0, sol.y[2]) * 100}

def update_status(message): status_div.text = message

# --- UI Widgets ---
title_div = Div(text="<h1>Lutein Production Bayesian Optimizer</h1>")
description_p = Paragraph(text="""This application uses Bayesian Optimization to find the optimal operating conditions for a photobioreactor. Follow the steps to run a virtual experiment.""", width=450)

# Step 1: Parameter Ranges
param_range_title = Div(text="<h4>1. Define Parameter Search Space</h4>")
cx0_range = RangeSlider(title="C_x0 Range (g/L)", start=0, end=10, value=(0.2, 2.0), step=0.1)
cn0_range = RangeSlider(title="C_N0 Range (g/L)", start=0, end=10, value=(0.2, 2.0), step=0.1)
fin_range = RangeSlider(title="F_in Range", start=1e-5, end=1.5e-1, value=(1e-3, 1.5e-2), step=1e-4, format="0.0000")
cnin_range = RangeSlider(title="C_N_in Range (g/L)", start=0, end=50, value=(5.0, 15.0), step=0.5)
i0_range = RangeSlider(title="I0 Range (umol/m2-s)", start=0, end=1000, value=(100, 200), step=10)

# Step 2: Sampler Settings
settings_title = Div(text="<h4>2. Configure Initial Sampling & Model</h4>")
surrogate_select = Select(title="Surrogate Model:", value="GP", options=["GP", "RF", "ET"])
acq_func_select = Select(title="Acquisition Function:", value="gp_hedge", options=["gp_hedge", "EI", "PI", "LCB"])
sampler_select = Select(title="Sampling Method:", value="LHS", options=["LHS", "Sobol", "Random"])
n_initial_input = Spinner(title="Number of Initial Points:", low=1, step=1, value=10, width=150)
param_and_settings_widgets = [cx0_range, cn0_range, fin_range, cnin_range, i0_range, surrogate_select, acq_func_select, sampler_select, n_initial_input]

# Step 3: Experiment Workflow
actions_title = Div(text="<h4>3. Run Experiment Workflow</h4>")
generate_button = Button(label="A) Generate Initial Points", button_type="primary", width=400)
calculate_button = Button(label="B) Calculate Lutein for Initial Points", button_type="default", width=400)
suggest_button = Button(label="C) Suggest Next Experiment & Show Prediction", button_type="success", width=400)
suggestion_div = Div(text="", width=400)
run_suggestion_button = Button(label="D) Run Suggested Experiment & Update Model", button_type="warning", width=400)
reset_button = Button(label="Reset Experiment", button_type="danger", width=400)
all_buttons = [generate_button, calculate_button, suggest_button, run_suggestion_button, reset_button]

generate_button.on_click(generate_initial_points)
calculate_button.on_click(calculate_lutein_for_table)
suggest_button.on_click(suggest_next_experiment)
run_suggestion_button.on_click(run_suggestion)
reset_button.on_click(reset_experiment)

status_div = Div(text="🟢 Ready. Define parameters and generate initial points.")
results_div = Div(text="")

# --- Data Table & Plots ---
columns = [TableColumn(field=name, title=name, formatter=NumberFormatter(format="0.0000")) for name in experiments_source.data.keys()]
data_table = DataTable(source=experiments_source, columns=columns, width=800, height=280, editable=False)

p_conv = figure(height=300, width=800, title="Optimizer Convergence", x_axis_label="Optimization Step", y_axis_label="Max Lutein Found (g/L)", y_range=DataRange1d(start=0, range_padding=0.1, range_padding_units='percent'))
p_conv.xaxis.formatter = NumeralTickFormatter(format="0")
p_conv.line(x="iter", y="best_lutein", source=convergence_source, line_width=2)

p_sim = figure(height=300, width=800, title="Simulation with Best Parameters", x_axis_label="Time (hours)", y_axis_label="Biomass & Nitrate Conc. (g/L)", y_range=DataRange1d(start=0))
p_sim.extra_y_ranges = {"lutein_range": DataRange1d(start=0)}
p_sim.add_layout(LinearAxis(y_range_name="lutein_range", axis_label="Lutein Conc. (x100) [g/L]"), 'right')

p_sim.line(x="time", y="C_X", source=simulation_source, color="green", line_width=2, legend_label="Biomass (C_X)")
p_sim.line(x="time", y="C_N", source=simulation_source, color="blue", line_width=2, legend_label="Nitrate (C_N)")
p_sim.line(x="time", y="C_L_scaled", source=simulation_source, color="orange", line_width=3, legend_label="Lutein (C_L) x100", y_range_name="lutein_range")
p_sim.legend.location = "top_left"
p_sim.legend.click_policy = "hide"

# --- Layout ---
controls_col = column(
    title_div, description_p,
    param_range_title, cx0_range, cn0_range, fin_range, cnin_range, i0_range,
    settings_title, surrogate_select, acq_func_select, sampler_select, n_initial_input,
    actions_title, generate_button, calculate_button, suggest_button, suggestion_div, run_suggestion_button, reset_button,
    status_div,
    width=470,
)
results_col = column(data_table, results_div, p_conv, p_sim)
layout = row(controls_col, results_col)
doc.add_root(layout)

# Initialize UI
set_ui_state()
