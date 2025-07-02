import numpy as np
from scipy.integrate import solve_ivp
from skopt import Optimizer
from skopt.space import Real
from skopt.sampler import Lhs, Sobol
import threading

from bokeh.plotting import figure # Removed curdoc here
from bokeh.layouts import column, row, layout # Import layout here
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
    CustomJS,
    # Tabs, # Not directly used for embedding in Flask, handled by HTML
    # TabPanel # Not directly used for embedding in Flask, handled by HTML
)
# Removed bokeh.embed and bokeh.resources imports as they are handled in app.py

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

# --- Cost and Process Constants ---
VOLUME_L = 500  # L
AREA_M2 = 1.26  # m^2
EFFICIENCY = 2e-6  # micromol/J to mol/J

# Prices
PRICE_BIOMASS_PER_G = 0.01  # $/g for biomass
PRICE_NITROGEN_PER_G = 0.40  # $/g for nitrogen
PRICE_ENERGY_PER_KWH = 0.15  # $/kWh for energy
PRICE_LUTEIN_PER_MG = 0.20  # $/mg for lutein

# --- New Constant for Lutein Yield Calculation ---
THEORETICAL_LUTEIN_PER_BIOMASS = 0.00556 # g Lu / 1 g Biomass

# --- Global Variables for ODE solver ---
# These global variables define the current simulation parameters.
# In a multi-user setting, these would be part of a Bokeh session state.
C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = 0.5, 1.0, 8e-3, 10.0, 150.0
TIME_HOURS = 150 # Make it global and user-adjustable

# --- Global state for the interactive experiment ---
optimization_history = [] # Stores all evaluated points [x_params, y_value]
optimizer = None # The scikit-optimize Optimizer object
previewed_point = None # Stores the point from the "Preview" action
optimization_mode = "concentration"  # "concentration", "cost", or "yield"

# --- 2. Define the Photobioreactor ODE Model ---
def pbr(t, C):
    """Defines the system of Ordinary Differential Equations for the photobioreactor."""
    # This function relies on the global model parameters (C_x0_model etc.)
    # which are set by _evaluate_lutein_model_objective or run_final_simulation.
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

# --- 3. Updated Cost Calculation Functions ---
def calculate_biomass_cost(C_x0):
    """Calculate biomass cost based on initial concentration"""
    return C_x0 * VOLUME_L * PRICE_BIOMASS_PER_G

def calculate_nitrogen_cost(C_N0, C_N_in, F_in, time_hours):
    """Calculate nitrogen cost based on initial and feed concentrations"""
    initial_cost = C_N0 * VOLUME_L * PRICE_NITROGEN_PER_G
    feed_cost = F_in * C_N_in * time_hours * PRICE_NITROGEN_PER_G
    return initial_cost + feed_cost

def calculate_energy_cost(I0, time_hours):
    """Calculate energy cost for lighting"""
    # Convert I0 from umol/m2-s to mol/m2-s
    I0_mol = I0 * 1e-6
    # Calculate energy required in Joules: mol/s / (mol/J) = J/s
    energy_J_per_s = I0_mol * AREA_M2 / EFFICIENCY
    # Convert to kWh: J/s * (3600 s/h) / (3.6e6 J/kWh) = kWh/h
    energy_kWh_per_h = energy_J_per_s * 3600 / 3.6e6
    # Total energy cost
    return energy_kWh_per_h * time_hours * PRICE_ENERGY_PER_KWH

def calculate_lutein_profit(lutein_concentration):
    """Calculate profit from lutein production"""
    lutein_mg_per_L = lutein_concentration * 1000  # g/L to mg/L
    return lutein_mg_per_L * VOLUME_L * PRICE_LUTEIN_PER_MG

def calculate_total_cost_and_profit(params, lutein_concentration, time_hours):
    """Calculate total cost, profit, and revenue (J)"""
    C_x0, C_N0, F_in, C_N_in, I0 = params
    
    biomass_cost = calculate_biomass_cost(C_x0)
    nitrogen_cost = calculate_nitrogen_cost(C_N0, C_N_in, F_in, time_hours)
    energy_cost = calculate_energy_cost(I0, time_hours)
    total_cost = biomass_cost + nitrogen_cost + energy_cost
    
    lutein_profit = calculate_lutein_profit(lutein_concentration)
    
    J = total_cost - lutein_profit # Revenue defined as (Cost - Profit) to be minimized for optimization
    
    return {
        'biomass_cost': biomass_cost,
        'nitrogen_cost': nitrogen_cost,
        'energy_cost': energy_cost,
        'total_cost': total_cost,
        'lutein_profit': lutein_profit,
        'revenue_J': J
    }

# --- New Yield Calculation Function ---
def calculate_lutein_yield(C_x_final, C_L_final):
    """Calculate the yield of lutein."""
    total_lutein_concentration_at_end = C_x_final + C_L_final
    theoretical_lutein_concentration = THEORETICAL_LUTEIN_PER_BIOMASS * total_lutein_concentration_at_end
    
    if theoretical_lutein_concentration <= 0: return 0.0 # Avoid division by zero
    
    yield_percent = (C_L_final / theoretical_lutein_concentration) * 100
    return yield_percent

# --- 4. Helper function to evaluate the model and objective ---
def _evaluate_lutein_model_objective(*args):
    """
    Sets up and runs a single simulation to find the final lutein concentration,
    and returns the objective value based on the current optimization mode.
    Temporarily sets global model parameters for the ODE solver.
    """
    global C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model, optimization_mode, TIME_HOURS

    # Store current global parameters to restore them later
    temp_Cx0, temp_CN0, temp_Fin, temp_CNin, temp_I0 = C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model

    # Set global parameters for the current simulation
    C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = args

    # Run the ODE simulation
    sol = solve_ivp(pbr, [0, TIME_HOURS], [C_x0_model, C_N0_model, 0.0], t_eval=[TIME_HOURS], method="RK45")
    final_lutein = sol.y[2, -1]
    final_biomass = sol.y[0, -1]

    # Restore global parameters to their state before this function call
    C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = temp_Cx0, temp_CN0, temp_Fin, temp_CNin, temp_I0

    if not np.isfinite(final_lutein) or final_lutein <= 0:
        return 1e6 # Return a large penalty for invalid or non-positive lutein concentrations
    
    if optimization_mode == "concentration":
        return -final_lutein # Minimize the negative concentration to maximize concentration
    elif optimization_mode == "cost":
        cost_analysis = calculate_total_cost_and_profit(args, final_lutein, TIME_HOURS)
        return cost_analysis['revenue_J'] # Minimize revenue_J (Total_Cost - Lutein_Profit) to maximize overall profit
    else: # optimization_mode == "yield"
        lutein_yield = calculate_lutein_yield(final_biomass, final_lutein)
        return -lutein_yield # Minimize the negative yield to maximize yield

# --- Main function to create and return the Bokeh layout ---
def create_bokeh_layout_for_flask():
    """
    Creates and returns the Bokeh layout for the Lutein Production Optimizer application.
    This function is designed to be called by a web framework like Flask for embedding.
    It should NOT contain `curdoc()` calls for adding roots, as it just returns the layout.
    """

    # --- Data Sources (Must be created within this function scope) ---
    convergence_source = ColumnDataSource(data=dict(iter=[], best_value=[]))
    simulation_source = ColumnDataSource(data=dict(time=[], C_X=[], C_N=[], C_L=[], C_L_scaled=[]))
    experiments_source = ColumnDataSource(data=dict(
        C_x0=[], C_N0=[], F_in=[], C_N_in=[], I0=[], 
        Lutein=[], Total_Cost=[], Lutein_Profit=[], Revenue_J=[],
        Lutein_Yield=[], Biomass_Cost=[], Nitrogen_Cost=[], Energy_Cost=[]
    ))

    # --- UI Widgets (Must be created within this function scope) ---
    title_div = Div(text="<h1>Lutein Production Bayesian Optimizer with Cost Analysis</h1>")
    description_p = Paragraph(text="""This application uses Bayesian Optimization to find optimal operating conditions for a photobioreactor. You can optimize for maximum lutein concentration, minimum cost, or maximum yield. Follow the steps to run a virtual experiment.""", width=450)

    objective_title = Div(text="<h4>0. Select Optimization Objective</h4>")
    objective_select = Select(title="Optimization Objective:", value="concentration", 
                              options=[("concentration", "Maximize Lutein Concentration"), 
                                       ("cost", "Minimize Cost (Maximize Revenue)"),
                                       ("yield", "Maximize Lutein Yield")])

    time_hours_input = Spinner(title="Simulation Time (Hours):", low=1, step=1, value=TIME_HOURS, width=150, visible=False)

    param_range_title = Div(text="<h4>1. Define Parameter Search Space</h4>")
    cx0_range = RangeSlider(title="C_x0 Range (g/L)", start=0, end=10, value=(0.2, 2.0), step=0.1)
    cn0_range = RangeSlider(title="C_N0 Range (g/L)", start=0, end=10, value=(0.2, 2.0), step=0.1) 
    fin_range = RangeSlider(title="F_in Range", start=1e-5, end=1.5e-1, value=(1e-3, 1.5e-2), step=1e-4, format="0.0000")
    cnin_range = RangeSlider(title="C_N_in Range (g/L)", start=0, end=50, value=(5.0, 15.0), step=0.5)
    i0_range = RangeSlider(title="I0 Range (umol/m2-s)", start=0, end=1000, value=(100, 200), step=10)

    # Photobioreactor Visualization UI elements
    indicator_panel_title = Div(text="<h4>Photobioreactor State</h4>")
    lights = [Div(text="<p>...</p>", width=60, height=60, styles={'text-align': 'center'}) for _ in range(6)]
    spacer = Div(text="", width=200, height=200) # This div is used for the "liquid" visualization

    settings_title = Div(text="<h4>2. Configure Initial Sampling & Model</h4>")
    surrogate_select = Select(title="Surrogate Model:", value="GP", options=["GP", "RF", "ET"])
    acq_func_select = Select(title="Acquisition Function:", value="gp_hedge", options=["EI", "PI", "LCB", "gp_hedge"])
    sampler_select = Select(title="Sampling Method:", value="LHS", options=["LHS", "Sobol", "Random"])
    n_initial_input = Spinner(title="Number of Initial Points:", low=1, step=1, value=10, width=150)
    param_and_settings_widgets = [objective_select, cx0_range, cn0_range, fin_range, cnin_range, i0_range, surrogate_select, acq_func_select, sampler_select, n_initial_input]

    actions_title = Div(text="<h4>3. Run Experiment Workflow</h4>")
    generate_button = Button(label="A) Generate Initial Points", button_type="primary", width=400)
    calculate_button = Button(label="B) Calculate Lutein & Costs for Initial Points", button_type="default", width=400)
    suggest_button = Button(label="C) Suggest Next Experiment & Show Prediction", button_type="success", width=400)
    suggestion_div = Div(text="", width=400)
    run_suggestion_button = Button(label="D) Run Suggested Experiment & Update Model", button_type="warning", width=400)
    reset_button = Button(label="Reset Experiment", button_type="danger", width=400)
    all_buttons = [generate_button, calculate_button, suggest_button, run_suggestion_button, reset_button]

    status_div = Div(text="🟢 Ready. Select objective and define parameters, then generate initial points.")
    results_div = Div(text="")

    # --- Callbacks / Workflow Functions (defined within this function scope) ---

    def update_status(message):
        """Updates the text in the status_div."""
        status_div.text = message

    def set_ui_state(lock_all=False):
        """Central function to manage the enabled/disabled state of all buttons and inputs."""
        if lock_all:
            for w in all_buttons: w.disabled = True
            for w in param_and_settings_widgets: w.disabled = True
            time_hours_input.disabled = True
            return

        has_points = len(experiments_source.data['C_x0']) > 0
        has_uncalculated_points = has_points and any(np.isnan(v) for v in experiments_source.data['Lutein'])
        has_calculated_points = has_points and not has_uncalculated_points
        is_preview_pending = previewed_point is not None

        for widget in param_and_settings_widgets:
            widget.disabled = has_points # Disable parameter inputs once points are generated

        # Only enable time_hours_input if in cost optimization mode AND no points generated yet
        time_hours_input.disabled = not (optimization_mode == "cost" and not has_points)

        generate_button.disabled = has_points
        calculate_button.disabled = not has_uncalculated_points
        suggest_button.disabled = not has_calculated_points or is_preview_pending
        run_suggestion_button.disabled = not is_preview_pending
        reset_button.disabled = not has_points

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

    def reset_experiment(attr=None, old=None, new=None): # Added args for on_click compatibility
        """Resets the entire application state to the beginning."""
        global optimization_history, optimizer, previewed_point, TIME_HOURS, optimization_mode
        optimization_history.clear()
        optimizer = None
        previewed_point = None

        # Clear all data sources
        experiments_source.data = {k: [] for k in experiments_source.data}
        convergence_source.data = {k: [] for k in convergence_source.data}
        simulation_source.data = {k: [] for k in simulation_source.data}
        
        suggestion_div.text = ""
        results_div.text = ""

        # Reset TIME_HOURS spinner and global variable
        time_hours_input.value = 150
        TIME_HOURS = 150
        
        # Reset optimization mode selection to default
        objective_select.value = "concentration"
        optimization_mode = "concentration"

        # Reset visual components to their default appearance based on slider values
        # These are Python-side updates to Div.text or Div.styles
        cx0_default, cn0_default = cx0_range.value[1], cn0_range.value[1]
        cx_max_default, cn_max_default = cx0_range.end, cn0_range.end
        green_opacity_default = min((cx0_default / cx_max_default) * 0.8, 0.8)
        blue_opacity_default = min((cn0_default / cn_max_default) * 0.6, 0.6)
        green_gradient_default = f"linear-gradient(rgba(48, 128, 64, {green_opacity_default}), rgba(24, 64, 32, {green_opacity_default}))"
        blue_gradient_default = f"linear-gradient(rgba(52, 152, 219, {blue_opacity_default}), rgba(41, 128, 185, {blue_opacity_default}))"
        liquid_css_default = f"""
            position: absolute; bottom: 0; left: 0; right: 0; height: 95%;
            background: {green_gradient_default}, {blue_gradient_default};
            border-bottom-left-radius: 18px; border-bottom-right-radius: 18px; z-index: -1;
        """
        spacer.text = f'<div style="{liquid_css_default}"></div>'

        # Set lights to default color (based on i0_range default value)
        i0_default = i0_range.value[1]
        new_color_default = 'grey'
        if i0_default < 100: new_color_default = '#4D4C00'
        elif i0_default < 200: new_color_default = '#807E00'
        elif i0_default < 300: new_color_default = '#B3B000'
        elif i0_default < 400: new_color_default = '#E6E200'
        elif i0_default < 500: new_color_default = '#FFFF00'
        elif i0_default < 600: new_color_default = '#FFFF33'
        elif i0_default < 700: new_color_default = '#FFFF66'
        elif i0_default < 800: new_color_default = '#FFFF99'
        elif i0_default < 900: new_color_default = '#FFFFCC'
        else: new_color_default = '#FFFFF0'
        
        # Corrected f-string for HTML content with triple quotes
        new_html_default = f"""<svg height="50" width="50"><circle cx="25" cy="25" r="20" stroke="black" stroke-width="2" fill="{new_color_default}" /></svg>"""
        
        for l in lights:
            l.text = new_html_default


        update_status("🟢 Ready. Define parameters and generate initial points.")
        # Call with dummy arguments to trigger re-setup for the Python callback
        set_optimization_mode_internal("value", None, objective_select.value)
        set_ui_state()

    def set_optimization_mode_internal(attr, old, new):
        """Update optimization mode based on user selection and toggle cost/yield visibility."""
        global optimization_mode, optimizer, TIME_HOURS
        new_mode = objective_select.value
        
        # Only update if mode actually changed or optimizer is not initialized
        if new_mode == optimization_mode and optimizer is not None and len(optimization_history) > 0:
            return
        
        optimization_mode = new_mode
        optimizer = None # Reset optimizer when objective changes

        # Update plot titles and axis labels based on selected mode
        if optimization_mode == "concentration":
            p_conv.title.text = "Optimizer Convergence - Lutein Concentration"
            p_conv.yaxis.axis_label = "Max Lutein Found (g/L)"
            for col in cost_columns: col.visible = False
            for col in yield_columns: col.visible = False
            time_hours_input.visible = False
        elif optimization_mode == "cost":
            p_conv.title.text = "Optimizer Convergence - Revenue Optimization"
            p_conv.yaxis.axis_label = "Best Revenue (Profit - Cost) [$]"
            for col in cost_columns: col.visible = True
            for col in yield_columns: col.visible = False
            time_hours_input.visible = True
        else: # optimization_mode == "yield"
            p_conv.title.text = "Optimizer Convergence - Lutein Yield Optimization"
            p_conv.yaxis.axis_label = "Best Lutein Yield (%)"
            for col in cost_columns: col.visible = False
            for col in yield_columns: col.visible = True
            time_hours_input.visible = False
        
        # Update global TIME_HOURS from spinner value when switching to cost mode, or keep default
        TIME_HOURS = time_hours_input.value

        # Update status and potentially re-process results if there are existing points
        if not experiments_source.data['C_x0']:
            update_status("🟢 Ready. Select objective and define parameters, then generate initial points.")
        else:
            process_and_plot_latest_results() # Re-evaluate results based on new objective
            update_status(f"Objective changed to '{optimization_mode}'. Recalculated best results.")
        
        set_ui_state() # Update button states

    def update_time_hours(attr, old, new):
        """Updates the global TIME_HOURS variable when the spinner changes."""
        global TIME_HOURS
        TIME_HOURS = new
        update_status(f"Simulation time set to {TIME_HOURS} hours.")

    def generate_initial_points():
        """Generates initial experimental points based on UI settings."""
        # Removed doc.add_next_tick_callback as it's for Bokeh server.
        update_status("🔄 Generating initial points...")
        set_ui_state(lock_all=True) # Lock UI during generation

        def worker():
            dims = get_current_dimensions()
            if dims is None:
                update_status("❌ Failed to get dimensions. Cannot generate points.")
                set_ui_state()
                return

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
                new_data.update({
                    'Lutein': [np.nan] * n_initial, # Initialize results as NaN
                    'Total_Cost': [np.nan] * n_initial,
                    'Lutein_Profit': [np.nan] * n_initial,
                    'Revenue_J': [np.nan] * n_initial,
                    'Lutein_Yield': [np.nan] * n_initial,
                    'Biomass_Cost': [np.nan] * n_initial,
                    'Nitrogen_Cost': [np.nan] * n_initial,
                    'Energy_Cost': [np.nan] * n_initial
                })
                
                # Directly update the source data (Flask serves this)
                experiments_source.data = new_data
                update_status("🟢 Generated initial points. Ready to calculate.")
                set_ui_state()

            except Exception as e:
                update_status(f"❌ Error generating points: {e}")
                set_ui_state()

        threading.Thread(target=worker).start() # Run in a separate thread to keep UI responsive


    def calculate_lutein_for_table():
        """Runs simulation for the points in the table that haven't been calculated yet."""
        update_status("🔄 Calculating Lutein and costs for initial points...")
        set_ui_state(lock_all=True) # Lock UI during calculation

        def worker():
            try:
                nan_indices = [i for i, v in enumerate(experiments_source.data['Lutein']) if np.isnan(v)]
                if not nan_indices:
                    update_status("🟢 All points already calculated.")
                    set_ui_state()
                    return
                
                points_to_calc = []
                for i in nan_indices: 
                    points_to_calc.append([experiments_source.data[name][i] for name in ['C_x0', 'C_N0', 'F_in', 'C_N_in', 'I0']])

                results = []
                for i, p in enumerate(points_to_calc):
                    update_status(f"🔄 Calculating point {i+1}/{len(points_to_calc)}...")
                    
                    obj_val = _evaluate_lutein_model_objective(*p)
                    
                    # Temporarily set globals for precise calculation of results to display
                    global C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model, TIME_HOURS
                    temp_Cx0, temp_CN0, temp_Fin, temp_CNin, temp_I0 = C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model
                    C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = p
                    sol = solve_ivp(pbr, [0, TIME_HOURS], [p[0], p[1], 0.0], t_eval=[TIME_HOURS], method="RK45")
                    C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = temp_Cx0, temp_CN0, temp_Fin, temp_CNin, temp_I0

                    lutein_conc = sol.y[2, -1]
                    biomass_conc = sol.y[0, -1]

                    cost_analysis = calculate_total_cost_and_profit(p, lutein_conc, TIME_HOURS)
                    lutein_yield = calculate_lutein_yield(biomass_conc, lutein_conc)
                    
                    results.append({
                        'lutein': lutein_conc,
                        'total_cost': cost_analysis['total_cost'],
                        'lutein_profit': cost_analysis['lutein_profit'],
                        'revenue_j': cost_analysis['revenue_J'],
                        'lutein_yield': lutein_yield,
                        'biomass_cost': cost_analysis['biomass_cost'],
                        'nitrogen_cost': cost_analysis['nitrogen_cost'],
                        'energy_cost': cost_analysis['energy_cost']
                    })
                    
                    if not any(np.array_equal(p, item[0]) for item in optimization_history):
                        optimization_history.append([p, obj_val])

                current_data = experiments_source.data.copy()
                for i, res_idx in enumerate(nan_indices):
                    current_data['Lutein'][res_idx] = results[i]['lutein']
                    current_data['Total_Cost'][res_idx] = results[i]['total_cost']
                    current_data['Lutein_Profit'][res_idx] = results[i]['lutein_profit']
                    current_data['Revenue_J'][res_idx] = results[i]['revenue_j']
                    current_data['Lutein_Yield'][res_idx] = results[i]['lutein_yield']
                    current_data['Biomass_Cost'][res_idx] = results[i]['biomass_cost']
                    current_data['Nitrogen_Cost'][res_idx] = results[i]['nitrogen_cost']
                    current_data['Energy_Cost'][res_idx] = results[i]['energy_cost']
                
                experiments_source.data = current_data
                update_status("✅ Calculation complete. Ready to get a suggestion.")
                process_and_plot_latest_results()
                set_ui_state()
            
            except Exception as e:
                error_message = f"❌ Error during calculation: {e}"
                update_status(error_message)
                set_ui_state()

        threading.Thread(target=worker).start()


    def _ensure_optimizer_is_ready():
        """Internal helper to create and prime the optimizer if it doesn't exist."""
        global optimizer
        if optimizer is None:
            dims = get_current_dimensions()
            if not dims: raise ValueError("Optimization dimensions not defined.")
            
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
        update_status("🔄 Getting next suggestion preview... This may take a moment.")
        set_ui_state(lock_all=True)

        def worker():
            global previewed_point
            try:
                _ensure_optimizer_is_ready()
                next_point = optimizer.ask()
                previewed_point = next_point
                
                mean, std = optimizer.models[-1].predict([next_point], return_std=True)
                
                metric_text = ""
                if optimization_mode == "concentration":
                    predicted_lutein = -mean[0]
                    uncertainty = std[0]
                    metric_text = f"<b>Predicted Lutein: {predicted_lutein:.4f} ± {uncertainty:.4f} g/L</b>"
                elif optimization_mode == "cost":
                    predicted_revenue = mean[0]
                    uncertainty = std[0]
                    metric_text = f"<b>Predicted Revenue (J): ${predicted_revenue:.4f} ± {uncertainty:.4f}</b>"
                else: # optimization_mode == "yield"
                    predicted_yield = -mean[0]
                    uncertainty = std[0]
                    metric_text = f"<b>Predicted Lutein Yield: {predicted_yield:.4f} ± {uncertainty:.4f} %</b>"

                names = [d.name for d in get_current_dimensions()]
                suggestion_html = "<h5>Suggested Next Experiment:</h5>"
                suggestion_html += metric_text + "<ul>"
                for name, val in zip(names, next_point):
                    suggestion_html += f"<li><b>{name}:</b> {val:.4f}</li>"
                suggestion_html += "</ul>"
                
                suggestion_div.text = suggestion_html
                update_status("💡 Suggestion received. You can now run this specific experiment.")
                set_ui_state()
            except Exception as e:
                error_message = f"❌ Error getting preview: {e}"
                update_status(error_message)
                set_ui_state()
        threading.Thread(target=worker).start()
    
    def run_suggestion():
        """Runs the specific experiment that was previewed."""
        if previewed_point is None:
            update_status("⚠️ No suggestion to run. Please click 'Suggest Next Experiment' first.")
            return

        update_status("🔄 Running suggested experiment... This may take a moment.")
        set_ui_state(lock_all=True)

        def worker():
            global previewed_point
            try:
                point_to_run = previewed_point
                obj_val = _evaluate_lutein_model_objective(*point_to_run)
                
                global C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model, TIME_HOURS
                temp_Cx0, temp_CN0, temp_Fin, temp_CNin, temp_I0 = C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model
                C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = point_to_run
                sol = solve_ivp(pbr, [0, TIME_HOURS], [point_to_run[0], point_to_run[1], 0.0], t_eval=[TIME_HOURS], method="RK45")
                C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = temp_Cx0, temp_CN0, temp_Fin, temp_CNin, temp_I0
                lutein_val = sol.y[2, -1]
                biomass_val = sol.y[0, -1]
                cost_analysis = calculate_total_cost_and_profit(point_to_run, lutein_val, TIME_HOURS)
                lutein_yield_val = calculate_lutein_yield(biomass_val, lutein_val)
                
                _ensure_optimizer_is_ready()
                optimizer.tell(point_to_run, obj_val)
                optimization_history.append([point_to_run, obj_val])
                
                new_point_data = {
                    'C_x0': [point_to_run[0]],
                    'C_N0': [point_to_run[1]],
                    'F_in': [point_to_run[2]],
                    'C_N_in': [point_to_run[3]],
                    'I0': [point_to_run[4]],
                    'Lutein': [lutein_val],
                    'Total_Cost': [cost_analysis['total_cost']],
                    'Lutein_Profit': [cost_analysis['lutein_profit']],
                    'Revenue_J': [cost_analysis['revenue_j']],
                    'Lutein_Yield': [lutein_yield_val],
                    'Biomass_Cost': [cost_analysis['biomass_cost']],
                    'Nitrogen_Cost': [cost_analysis['nitrogen_cost']],
                    'Energy_Cost': [cost_analysis['energy_cost']]
                }
                experiments_source.stream(new_point_data)

                opt_step_number = len(optimization_history) - n_initial_input.value
                
                if optimization_mode == "concentration":
                    update_status(f"✅ Ran suggested experiment as Step {opt_step_number}. Lutein: {lutein_val:.4f} g/L")
                elif optimization_mode == "cost":
                    update_status(f"✅ Ran suggested experiment as Step {opt_step_number}. Revenue: ${cost_analysis['revenue_J']:.4f}")
                else: # optimization_mode == "yield"
                    update_status(f"✅ Ran suggested experiment as Step {opt_step_number}. Lutein Yield: {lutein_yield_val:.4f} %")
                
                previewed_point = None
                suggestion_div.text = ""
                process_and_plot_latest_results()
                set_ui_state()
            except Exception as e:
                error_message = f"❌ Error running suggestion: {e}"
                update_status(error_message)
                set_ui_state()

        threading.Thread(target=worker).start()

    def process_and_plot_latest_results():
        """Finds the best result from the history and updates plots."""
        if not optimization_history: 
            results_div.text = ""
            simulation_source.data = dict(time=[], C_X=[], C_N=[], C_L=[], C_L_scaled=[])
            return
        
        # Determine best item based on current optimization mode
        if optimization_mode == "concentration" or optimization_mode == "yield":
            best_item = max(optimization_history, key=lambda item: -item[1])
        else: # cost optimization
            best_item = min(optimization_history, key=lambda item: item[1])
            
        best_params, best_obj_val = best_item[0], best_item[1]
        
        # Run a full simulation for plotting and detailed metrics
        global C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model, TIME_HOURS
        temp_Cx0, temp_CN0, temp_Fin, temp_CNin, temp_I0 = C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model
        C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = best_params
        sol = run_final_simulation(best_params)
        C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = temp_Cx0, temp_CN0, temp_Fin, temp_CNin, temp_I0
        
        max_lutein = sol.y[2, -1]
        final_biomass = sol.y[0, -1]
        final_nitrate = sol.y[1, -1]
        cost_analysis = calculate_total_cost_and_profit(best_params, max_lutein, TIME_HOURS)
        best_yield = calculate_lutein_yield(final_biomass, max_lutein)
        
        optimal_params = {dim.name: val for dim, val in zip(get_current_dimensions(), best_params)}

        results_html = f"<h3>Overall Best Result So Far</h3>"
        if optimization_mode == "concentration":
            results_html += f"<b>Maximum Lutein Found:</b> {max_lutein:.4f} g/L<br/>"
        elif optimization_mode == "cost":
            results_html += f"<b>Best Revenue (J):</b> ${cost_analysis['revenue_J']:.4f}<br/>"
            results_html += f"<b>Lutein Concentration:</b> {max_lutein:.4f} g/L<br/>"
            results_html += f"<b>Total Cost:</b> ${cost_analysis['total_cost']:.4f}<br/>"
            results_html += f"<b>Biomass Cost:</b> ${cost_analysis['biomass_cost']:.4f}<br/>"
            results_html += f"<b>Nitrogen Cost:</b> ${cost_analysis['nitrogen_cost']:.4f}<br/>"
            results_html += f"<b>Energy Cost:</b> ${cost_analysis['energy_cost']:.4f}<br/>"
            results_html += f"<b>Lutein Profit:</b> ${cost_analysis['lutein_profit']:.4f}<br/>"
        else: # yield optimization
            results_html += f"<b>Maximum Lutein Yield Found:</b> {best_yield:.4f} %<br/>"
            results_html += f"<b>Lutein Concentration:</b> {max_lutein:.4f} g/L<br/>"
            results_html += f"<b>Total Cost:</b> ${cost_analysis['total_cost']:.4f}<br/>"
            results_html += f"<b>Lutein Profit:</b> ${cost_analysis['lutein_profit']:.4f}<br/>"
        
        results_html += "<b>Corresponding Parameters:</b><ul>"
        for param, value in optimal_params.items(): 
            results_html += f"<li><b>{param}:</b> {value:.4f}</li>"
        results_html += "</ul>"
        results_div.text = results_html

        # Update liquid visualization based on final biomass, nitrate, lutein
        GLASS_STYLE = {
            'border': '2px solid rgba(255, 255, 255, 0.4)', 'border-radius': '20px',
            'background': 'rgba(255, 255, 255, 0.15)', 'box-shadow': 'inset 0 2px 6px rgba(0, 0, 0, 0.1)',
            'position': 'relative', 'overflow': 'hidden', 'backdrop-filter': 'blur(3px)',
            '-webkit-backdrop-filter': 'blur(3px)', 'box-sizing': 'border-box'
        }
        spacer.styles = GLASS_STYLE # Apply glass style to the spacer
        orange_opacity = np.clip(max_lutein / 0.018, 0.0, 1.0) * 0.9 # Scale lutein to opacity
        orange_gradient = f"linear-gradient(rgba(255, 140, 0, {orange_opacity}), rgba(210, 105, 30, {orange_opacity}))"
        green_opacity = np.clip(final_biomass / 5.0, 0.0, 1.0) * 0.6 # Scale biomass to opacity
        green_gradient = f"linear-gradient(rgba(48, 128, 64, {green_opacity}), rgba(24, 64, 32, {green_opacity}))"
        blue_opacity = np.clip(final_nitrate / 2.0, 0.0, 1.0) * 0.5 # Scale nitrate to opacity
        blue_gradient = f"linear-gradient(rgba(52, 152, 219, {blue_opacity}), rgba(41, 128, 185, {blue_opacity}))"
        liquid_css = f"""
            position: absolute; bottom: 0; left: 0; right: 0; height: 95%;
            background: {orange_gradient}, {green_gradient}, {blue_gradient};
            border-bottom-left-radius: 18px; border-bottom-right-radius: 18px; z-index: -1;
        """
        spacer.text = f'<div style="{liquid_css}"></div>' # Update spacer's inner HTML for visualization
        
        update_convergence_plot_from_history()

    def update_convergence_plot_from_history():
        """Recalculates and updates the entire convergence plot from the history."""
        num_initial = n_initial_input.value
        
        initial_points_history = optimization_history[:num_initial]
        opt_steps_history = optimization_history[num_initial:]

        if not opt_steps_history:
            convergence_source.data = dict(iter=[], best_value=[])
            return
            
        iters = list(range(1, len(opt_steps_history) + 1))
        best_values_so_far = []
        
        current_best = None
        if initial_points_history:
            initial_obj_values = [p[1] for p in initial_points_history]
            if optimization_mode == "concentration" or optimization_mode == "yield":
                current_best = -min(initial_obj_values)
            else: # cost optimization (minimization)
                current_best = min(initial_obj_values)
        
        if current_best is None:
            if optimization_mode in ["concentration", "yield"]:
                current_best = -np.inf
            else:
                current_best = np.inf

        for _, y_val in opt_steps_history:
            if optimization_mode == "concentration" or optimization_mode == "yield":
                value_for_display = -y_val
                if value_for_display > current_best: current_best = value_for_display
            else:
                value_for_display = y_val
                if value_for_display < current_best: current_best = value_for_display
            best_values_so_far.append(current_best)
            
        convergence_source.data = {'iter': iters, 'best_value': best_values_so_far}
        p_conv.xaxis.ticker = FixedTicker(ticks=iters)
        if iters:
            p_conv.x_range.end = iters[-1] + 0.5


    def run_final_simulation(best_params):
        """Runs and plots a full simulation using the provided parameter set."""
        global C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model, TIME_HOURS
        temp_Cx0, temp_CN0, temp_Fin, temp_CNin, temp_I0 = C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model
        C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = best_params
        
        t_eval = np.linspace(0, TIME_HOURS, 300)
        initial_conditions = [best_params[0], best_params[1], 0.0]
        sol = solve_ivp(pbr, [0, TIME_HOURS], initial_conditions, t_eval=t_eval, method="RK45")
        
        C_x0_model, C_N0_model, F_in_model, C_N_in_model, I0_model = temp_Cx0, temp_CN0, temp_Fin, temp_CNin, temp_I0

        simulation_source.data = {
            "time": sol.t, 
            "C_X": np.maximum(0, sol.y[0]), 
            "C_N": np.maximum(0, sol.y[1]), 
            "C_L": np.maximum(0, sol.y[2]), 
            "C_L_scaled": np.maximum(0, sol.y[2]) * 100
        }
        return sol


    # --- Data Table & Plots Configuration ---
    columns = [
        TableColumn(field="C_x0", title="C_x0", formatter=NumberFormatter(format="0.0000")),
        TableColumn(field="C_N0", title="C_N0", formatter=NumberFormatter(format="0.0000")),
        TableColumn(field="F_in", title="F_in", formatter=NumberFormatter(format="0.0000")),
        TableColumn(field="C_N_in", title="C_N_in", formatter=NumberFormatter(format="0.0000")),
        TableColumn(field="I0", title="I0", formatter=NumberFormatter(format="0.0000")),
        TableColumn(field="Lutein", title="Lutein (g/L)", formatter=NumberFormatter(format="0.0000")),
        TableColumn(field="Total_Cost", title="Total Cost ($)", formatter=NumberFormatter(format="0.0000")),
        TableColumn(field="Lutein_Profit", title="Lutein Profit ($)", formatter=NumberFormatter(format="0.0000")),
        TableColumn(field="Revenue_J", title="Revenue J ($)", formatter=NumberFormatter(format="0.0000")),
        TableColumn(field="Lutein_Yield", title="Lutein Yield (%)", formatter=NumberFormatter(format="0.0000")),
        TableColumn(field="Biomass_Cost", title="Biomass Cost ($)", formatter=NumberFormatter(format="0.0000")),
        TableColumn(field="Nitrogen_Cost", title="Nitrogen Cost ($)", formatter=NumberFormatter(format="0.0000")),
        TableColumn(field="Energy_Cost", title="Energy Cost ($)", formatter=NumberFormatter(format="0.0000"))
    ]

    cost_columns = [col for col in columns if col.field in ['Total_Cost', 'Lutein_Profit', 'Revenue_J', 'Biomass_Cost', 'Nitrogen_Cost', 'Energy_Cost']]
    yield_columns = [col for col in columns if col.field in ['Lutein_Yield']]

    data_table = DataTable(source=experiments_source, columns=columns, width=1000, height=280, editable=False)

    p_conv = figure(height=300, width=800, title="Optimizer Convergence", x_axis_label="Optimization Step", y_axis_label="Best Value", y_range=DataRange1d(start=0, range_padding=0.1, range_padding_units='percent'))
    p_conv.xaxis.formatter = NumeralTickFormatter(format="0")
    p_conv.line(x="iter", y="best_value", source=convergence_source, line_width=2)

    p_sim = figure(height=300, width=800, title="Simulation with Best Parameters", x_axis_label="Time (hours)", y_axis_label="Biomass & Nitrate Conc. (g/L)", y_range=DataRange1d(start=0))
    p_sim.extra_y_ranges = {"lutein_range": DataRange1d(start=0)}
    p_sim.add_layout(LinearAxis(y_range_name="lutein_range", axis_label="Lutein Conc. (x100) [g/L]"), 'right')

    p_sim.line(x="time", y="C_X", source=simulation_source, color="green", line_width=2, legend_label="Biomass (C_X)")
    p_sim.line(x="time", y="C_N", source=simulation_source, color="blue", line_width=2, legend_label="Nitrate (C_N)")
    p_sim.line(x="time", y="C_L_scaled", source=simulation_source, color="orange", line_width=3, legend_label="Lutein (C_L) x100", y_range_name="lutein_range")
    p_sim.legend.location = "top_left"
    p_sim.legend.click_policy = "hide"

    # Photobioreactor visualization panel styles
    GLASS_STYLE = {
        'border': '2px solid rgba(255, 255, 255, 0.4)', 'border-radius': '20px',
        'background': 'rgba(255, 255, 255, 0.15)', 'box-shadow': 'inset 0 2px 6px rgba(0, 0, 0, 0.1)',
        'position': 'relative', 'overflow': 'hidden', 'backdrop-filter': 'blur(3px)',
        '-webkit-backdrop-filter': 'blur(3px)', 'box-sizing': 'border-box'
    }
    spacer.styles = GLASS_STYLE
    lamp_style = {
        'border': '2px solid rgba(85, 85, 85, 0.5)', 'border-radius': '15px',
        'background': 'linear-gradient(to right, rgba(136,136,136,0.5), rgba(211,211,211,0.5), rgba(136,136,136,0.5))',
        'padding': '15px 5px', 'box-sizing': 'border-box'
    }
    tube_div = Div(width=120, height=30, styles={'background': 'linear-gradient(to bottom, rgba(211,211,211,0.6), rgba(136,136,136,0.6))', 'border': '2px solid rgba(85, 85, 85, 0.6)', 'border-radius': '10px', 'margin': '0 auto -10px auto'})
    vertical_tube_div = Div(width=25, height=60, styles={'background': 'linear-gradient(to right, rgba(136,136,136,0.6), rgba(211,211,211,0.6), rgba(136,136,136,0.6))', 'border': '2px solid rgba(85, 85, 85, 0.6)', 'border-top-left-radius': '8px', 'border-top-right-radius': '8px', 'margin': '0 auto -5px auto'})

    left_light_col = column(lights[0], lights[1], lights[2], styles=lamp_style)
    right_light_col = column(lights[3], lights[4], lights[5], styles=lamp_style)
    center_column = column(vertical_tube_div, tube_div, spacer, styles={'gap': '0'})
    indicator_panel = row(left_light_col, center_column, right_light_col, align='center')

    # --- Layout Structure for the "Run Optimizer" Tab Panel ---
    controls_col = column(
        title_div, description_p,
        objective_title, objective_select, time_hours_input,
        param_range_title, cx0_range, cn0_range, fin_range, cnin_range, i0_range,
        settings_title, surrogate_select, acq_func_select, sampler_select, n_initial_input,
        actions_title, generate_button, calculate_button, suggest_button, suggestion_div, run_suggestion_button, reset_button,
        status_div,
        width=470,
    )

    results_col = column(
        data_table, 
        results_div, 
        p_conv, 
        p_sim,
        indicator_panel_title,
        indicator_panel
    )

    # This is the single Bokeh layout that will be passed to Flask
    run_optimizer_tab_layout = row(controls_col, results_col)

    # --- Callbacks Assignment (after all widgets are defined) ---
    # These connect Python functions to Bokeh widget events.
    objective_select.on_change('value', set_optimization_mode_internal)
    time_hours_input.on_change('value', update_time_hours)
    generate_button.on_click(generate_initial_points)
    calculate_button.on_click(calculate_lutein_for_table)
    suggest_button.on_click(suggest_next_experiment)
    run_suggestion_button.on_click(run_suggestion)
    reset_button.on_click(reset_experiment)

    # CustomJS for client-side interactivity (light and liquid visualization)
    # These JS callbacks will be embedded with the Bokeh components.
    color_change_callback = CustomJS(
        args=dict(slider=i0_range, l1=lights[0], l2=lights[1], l3=lights[2], l4=lights[3], l5=lights[4], l6=lights[5]),
        code="""
        const i0_value = slider.value[1];
        let new_color = 'grey';
        if (i0_value < 100) { new_color = '#4D4C00'; } else if (i0_value < 200) { new_color = '#807E00'; }
        else if (i0_value < 300) { new_color = '#B3B000'; } else if (i0_value < 400) { new_color = '#E6E200'; }
        else if (i0_value < 500) { new_color = '#FFFF00'; } else if (i0_value < 600) { new_color = '#FFFF33'; }
        else if (i0_value < 700) { new_color = '#FFFF66'; } else if (i0_value < 800) { new_color = '#FFFF99'; }
        else if (i0_value < 900) { new_color = '#FFFFCC'; } else { new_color = '#FFFFF0'; }
        const new_html = `<svg height="50" width="50"><circle cx="25" cy="25" r="20" stroke="black" stroke-width="2" fill="${new_color}" /></svg>`;
        const all_lights = [l1, l2, l3, l4, l5, l6];
        all_lights.forEach(light => { light.text = new_html; });
    """)
    i0_range.js_on_change('value', color_change_callback)

    update_initial_liquid_callback = CustomJS(args=dict(cx_slider=cx0_range, cn_slider=cn0_range, liquid_div=spacer), code="""
        const cx0 = cx_slider.value[1];
        const cn0 = cn_slider.value[1];
        const cx_max = cx_slider.end;
        const cn_max = cn_slider.end;
        const green_opacity = Math.min((cx0 / cx_max) * 0.8, 0.8);
        const blue_opacity = Math.min((cn0 / cn_max) * 0.6, 0.6);
        const green_gradient = `linear-gradient(rgba(48, 128, 64, ${green_opacity}), rgba(24, 64, 32, ${green_opacity}))`;
        const blue_gradient = `linear-gradient(rgba(52, 152, 219, ${blue_opacity}), rgba(41, 128, 185, ${blue_opacity}))`;
        const liquid_css = `
            position: absolute; bottom: 0; left: 0; right: 0; height: 95%;
            background: ${green_gradient}, ${blue_gradient};
            border-bottom-left-radius: 18px; border-bottom-right-radius: 18px; z-index: -1;
        `;
        liquid_div.text = `<div style="${liquid_css}"></div>`;
    """)
    cx0_range.js_on_change('value', update_initial_liquid_callback)
    cn0_range.js_on_change('value', update_initial_liquid_callback)

    # Initial state setup for Python callbacks and visuals
    # These lines run once when the Flask route is accessed to set initial state.
    # Pass dummy arguments to set_optimization_mode_internal to satisfy its signature for initial call
    set_optimization_mode_internal("value", None, objective_select.value)
    set_ui_state()

    # To ensure initial display of CustomJS-driven visuals (lights and liquid),
    # we manually trigger their updates by briefly modifying and restoring slider values.
    # This forces the js_on_change callbacks to execute with the initial default values.
    temp_i0_range_value = i0_range.value
    i0_range.value = (i0_range.value[0], i0_range.value[1] + 1) # Temporarily change
    i0_range.value = temp_i0_range_value # Restore, triggering JS.

    temp_cx0_range_value = cx0_range.value
    cx0_range.value = (cx0_range.value[0], cx0_range.value[1] + 0.1)
    cx0_range.value = temp_cx0_range_value

    temp_cn0_range_value = cn0_range.value
    cn0_range.value = (cn0_range.value[0], cn0_range.value[1] + 0.1)
    cn0_range.value = temp_cn0_range_value

    return run_optimizer_tab_layout # Return the Bokeh layout to Flask

# This block allows BO_final.py to still be run as a standalone Bokeh server app
# if desired (e.g., `bokeh serve --show BO_final.py`).
# It will NOT execute when imported by Flask.
if __name__ == '__main__':
    from bokeh.io import curdoc
    doc = curdoc()
    # The title for the browser tab
    doc.title = "Lutein Production Optimizer (Standalone)"
    # Get the layout from the function
    main_app_layout = create_bokeh_layout_for_flask()
    # Add it to the current document for Bokeh server
    doc.add_root(main_app_layout)
    # For standalone bokeh serve, you might manually trigger initial JS display
    # if it doesn't automatically. However, for Flask embedding, the Flask side handles it.
    # For Bokeh server, the 'document_ready' event handler could be explicitly added here
    # if not implicitly triggered.