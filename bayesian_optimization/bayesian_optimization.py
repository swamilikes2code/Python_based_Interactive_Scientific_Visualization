import numpy as np
from scipy.integrate import solve_ivp
from skopt import Optimizer
from skopt.space import Real
from skopt.sampler import Lhs, Sobol
import threading
import warnings
from functools import partial

from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    Button,
    Select,
    Div,
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
    Spinner,
)

warnings.filterwarnings("ignore", category=UserWarning)

U_M = 0.152
U_D = 5.95e-3
K_N = 30.0e-3
Y_NX = 0.305

K_M = (0.350e-3 * 2) * 1000
K_D = 3.71 * 0.05 / 90

K_NL = 10.0e-3
K_S = 142.8
K_I = 214.2
K_SL = 320.6
K_IL = 480.9
TAU = 0.120
KA = 0.0

VOLUME_L = 500
AREA_M2 = 1.26
EFFICIENCY = 2e-6

PRICE_BIOMASS_PER_G = 0.01
PRICE_NITROGEN_PER_G = 0.40
PRICE_ENERGY_PER_KWH = 0.15
PRICE_LUTEIN_PER_MG = 0.20

THEORETICAL_LUTEIN_PER_BIOMASS = 0.00556 * 1000

TIME_HOURS = 150

optimization_history = []
optimizer = None
previewed_point = None
optimization_mode = "concentration"

best_observed_point = None
best_observed_value = -float('inf')

converged = False

optimization_stagnation_count = 0
STAGNATION_RESET_THRESHOLD = 15
MAX_RESTARTS = 3
current_restarts_count = 0

MIN_OPTIMIZATION_STEPS = 5
PREDICTED_TOLERANCE_THRESHOLD = 0.5
X_CANDIDATES = 10


def pbr(t, C, F_in, C_N_in, I0):
    C_X, C_N, C_L = C

    C_X = max(C_X, 1e-12)
    C_N = max(C_N, 1e-12)
    C_L = max(C_L, 1e-12)

    I = 2 * I0 * (np.exp(-(TAU * 0.01 * 1000 * C_X)))

    Iscaling_u = I / (I + K_S + I ** 2 / K_I)
    Iscaling_k = I / (I + K_SL + I ** 2 / K_IL)

    u0 = U_M * Iscaling_u
    dCxdt = u0 * C_N * C_X / (C_N + K_N) - U_D * C_X

    dCndt = -Y_NX * u0 * C_N * C_X / (C_N + K_N) + F_in * C_N_in

    k0 = K_M * Iscaling_k

    dCldt = k0 * C_N * C_X / (C_N + K_NL) - K_D * C_L * C_X

    return np.array([dCxdt, dCndt, dCldt])


def calculate_biomass_cost(C_x0):
    return C_x0 * VOLUME_L * PRICE_BIOMASS_PER_G


def calculate_nitrogen_cost(C_N0, C_N_in, F_in, time_hours):
    initial_cost = C_N0 * VOLUME_L * PRICE_NITROGEN_PER_G
    feed_cost = F_in * C_N_in * time_hours * PRICE_NITROGEN_PER_G
    return initial_cost + feed_cost


def calculate_energy_cost(I0, time_hours):
    I0_mol = I0 * 1e-6
    energy_J_per_s = I0_mol * AREA_M2 / EFFICIENCY
    energy_kWh_per_h = energy_J_per_s * 3600 / 3.6e6
    return energy_kWh_per_h * time_hours * PRICE_ENERGY_PER_KWH


def calculate_lutein_profit(lutein_concentration_mg_per_L):
    return lutein_concentration_mg_per_L * VOLUME_L * PRICE_LUTEIN_PER_MG


def calculate_total_cost_and_profit(params, lutein_concentration_mg_per_L, time_hours):
    C_x0, C_N0, F_in, C_N_in, I0 = params

    biomass_cost = calculate_biomass_cost(C_x0)
    nitrogen_cost = calculate_nitrogen_cost(C_N0, C_N_in, F_in, time_hours)
    energy_cost = calculate_energy_cost(I0, time_hours)
    total_cost = biomass_cost + nitrogen_cost + energy_cost

    lutein_profit = calculate_lutein_profit(lutein_concentration_mg_per_L)

    J = lutein_profit - total_cost

    return {
        'biomass_cost': biomass_cost,
        'nitrogen_cost': nitrogen_cost,
        'energy_cost': energy_cost,
        'total_cost': total_cost,
        'lutein_profit': lutein_profit,
        'revenue_J': J
    }


def calculate_lutein_yield(C_x_final_g_per_L, C_L_final_mg_per_L):
    if C_x_final_g_per_L <= 0: return 0.0

    theoretical_lutein_from_biomass_mg_per_L = THEORETICAL_LUTEIN_PER_BIOMASS * C_x_final_g_per_L

    if theoretical_lutein_from_biomass_mg_per_L <= 0: return 0.0

    yield_percent = (C_L_final_mg_per_L / theoretical_lutein_from_biomass_mg_per_L) * 100
    return yield_percent


def _evaluate_lutein_model_objective(params):
    global TIME_HOURS

    C_x0, C_N0, F_in, C_N_in, I0 = params

    initial_conditions = [C_x0, C_N0, 0.0]

    sol = solve_ivp(pbr, [0, TIME_HOURS], initial_conditions, t_eval=[TIME_HOURS], method="RK45", args=(F_in, C_N_in, I0))

    final_lutein_mg_per_L = sol.y[2, -1]
    final_biomass_g_per_L = sol.y[0, -1]

    if not np.isfinite(final_lutein_mg_per_L) or final_lutein_mg_per_L <= 1e-9:
        return 1e12

    if optimization_mode == "concentration":
        return -final_lutein_mg_per_L
    elif optimization_mode == "cost":
        cost_analysis = calculate_total_cost_and_profit(params, final_lutein_mg_per_L, TIME_HOURS)
        return -cost_analysis['revenue_J']
    else:
        lutein_yield = calculate_lutein_yield(final_biomass_g_per_L, final_lutein_mg_per_L)
        if not np.isfinite(lutein_yield) or lutein_yield <= 1e-9:
            return 1e12
        return -lutein_yield


def run_final_simulation(params_to_simulate):
    global TIME_HOURS

    C_x0, C_N0, F_in, C_N_in, I0 = params_to_simulate

    initial_conditions = [C_x0, C_N0, 0.0]

    if not all(np.isfinite(val) for val in initial_conditions):
        raise ValueError(f"Non-finite initial conditions provided for simulation: {initial_conditions}")

    sim_data = {}
    try:
        sol = solve_ivp(pbr, [0, TIME_HOURS], initial_conditions, t_eval=np.linspace(0, TIME_HOURS, 300), method="RK45", args=(F_in, C_N_in, I0))

        sim_data = {
            "time": sol.t,
            "C_X": np.maximum(0, sol.y[0]),
            "C_N": np.maximum(0, sol.y[1]),
            "C_L": np.maximum(0, sol.y[2])
        }
    except Exception as e:
        print(f"Simulation error: {e}")
        t_eval = np.linspace(0, TIME_HOURS, 300)
        sim_data = {
            "time": t_eval,
            "C_X": [np.nan] * len(t_eval),
            "C_N": [np.nan] * len(t_eval),
            "C_L": [np.nan] * len(t_eval)
        }
        raise e

    return sim_data

doc = curdoc()
doc.title = "Lutein Production Optimizer"

js_helper_code = """
window.toggleVisibility = (widget_id) => {
    const widget = Bokeh.documents[0].get_model_by_id(widget_id);
    if (widget) {
        widget.visible = !widget.visible;
    }
}
"""
doc.js_on_event('document_ready', CustomJS(code=js_helper_code))

convergence_source = ColumnDataSource(data=dict(iter=[], best_value=[]))
simulation_source = ColumnDataSource(data=dict(time=[], C_X=[], C_N=[], C_L=[]))
experiments_source = ColumnDataSource(data=dict(
    C_x0=[], C_N0=[], F_in=[], C_N_in=[], I0=[],
    Lutein_mg_per_L=[], Total_Cost=[], Lutein_Profit=[], Revenue_J=[],
    Lutein_Yield=[], Biomass_Cost=[], Nitrogen_Cost=[], Energy_Cost=[]
))


def update_status(message):
    status_div.text = message


def update_time_hours(attr, old, new):
    global TIME_HOURS
    TIME_HOURS = new
    update_status(f"Simulation time set to {TIME_HOURS} hours. This will apply to new calculations.")


def set_optimization_mode():
    global optimization_mode, optimizer, best_observed_point, best_observed_value, converged, optimization_stagnation_count, current_restarts_count
    new_mode = objective_select.value

    if new_mode == optimization_mode and optimizer is not None:
        return

    optimization_mode = new_mode
    optimizer = None
    best_observed_point = None
    best_observed_value = -float('inf')
    converged = False
    optimization_stagnation_count = 0
    current_restarts_count = 0

    if optimization_mode == "concentration":
        p_conv.title.text = "Optimizer Convergence - Lutein Concentration (Maximize)"
        p_conv.yaxis.axis_label = "Max Lutein Found (mg/L)"
        for col in cost_columns: col.visible = False
        for col in yield_columns: col.visible = False
        time_hours_input.visible = False
    elif optimization_mode == "cost":
        p_conv.title.text = "Optimizer Convergence - Revenue Optimization (Maximize)"
        p_conv.yaxis.axis_label = "Max Revenue (Profit - Cost) [$]"
        for col in cost_columns: col.visible = True
        for col in yield_columns: col.visible = False
        time_hours_input.visible = True
    else:
        p_conv.title.text = "Optimizer Convergence - Lutein Yield (Maximize)"
        p_conv.yaxis.axis_label = "Max Lutein Yield (%)"
        for col in cost_columns: col.visible = False
        for col in yield_columns: col.visible = True
        time_hours_input.visible = False

    if experiments_source.data['C_x0'] and not any(np.isnan(v) for v in experiments_source.data['Lutein_mg_per_L']):
        process_and_plot_latest_results()
    else:
        update_status("🟢 Ready. Select objective and define parameters, then generate initial points.")

    set_ui_state()


def set_ui_state(lock_all=False):
    if lock_all:
        for w in all_buttons: w.disabled = True
        for w in param_and_settings_widgets: w.disabled = True
        time_hours_input.disabled = True
        return

    has_points = len(experiments_source.data['C_x0']) > 0
    has_uncalculated_points = has_points and any(np.isnan(v) for v in experiments_source.data['Lutein_mg_per_L'])
    has_calculated_points = has_points and not has_uncalculated_points
    is_preview_pending = previewed_point is not None

    for widget in param_and_settings_widgets:
        widget.disabled = has_points

    time_hours_input.disabled = not (optimization_mode == "cost" and not has_points)

    generate_button.disabled = has_points
    reset_button.disabled = not has_points
    calculate_button.disabled = not has_uncalculated_points
    suggest_button.disabled = not has_calculated_points or is_preview_pending
    run_suggestion_button.disabled = not is_preview_pending


def get_current_dimensions():
    try:
        return [
            Real(cx0_range.value[0], cx0_range.value[1], name="C_x0"),
            Real(cn0_range.value[0], cn0_range.value[1], name="C_N0"),
            Real(fin_range.value[0], fin_range.value[1], name="F_in", prior='log-uniform'),
            Real(cnin_range.value[0], cnin_range.value[1], name="C_N_in"),
            Real(i0_range.value[0], i0_range.value[1], name="I0"),
        ]
    except Exception as e:
        doc.add_next_tick_callback(partial(update_status, f"❌ Error creating dimensions: {e}"))
        return None


def reset_experiment():
    global optimization_history, optimizer, previewed_point, TIME_HOURS, best_observed_point, best_observed_value, converged, optimization_stagnation_count, current_restarts_count
    optimization_history.clear()
    optimizer = None
    previewed_point = None
    best_observed_point = None
    best_observed_value = -float('inf')
    converged = False
    optimization_stagnation_count = 0
    current_restarts_count = 0

    experiments_source.data = {k: [] for k in experiments_source.data}
    convergence_source.data = {k: [] for k in convergence_source.data}
    simulation_source.data = {k: [] for k in simulation_source.data}

    suggestion_div.text = ""
    results_div.text = ""

    time_hours_input.value = 150
    TIME_HOURS = 150

    cx0_val, cn0_val = cx0_range.value[1], cn0_range.value[1]
    cx_max, cn_max = cx0_range.end, cn0_range.end
    green_opacity = min((cx0_val / cx_max) * 0.8, 0.8)
    blue_opacity = min((cn0_val / cn_max) * 0.6, 0.6)
    green_gradient = f"linear-gradient(rgba(48, 128, 64, {green_opacity}), rgba(24, 64, 32, {green_opacity}))"
    blue_gradient = f"linear-gradient(rgba(52, 152, 219, {blue_opacity}), rgba(41, 128, 185, {blue_opacity}))";
    liquid_css = f"""
        position: absolute; bottom: 0; left: 0; right: 0; height: 95%;
        background: {green_gradient}, {blue_gradient};
        border-bottom-left-radius: 18px; border-bottom-right-radius: 18px; z-index: -1;
    """
    spacer.text = f'<div style="{liquid_css}"></div>'

    update_status("🟢 Ready. Define parameters and generate initial points.")
    set_optimization_mode()
    set_ui_state()


def generate_initial_points():
    doc.add_next_tick_callback(lambda: update_status("🔄 Generating initial points..."))
    doc.add_next_tick_callback(lambda: set_ui_state(lock_all=True))

    def worker():
        dims = get_current_dimensions()
        if dims is None:
            doc.add_next_tick_callback(set_ui_state)
            return

        n_initial = n_initial_input.value
        sampler_choice = sampler_select.value

        try:
            seed = np.random.randint(1000)

            if sampler_choice == 'LHS':
                sampler = Lhs(lhs_type="centered", criterion="maximin")
                x0 = sampler.generate(dims, n_samples=n_initial, random_state=seed)
            elif sampler_choice == 'Sobol':
                sampler = Sobol()
                x0 = sampler.generate(dims, n_samples=n_initial, random_state=seed)
            else:
                rng_state = np.random.RandomState(seed)
                x0 = [[d.rvs(random_state=rng_state)[0] for d in dims] for _ in range(n_initial)]

            new_data = {name.name: [point[i] for point in x0] for i, name in enumerate(dims)}
            new_data.update({
                'Lutein_mg_per_L': [np.nan] * n_initial,
                'Total_Cost': [np.nan] * n_initial,
                'Lutein_Profit': [np.nan] * n_initial,
                'Revenue_J': [np.nan] * n_initial,
                'Lutein_Yield': [np.nan] * n_initial,
                'Biomass_Cost': [np.nan] * n_initial,
                'Nitrogen_Cost': [np.nan] * n_initial,
                'Energy_Cost': [np.nan] * n_initial
            })

            def callback():
                experiments_source.data = new_data
                update_status("🟢 Generated initial points. Ready to calculate.")
                set_ui_state()

            doc.add_next_tick_callback(callback)

        except Exception as e:
            doc.add_next_tick_callback(partial(update_status, f"❌ Error generating points: {e}"))
            doc.add_next_tick_callback(set_ui_state)

    threading.Thread(target=worker).start()


def calculate_lutein_for_table():
    doc.add_next_tick_callback(lambda: update_status("🔄 Calculating Lutein and costs for initial points..."))
    doc.add_next_tick_callback(lambda: set_ui_state(lock_all=True))

    def worker():
        global optimization_history, best_observed_point, best_observed_value
        try:
            nan_indices = [i for i, v in enumerate(experiments_source.data['Lutein_mg_per_L']) if np.isnan(v)]
            if not nan_indices:
                doc.add_next_tick_callback(lambda: update_status("🟢 All points already calculated."))
                doc.add_next_tick_callback(set_ui_state)
                return

            points_to_calc_with_idx = []
            for i in nan_indices:
                params = [experiments_source.data[name][i] for name in ['C_x0', 'C_N0', 'F_in', 'C_N_in', 'I0']]
                points_to_calc_with_idx.append((i, params))

            results = []
            new_optimization_history_entries = []

            for i_table, p in points_to_calc_with_idx:
                doc.add_next_tick_callback(
                    lambda i_t=i_table: update_status(f"🔄 Calculating point {i_t + 1}/{len(experiments_source.data['C_x0'])}..."))

                obj_val_internal = _evaluate_lutein_model_objective(p)

                sim_results_dict = run_final_simulation(p)
                lutein_conc_mg_per_L = sim_results_dict['C_L'][-1]
                biomass_conc_g_per_L = sim_results_dict['C_X'][-1]

                cost_analysis = calculate_total_cost_and_profit(p, lutein_conc_mg_per_L, TIME_HOURS)
                lutein_yield = calculate_lutein_yield(biomass_conc_g_per_L, lutein_conc_mg_per_L)

                results.append({
                    'index': i_table,
                    'lutein_mg_per_L': lutein_conc_mg_per_L,
                    'total_cost': cost_analysis['total_cost'],
                    'lutein_profit': cost_analysis['lutein_profit'],
                    'revenue_j': cost_analysis['revenue_J'],
                    'lutein_yield': lutein_yield,
                    'biomass_cost': cost_analysis['biomass_cost'],
                    'nitrogen_cost': cost_analysis['nitrogen_cost'],
                    'energy_cost': cost_analysis['energy_cost']
                })

                if np.isfinite(obj_val_internal) and not any(np.array_equal(p, item[0]) for item in optimization_history):
                    new_optimization_history_entries.append([p, obj_val_internal])

                if optimization_mode == "concentration":
                    current_actual_value = lutein_conc_mg_per_L
                elif optimization_mode == "cost":
                    current_actual_value = cost_analysis['revenue_J']
                else:
                    current_actual_value = lutein_yield

                if current_actual_value > best_observed_value + 1e-6:
                    best_observed_value = current_actual_value
                    best_observed_point = p

            def callback():
                current_data = experiments_source.data.copy()
                for res in results:
                    idx = res['index']
                    current_data['Lutein_mg_per_L'][idx] = res['lutein_mg_per_L']
                    current_data['Total_Cost'][idx] = res['total_cost']
                    current_data['Lutein_Profit'][idx] = res['lutein_profit']
                    current_data['Revenue_J'][idx] = res['revenue_j']
                    current_data['Lutein_Yield'][idx] = res['lutein_yield']
                    current_data['Biomass_Cost'][idx] = res['biomass_cost']
                    current_data['Nitrogen_Cost'][idx] = res['nitrogen_cost']
                    current_data['Energy_Cost'][idx] = res['energy_cost']

                experiments_source.data = current_data
                optimization_history.extend(new_optimization_history_entries)

                update_status("✅ Calculation complete. Ready to get a suggestion.")
                process_and_plot_latest_results()
                set_ui_state()

            doc.add_next_tick_callback(callback)

        except Exception as e:
            error_message = f"❌ Error during calculation: {e}"
            doc.add_next_tick_callback(partial(update_status, error_message))
            doc.add_next_tick_callback(set_ui_state)

    threading.Thread(target=worker).start()


def _ensure_optimizer_is_ready():
    global optimizer, optimization_stagnation_count, current_restarts_count

    dims = get_current_dimensions()
    if dims is None:
        return False

    valid_history_x = []
    valid_history_y = []
    for x_point, y_val_internal in optimization_history:
        if np.isfinite(y_val_internal):
            valid_history_x.append(x_point)
            valid_history_y.append(y_val_internal)

    min_skopt_initial_points = max(5, 2 * len(dims), n_initial_input.value)

    if optimizer is None or (len(optimizer.Xi) != len(valid_history_x)) or \
       (optimization_stagnation_count >= STAGNATION_RESET_THRESHOLD and current_restarts_count < MAX_RESTARTS):
        
        if optimizer is not None and optimization_stagnation_count >= STAGNATION_RESET_THRESHOLD:
            current_restarts_count += 1
            optimization_stagnation_count = 0
            doc.add_next_tick_callback(partial(update_status, f"🔄 Performing warm restart {current_restarts_count}/{MAX_RESTARTS}..."))

        seed = np.random.randint(1000)
        
        optimizer = Optimizer(
            dimensions=dims,
            base_estimator=surrogate_select.value,
            acq_func=acq_func_select.value,
            n_initial_points=max(1, min_skopt_initial_points - len(valid_history_x)),
            random_state=seed
        )

        existing_in_optimizer_tuples = set(tuple(x) for x in optimizer.Xi)
        points_to_tell_x = []
        points_to_tell_y = []
        for x_point, y_val in zip(valid_history_x, valid_history_y):
            if tuple(x_point) not in existing_in_optimizer_tuples:
                points_to_tell_x.append(x_point)
                points_to_tell_y.append(y_val)

        if points_to_tell_x:
            optimizer.tell(points_to_tell_x, points_to_tell_y)

    return True


def suggest_next_experiment():
    global previewed_point, best_observed_point, best_observed_value, converged, optimization_stagnation_count, current_restarts_count

    doc.add_next_tick_callback(lambda: update_status("🔄 Getting next suggestion preview..."))
    doc.add_next_tick_callback(lambda: set_ui_state(lock_all=True))

    def worker():
        global previewed_point, best_observed_point, best_observed_value, converged, optimization_stagnation_count, current_restarts_count

        try:
            final_suggested_point = None
            converged_note = "" # Initialize here, will be updated below

            current_opt_steps = len(optimization_history) - n_initial_input.value

            # Determine true convergence: exhausted restarts AND hit stagnation threshold
            if current_restarts_count >= MAX_RESTARTS and optimization_stagnation_count >= STAGNATION_RESET_THRESHOLD:
                converged = True
                final_suggested_point = best_observed_point
                converged_note = "<br/><b>✅ Global Maximum Reached and Validated. Subsequent suggestions will be the same.</b>"
            # Otherwise, proceed with optimization steps or warm restart logic
            else:
                if not _ensure_optimizer_is_ready():
                    doc.add_next_tick_callback(partial(update_status,
                                                       "❌ Optimizer not ready or no initial data to train. Please calculate initial points first."))
                    doc.add_next_tick_callback(set_ui_state)
                    return

                # Phase 1: Mandatory initial exploration steps OR Warm Restart initial steps
                if current_opt_steps < MIN_OPTIMIZATION_STEPS or \
                   (optimization_stagnation_count >= STAGNATION_RESET_THRESHOLD and current_restarts_count < MAX_RESTARTS):
                    
                    final_suggested_point = optimizer.ask()
                    doc.add_next_tick_callback(partial(update_status, f"🔄 Performing exploration step (Restart Phase {current_restarts_count + 1}/{MAX_RESTARTS})..."))

                # Phase 2: Intelligent Exploitation / Filtering (after mandatory exploration)
                else:
                    candidate_points = optimizer.ask(n_points=X_CANDIDATES)
                    
                    chosen_candidate_based_on_prediction = None
                    
                    if best_observed_point is not None and best_observed_value > -float('inf') and optimizer.models:
                        model = optimizer.models[-1]
                        
                        scored_candidates = []
                        for candidate_p in candidate_points:
                            X_transformed = optimizer.space.transform([candidate_p])
                            if hasattr(model, 'predict') and 'return_std' in model.predict.__code__.co_varnames:
                                predicted_mean_internal_for_candidate = model.predict(X_transformed)[0]
                            else:
                                predictions = np.array([tree.predict(X_transformed)[0] for tree in model.estimators_])
                                predicted_mean_internal_for_candidate = np.mean(predictions)
                            
                            predicted_value = -predicted_mean_internal_for_candidate
                            scored_candidates.append((predicted_value, candidate_p))
                        
                        scored_candidates.sort(key=lambda x: x[0], reverse=True)

                        for predicted_val, candidate_p in scored_candidates:
                            if predicted_val >= (best_observed_value - PREDICTED_TOLERANCE_THRESHOLD):
                                chosen_candidate_based_on_prediction = candidate_p
                                break
                        
                        if chosen_candidate_based_on_prediction is None:
                            final_suggested_point = best_observed_point
                            optimization_stagnation_count += 1
                            doc.add_next_tick_callback(partial(update_status, f"ℹ️ All {X_CANDIDATES} optimizer suggestions predicted significantly lower than Best. Suggesting best_observed_point. Stagnation: {optimization_stagnation_count}"))
                        else:
                            final_suggested_point = chosen_candidate_based_on_prediction
                            optimization_stagnation_count = 0 

                    else: # No meaningful best_observed_value yet, or model not ready
                        if optimizer.models and candidate_points:
                            model = optimizer.models[-1]
                            best_of_optimizer_candidates_val = -float('inf')
                            best_of_optimizer_candidates_point = None
                            for candidate_p in candidate_points:
                                 X_transformed = optimizer.space.transform([candidate_p])
                                 if hasattr(model, 'predict') and 'return_std' in model.predict.__code__.co_varnames:
                                    predicted_mean_internal_for_candidate = model.predict(X_transformed)[0]
                                 else:
                                    predictions = np.array([tree.predict(X_transformed)[0] for tree in model.estimators_])
                                    predicted_mean_internal_for_candidate = np.mean(predictions)
                                 
                                 current_predicted_value = -predicted_mean_internal_for_candidate
                                 if current_predicted_value > best_of_optimizer_candidates_val:
                                     highest_predicted_value = current_predicted_value
                                     best_of_optimizer_candidates_point = candidate_p
                            final_suggested_point = best_of_optimizer_candidates_point
                        else:
                            final_suggested_point = candidate_points[0] if candidate_points else None

            # Safety fallback for final_suggested_point
            if final_suggested_point is None and best_observed_point is not None:
                final_suggested_point = best_observed_point
            elif final_suggested_point is None and len(optimization_history) > 0:
                final_suggested_point = optimization_history[-1][0]
            elif final_suggested_point is None:
                 doc.add_next_tick_callback(partial(update_status, "❌ No points to suggest, try generating initial points."))
                 doc.add_next_tick_callback(set_ui_state)
                 return


            # --- DISPLAY PREDICTIONS: Ensure accuracy for known points ---
            predicted_objective_value_for_display = 0.0
            std = 0.0

            # If the final suggested point is our known best_observed_point (or we are globally converged)
            # This is the primary condition for displaying the exact value
            if (converged and best_observed_point is not None) or \
               (best_observed_point is not None and np.allclose(final_suggested_point, best_observed_point, atol=1e-5)):
                predicted_objective_value_for_display = best_observed_value
                std = 0.0 # Zero uncertainty for a known empirical point
            elif optimizer and optimizer.models and final_suggested_point is not None:
                # Otherwise, rely on the model's prediction for this point
                X_transformed = optimizer.space.transform([final_suggested_point])
                model = optimizer.models[-1]

                if hasattr(model, 'predict') and 'return_std' in model.predict.__code__.co_varnames:
                    mean_arr, std_arr = model.predict(X_transformed, return_std=True)
                    predicted_objective_value_for_display, std = -mean_arr[0], std_arr[0]
                elif hasattr(model, 'estimators_'):
                    predictions = np.array([tree.predict(X_transformed)[0] for tree in model.estimators_])
                    predicted_objective_value_for_display, std = -np.mean(predictions), np.std(predictions)
                else:
                    predicted_objective_value_for_display = -model.predict(X_transformed)[0]
                    std = 0.0

            # Formulate the metric text
            if optimization_mode == "concentration":
                metric_text = f"<b>Predicted Lutein: {predicted_objective_value_for_display:.4f} &plusmn; {std:.4f} mg/L</b>"
            elif optimization_mode == "cost":
                metric_text = f"<b>Predicted Revenue (J): ${predicted_objective_value_for_display:.4f} &plusmn; ${std:.4f}</b>"
            else:
                metric_text = f"<b>Predicted Lutein Yield: {predicted_objective_value_for_display:.4f} &plusmn; {std:.4f} %</b>"

            previewed_point = final_suggested_point

            def callback():
                names = [d.name for d in get_current_dimensions()]
                suggestion_html = "<h5>Suggested Next Experiment:</h5>"
                suggestion_html += metric_text + converged_note + "<ul>" # converged_note is now correctly appended here
                if previewed_point is not None:
                    for name, val in zip(names, previewed_point):
                        suggestion_html += f"<li><b>{name}:</b> {val:.4f}</li>"
                suggestion_html += "</ul>"
                suggestion_div.text = suggestion_html
                update_status("💡 Suggestion received. You can now run this specific experiment.")
                set_ui_state()

            doc.add_next_tick_callback(callback)
        except Exception as e:
            error_message = f"❌ Error getting preview: {e}"
            doc.add_next_tick_callback(partial(update_status, error_message))
            doc.add_next_tick_callback(set_ui_state)

    threading.Thread(target=worker).start()


def run_suggestion():
    global previewed_point, optimization_history, best_observed_point, best_observed_value, converged, optimization_stagnation_count, current_restarts_count

    if previewed_point is None:
        update_status("No suggestion to run. Please click 'Suggest Next Experiment' first.")
        return
    doc.add_next_tick_callback(lambda: update_status("🔄 Running suggested experiment..."))
    doc.add_next_tick_callback(lambda: set_ui_state(lock_all=True))

    def worker():
        global previewed_point, optimization_history, optimizer, best_observed_point, best_observed_value, converged, optimization_stagnation_count, current_restarts_count

        try:
            point_to_run = previewed_point

            obj_val_internal = _evaluate_lutein_model_objective(point_to_run)

            if not _ensure_optimizer_is_ready():
                doc.add_next_tick_callback(partial(update_status, "❌ Optimizer not ready."))
                doc.add_next_tick_callback(set_ui_state)
                return

            already_in_history = any(
                np.allclose(point_to_run, item[0], atol=1e-5) and np.isclose(obj_val_internal, item[1], atol=1e-5)
                for item in optimization_history
            )

            if not already_in_history:
                optimizer.tell(point_to_run, obj_val_internal)
                optimization_history.append([point_to_run, obj_val_internal])
            else:
                doc.add_next_tick_callback(partial(update_status,
                                                   "ℹ️ This exact experiment has already been run and added to the model. No new data added to optimizer."))

            sim_results_dict = run_final_simulation(point_to_run)
            lutein_val_mg_per_L = sim_results_dict['C_L'][-1]
            biomass_val_g_per_L = sim_results_dict['C_X'][-1]

            cost_analysis = calculate_total_cost_and_profit(point_to_run, lutein_val_mg_per_L, TIME_HOURS)
            lutein_yield_val = calculate_lutein_yield(biomass_val_g_per_L, lutein_val_mg_per_L)

            if optimization_mode == "concentration":
                current_actual_value = lutein_val_mg_per_L
            elif optimization_mode == "cost":
                current_actual_value = cost_analysis['revenue_J']
            else:
                current_actual_value = lutein_yield_val

            if current_actual_value > best_observed_value + 1e-6:
                best_observed_value = current_actual_value
                best_observed_point = point_to_run
                optimization_stagnation_count = 0
                converged = False
            else:
                optimization_stagnation_count += 1

            def callback():
                global previewed_point

                new_point_data = {
                    'C_x0': [point_to_run[0]],
                    'C_N0': [point_to_run[1]],
                    'F_in': [point_to_run[2]],
                    'C_N_in': [point_to_run[3]],
                    'I0': [point_to_run[4]],
                    'Lutein_mg_per_L': [lutein_val_mg_per_L],
                    'Total_Cost': [cost_analysis['total_cost']],
                    'Lutein_Profit': [cost_analysis['lutein_profit']],
                    'Revenue_J': [cost_analysis['revenue_J']],
                    'Lutein_Yield': [lutein_yield_val],
                    'Biomass_Cost': [cost_analysis['biomass_cost']],
                    'Nitrogen_Cost': [cost_analysis['nitrogen_cost']],
                    'Energy_Cost': [cost_analysis['energy_cost']]
                }
                experiments_source.stream(new_point_data)

                opt_step_number = len(optimization_history) - n_initial_input.value

                if optimization_mode == "concentration":
                    update_status(
                        f"✅ Ran suggested experiment as Optimization Step {opt_step_number}. Lutein: {lutein_val_mg_per_L:.4f} mg/L")
                elif optimization_mode == "cost":
                    update_status(
                        f"✅ Ran suggested experiment as Optimization Step {opt_step_number}. Revenue: ${cost_analysis['revenue_J']:.4f}")
                else:
                    update_status(
                        f"✅ Ran suggested experiment as Optimization Step {opt_step_number}. Lutein Yield: {lutein_yield_val:.4f} %")

                previewed_point = None
                suggestion_div.text = ""
                process_and_plot_latest_results()
                set_ui_state()

            doc.add_next_tick_callback(callback)
        except Exception as e:
            error_message = f"❌ Error running suggestion: {e}"
            doc.add_next_tick_callback(partial(update_status, error_message))
            doc.add_next_tick_callback(set_ui_state)

    threading.Thread(target=worker).start()


def process_and_plot_latest_results():
    global best_observed_point, best_observed_value

    if not optimization_history:
        results_div.text = ""
        cx0_val, cn0_val = cx0_range.value[1], cn0_range.value[1]
        cx_max, cn_max = cx0_range.end, cn0_range.end
        green_opacity = min((cx0_val / cx_max) * 0.8, 0.8)
        blue_opacity = min((cn0_val / cn_max) * 0.6, 0.6)
        green_gradient = f"linear-gradient(rgba(48, 128, 64, {green_opacity}), rgba(24, 64, 32, {green_opacity}))"
        blue_gradient = f"linear-gradient(rgba(52, 152, 219, {blue_opacity}), rgba(41, 128, 185, {blue_opacity}))";
        liquid_css = f"""
            position: absolute; bottom: 0; left: 0; right: 0; height: 95%;
            background: {green_gradient}, {blue_gradient};
            border-bottom-left-radius: 18px; border-bottom-right-radius: 18px; z-index: -1;
        """
        spacer.text = f'<div style="{liquid_css}"></div>'
        doc.add_next_tick_callback(partial(simulation_source.data.update, {'time': [], 'C_X': [], 'C_N': [], 'C_L': []}))
        return

    valid_optimization_history = [(x, y) for x, y in optimization_history if np.isfinite(y)]
    if not valid_optimization_history:
        results_div.text = "<h3>No valid results yet to determine best point.</h3>"
        doc.add_next_tick_callback(partial(simulation_source.data.update, {'time': [], 'C_X': [], 'C_N': [], 'C_L': []}))
        return

    if best_observed_point is None:
        temp_best_item = max([(p, -obj_val) for p, obj_val in valid_optimization_history], key=lambda item: item[1])
        best_observed_point = temp_best_item[0]
        best_observed_value = temp_best_item[1]

    if best_observed_point is None:
        results_div.text = "<h3>No valid results yet to determine best point.</h3>"
        doc.add_next_tick_callback(partial(simulation_source.data.update, {'time': [], 'C_X': [], 'C_N': [], 'C_L': []}))
        return

    sim_data = run_final_simulation(best_observed_point)
    doc.add_next_tick_callback(partial(simulation_source.data.update, sim_data))

    max_lutein_mg_per_L = sim_data['C_L'][-1]
    final_biomass_g_per_L = sim_data['C_X'][-1]
    final_nitrate_g_per_L = sim_data['C_N'][-1]

    cost_analysis = calculate_total_cost_and_profit(best_observed_point, max_lutein_mg_per_L, TIME_HOURS)
    best_yield = calculate_lutein_yield(final_biomass_g_per_L, max_lutein_mg_per_L)

    optimal_params = {dim.name: val for dim, val in zip(get_current_dimensions(), best_observed_point)}

    results_html = f"<h3>Overall Best Result So Far</h3>"
    if optimization_mode == "concentration":
        results_html += f"<b>Maximum Lutein Found:</b> {max_lutein_mg_per_L:.6f} mg/L<br/>"
    elif optimization_mode == "cost":
        results_html += f"<b>Max Revenue (J):</b> ${cost_analysis['revenue_J']:.6f}<br/>"
    elif optimization_mode == "yield":
        results_html += f"<b>Maximum Lutein Yield Found:</b> {best_yield:.6f} %<br/>"

    results_html += f"<b>Lutein Concentration:</b> {max_lutein_mg_per_L:.6f} mg/L<br/>"
    results_html += f"<b>Total Cost:</b> ${cost_analysis['total_cost']:.6f}<br/>"
    results_html += f"<b>Biomass Cost:</b> ${cost_analysis['biomass_cost']:.6f}<br/>"
    results_html += f"<b>Nitrogen Cost:</b> ${cost_analysis['nitrogen_cost']:.6f}<br/>"
    results_html += f"<b>Energy Cost:</b> ${cost_analysis['energy_cost']:.6f}<br/>"
    results_html += f"<b>Lutein Profit (Revenue from Lutein):</b> ${cost_analysis['lutein_profit']:.6f}<br/>"


    results_html += "<b>Corresponding Parameters:</b><ul>"
    for param, value in optimal_params.items():
        results_html += f"<li><b>{param}:</b> {value:.6f}</li>"
    results_html += "</ul>"
    results_div.text = results_html

    spacer.styles = GLASS_STYLE
    orange_opacity = np.clip(max_lutein_mg_per_L / 18.0, 0.0, 1.0) * 0.9
    green_opacity = np.clip(final_biomass_g_per_L / 5.0, 0.0, 1.0) * 0.6
    blue_opacity = np.clip(final_nitrate_g_per_L / 2.0, 0.0, 1.0) * 0.5

    orange_gradient = f"linear-gradient(rgba(255, 140, 0, {orange_opacity}), rgba(210, 105, 30, {orange_opacity}))"
    green_gradient = f"linear-gradient(rgba(48, 128, 64, {green_opacity}), rgba(24, 64, 32, {green_opacity}))"
    blue_gradient = f"linear-gradient(rgba(52, 152, 219, {blue_opacity}), rgba(41, 128, 185, {blue_opacity}))";

    liquid_css = f"""
        position: absolute; bottom: 0; left: 0; right: 0; height: 95%;
        background: {orange_gradient}, {green_gradient}, {blue_gradient};
        border-bottom-left-radius: 18px; border-bottom-right-radius: 18px; z-index: -1;
    """
    spacer.text = f'<div style="{liquid_css}"></div>'

    update_convergence_plot_from_history()


def update_convergence_plot_from_history():
    num_initial = n_initial_input.value

    valid_optimization_history = [(x, y) for x, y in optimization_history if np.isfinite(y)]

    if not valid_optimization_history:
        convergence_source.data = dict(iter=[], best_value=[])
        return

    current_best_display = -np.inf

    initial_segment = valid_optimization_history[:min(num_initial, len(valid_optimization_history))]
    if initial_segment:
        initial_best_internal_objective = min(p[1] for p in initial_segment)
        current_best_display = -initial_best_internal_objective

    iters = []
    best_values_so_far = []

    for i, (x_point, y_val_internal) in enumerate(optimization_history):
        if np.isfinite(y_val_internal):
            display_value = -y_val_internal

            if display_value > current_best_display:
                current_best_display = display_value

            if i >= num_initial:
                iters.append(i - num_initial + 1)
                best_values_so_far.append(current_best_display)

    convergence_source.data = {'iter': iters, 'best_value': best_values_so_far}
    p_conv.xaxis.ticker = FixedTicker(ticks=iters)
    if iters:
        p_conv.x_range.end = iters[-1] + 0.5
    else:
        p_conv.x_range.end = 1.0


def create_slider_with_info(label_text, tooltip_text, slider_widget):
    help_paragraph = Paragraph(
        text=tooltip_text,
        styles={
            'font-size': '12px',
            'color': '#555',
            'margin-left': '15px',
            'margin-top': '5px',
            'margin-bottom': '5px',
            'border-left': '3px solid #ccc',
            'padding-left': '10px',
            'background-color': '#f9f9f9',
        },
        visible=False,
        width=400
    )
    info_icon_html = f'<span onclick="window.toggleVisibility(\'{help_paragraph.id}\')" style="cursor: pointer; user-select: none;">❓</span>'
    label_div = Div(
        text=f"<b>{label_text}</b> {info_icon_html}",
        styles={'font-size': '13px', 'color': '#444', 'margin-bottom': '-5px'}
    )
    slider_widget.title = ""
    return column(label_div, help_paragraph, slider_widget, styles={'margin-top': '8px'})


title_div = Div(text="<h1>Lutein Production Bayesian Optimizer with Cost Analysis</h1>")
description_p = Paragraph(
    text="""This application uses Bayesian Optimization to find optimal operating conditions for a photobioreactor. You can optimize for maximum lutein concentration (mg/L), maximum profit (Revenue), or maximum yield. Follow the steps to run a virtual experiment.""",
    width=450)

objective_title = Div(text="<h4>0. Select Optimization Objective</h4>")
objective_select = Select(title="Optimization Objective:", value="concentration",
                               options=[("concentration", "Maximize Lutein Concentration (mg/L)"),
                                        ("cost", "Maximize Revenue (Profit - Cost)"),
                                        ("yield", "Maximize Lutein Yield")])
objective_select.on_change('value', lambda attr, old, new: set_optimization_mode())

time_hours_input = Spinner(title="Simulation Time (Hours):", low=1, step=1, value=TIME_HOURS, width=150,
                           visible=False)
time_hours_input.on_change('value', update_time_hours)

param_range_title = Div(text="<h4>1. Define Parameter Search Space</h4>")

hover_info_p = Paragraph(text="""Click ❓ next to a parameter for more details.""",
                         styles={'font-size': '13px', 'color': '#444', 'margin-top': '-15px'})
tooltips = {
    "C_x0": "Initial Biomass Concentration (g/L): This is the starting amount of algae in the photobioreactor (PBR).",
    "C_N0": "Initial Nitrate Concentration (g/L): This is the starting amount of the primary nutrient (nitrate) in the PBR.",
    "F_in": "Inlet Flow Rate (1/hr): The rate at which new, nutrient-rich medium is fed to the PBR.",
    "C_N_in": "Inlet Nitrate Concentration (g/L): The concentration of nitrate in the feed medium.",
    "I0": "Incident Light Intensity (μmol/m²-s): The amount of light energy supplied to the PBR surface."
}

cx0_range = RangeSlider(start=0, end=10, value=(0.2, 2.0), step=0.1)
cn0_range = RangeSlider(start=0, end=10, value=(0.2, 2.0), step=0.1)
fin_range = RangeSlider(start=1e-5, end=1.5e-1, value=(1e-3, 1.5e-2), step=1e-4, format="0.0000")
cnin_range = RangeSlider(start=0, end=50, value=(5.0, 15.0), step=0.5)
i0_range = RangeSlider(start=0, end=1000, value=(100, 200), step=10)

cx0_input = create_slider_with_info("C_x0 Range (g/L)", tooltips["C_x0"], cx0_range)
cn0_input = create_slider_with_info("C_N0 Range (g/L)", tooltips["C_N0"], cn0_range)
fin_input = create_slider_with_info("F_in Range (1/hr)", tooltips["F_in"], fin_range)
cnin_input = create_slider_with_info("C_N_in Range (g/L)", tooltips["C_N_in"], cnin_range)
i0_input = create_slider_with_info("I0 Range (μmol/m²-s)", tooltips["I0"], i0_range)


indicator_panel_title = Div(text="<h4>Photobioreactor State</h4>")
lights = [Div(text="<p>...</p>", width=60, height=60, styles={'text-align': 'center'}) for _ in range(6)]

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

spacer = Div(text="", width=200, height=200)
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
doc.js_on_event('document_ready', color_change_callback, update_initial_liquid_callback)

settings_title = Div(text="<h4>2. Configure Initial Sampling & Model</h4>")
surrogate_select = Select(title="Surrogate Model:", value="GP", options=["GP", "RF", "ET"])
acq_func_select = Select(title="Acquisition Function:", value="EI", options=["EI", "PI", "LCB", "gp_hedge"])
sampler_select = Select(title="Sampling Method:", value="Sobol", options=["LHS", "Sobol", "Random"])
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

generate_button.on_click(generate_initial_points)
calculate_button.on_click(calculate_lutein_for_table)
suggest_button.on_click(suggest_next_experiment)
run_suggestion_button.on_click(run_suggestion)
reset_button.on_click(reset_experiment)

status_div = Div(text="🟢 Ready. Select objective and define parameters, then generate initial points.")
results_div = Div(text="")

columns = [
    TableColumn(field="C_x0", title="C_x0 (g/L)", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="C_N0", title="C_N0 (g/L)", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="F_in", title="F_in (1/hr)", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="C_N_in", title="C_N_in (g/L)", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="I0", title="I0 (umol/m2-s)", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="Lutein_mg_per_L", title="Lutein (mg/L)", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="Total_Cost", title="Total Cost ($)", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="Lutein_Profit", title="Lutein Profit ($)", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="Revenue_J", title="Revenue ($)", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="Lutein_Yield", title="Lutein Yield (%)", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="Biomass_Cost", title="Biomass Cost ($)", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="Nitrogen_Cost", title="Nitrogen Cost ($)", formatter=NumberFormatter(format="0.0000")),
    TableColumn(field="Energy_Cost", title="Energy Cost ($)", formatter=NumberFormatter(format="0.0000"))
]

cost_columns = [col for col in columns if col.field in ['Total_Cost', 'Lutein_Profit', 'Revenue_J', 'Biomass_Cost', 'Nitrogen_Cost', 'Energy_Cost']]
yield_columns = [col for col in columns if col.field in ['Lutein_Yield']]

data_table = DataTable(source=experiments_source, columns=columns, width=1000, height=280, editable=False)

p_conv = figure(height=300, width=800, title="Optimizer Convergence", x_axis_label="Optimization Step",
                y_axis_label="Best Value", y_range=DataRange1d(start=0, range_padding=0.1, range_padding_units='percent'))
p_conv.xaxis.formatter = NumeralTickFormatter(format="0")
p_conv.line(x="iter", y="best_value", source=convergence_source, line_width=2)

p_sim = figure(height=300, width=800, title="Simulation with Best Parameters", x_axis_label="Time (hours)",
               y_axis_label="Biomass & Nitrate Conc. (g/L)", y_range=DataRange1d(start=0))
p_sim.extra_y_ranges = {"lutein_range": DataRange1d(start=0)}
p_sim.add_layout(LinearAxis(y_range_name="lutein_range", axis_label="Lutein Conc. (mg/L)"), 'right')

p_sim.line(x="time", y="C_X", source=simulation_source, color="green", line_width=2, legend_label="Biomass (C_X) [g/L]")
p_sim.line(x="time", y="C_N", source=simulation_source, color="blue", line_width=2, legend_label="Nitrate (C_N) [g/L]")
p_sim.line(x="time", y="C_L", source=simulation_source, color="orange", line_width=3,
           legend_label="Lutein (C_L) [mg/L]", y_range_name="lutein_range")
p_sim.legend.location = "top_left"
p_sim.legend.click_policy = "hide"

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

controls_col = column(
    title_div, description_p,
    objective_title, objective_select, time_hours_input,
    param_range_title,
    hover_info_p,
    cx0_input, cn0_input, fin_input, cnin_input, i0_input,
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

layout = row(controls_col, results_col)
doc.add_root(layout)

set_optimization_mode()
set_ui_state()
