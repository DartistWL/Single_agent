"""Compute energy, cost, carbon and service metrics."""
import math
import numpy as np

def meter(env, state, nxt, reward):
    c = env._last_reward_components
    demand = sum((c[k] for k in ('base_load_kwh', 'hvac_energy_kwh', 'ewh_energy_kwh', 'wash_energy_kwh', 'ev_charge_input_kwh', 'ess_charge_input_kwh')))
    supply = sum((c[k] for k in ('pv_generation_kwh', 'ev_discharge_output_kwh', 'ess_discharge_output_kwh')))
    residual = demand - supply - c['grid_import_kwh'] + c['grid_export_kwh']
    if not math.isclose(residual, 0.0, abs_tol=1e-07):
        raise ValueError(f'Household electricity ledger fails by {residual} kWh')
    for name, field in (('ess', 'ess_state'), ('ev', 'ev_battery_state')):
        delta = env.charge_efficiency * c[f'{name}_charge_input_kwh'] - c[f'{name}_discharge_output_kwh'] / env.discharge_efficiency
        if name == 'ev':
            delta -= env.current_step_ev_trip_consumption_kwh
        if not math.isclose(nxt[field] - state[field], delta, rel_tol=0.0, abs_tol=1e-07):
            raise ValueError(f'{name} storage balance mismatch')
    bill = c['grid_import_kwh'] * state['electricity_price'] - c['grid_export_kwh'] * state['grid_export_price']
    if not math.isclose(bill, c['net_energy_cost_eur'], rel_tol=0.0, abs_tol=1e-07):
        raise ValueError('Import/export meter does not reconcile with the bill')
    carbon = c['grid_import_kwh'] * env.last_step_carbon_intensity_kg_per_kwh
    if not math.isclose(carbon, env.current_step_emission_kg, abs_tol=1e-09):
        raise ValueError('Carbon accounting does not reconcile')
    arrived = not state['ev_at_home'] and bool(nxt['ev_at_home'])
    planned_trip = state['declared_trip_fraction'] * env.ev_capacity if arrived else 0.0
    return dict(return_total=reward, cost_eur=c['net_energy_cost_eur'], carbon_kg=carbon, import_kwh=c['grid_import_kwh'], export_kwh=c['grid_export_kwh'], hot_water_shortfall_l_c=c['hot_water_shortfall_c'] * c['hot_water_draw_l'], ev_shortfall_kwh=c['ev_departure_shortfall'], ev_departures=c['ev_departure_event'], ev_successes=c['ev_departure_success'], trip_kwh=env.current_step_ev_trip_consumption_kwh, planned_trip_kwh=planned_trip, unserved_trip_kwh=max(0.0, planned_trip - env.current_step_ev_trip_consumption_kwh), ac_discomfort_hours=0.5 * (1.0 - c['ac_comfort_score']), wash_completions=c['wash_completed'], hvac_kwh=c['hvac_energy_kwh'], water_heating_kwh=c['ewh_energy_kwh'], ess_charge_kwh=c['ess_charge_input_kwh'], ess_discharge_kwh=c['ess_discharge_output_kwh'])

def json_scalar(x):
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)
