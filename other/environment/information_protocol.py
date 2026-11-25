"""information protocol."""
from functools import lru_cache
from math import erf, sqrt
INFORMATION_PROTOCOL = dict(version='causal_persistence_declared_daily_ev_v1', load_pv_carbon_forecast='current_measurement_persistence', price_forecast='published_tou', hot_water_forecast='fixed_expected_time_of_day_profile', ev_plan='current_day_declared_departure_arrival_trip_energy_zero_error', pv_metadata_input='excluded_full_year_inferred_metadata', carbon_normalization_kg_per_kwh=1.0, carbon_max_age_minutes=30)

@lru_cache(maxsize=256)
def expected_hot_water_l(step, slots, daily_litres):
    showers, meals = (42.0, 12.0)
    components = ((7.25, 0.7, 0.55 * showers), (19.5, 0.7, 0.45 * showers), (7.5, 0.9, 0.3 * meals), (13.0, 0.9, 0.2 * meals), (19.0, 0.9, 0.5 * meals))
    left = (step - 0.5) * 24 / slots if step else -float('inf')
    right = (step + 0.5) * 24 / slots if step < slots - 1 else float('inf')
    return daily_litres * sum((weight * 0.5 * (erf((right - mu) / (sigma * sqrt(2))) - erf((left - mu) / (sigma * sqrt(2)))) for mu, sigma, weight in components)) / (showers + meals)
