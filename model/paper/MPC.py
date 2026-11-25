"""MPC algorithm."""
import copy
from collections import deque
from datetime import date
import numpy as np
from other.environment.information_protocol import expected_hot_water_l
from other.environment.interface import DataInterface
from .TOU_RBC import RuleBasedController

class MeanNoise:

    def normal(self, mean, std):
        return mean

class ForecastData(DataInterface):

    def get_home_load(self, day, slot):
        return self.load

    def get_pv_generation(self, day, slot):
        return self.pv

    def get_outdoor_temp(self, day, slot):
        return self.temp

    def get_solar_irradiance_norm(self, day, slot):
        return self.solar

    def scenario_rng(self, *args, **kwargs):
        return MeanNoise()

    def get_daily_ev_trip_km(self, day, mean_km, sigma):
        return self.trip_km if day == self.today else mean_km

    def generate_daily_ev_schedule(self, day):
        if date.fromisoformat(day).weekday() >= 5:
            self.ev_schedule[day] = (None, None)
        else:
            params = self.ev_schedule_params
            self.ev_schedule[day] = (2 * params['weekday_arrival_mean_hour'], 2 * params['weekday_departure_mean_hour'])

def forecast(env, state):
    data = object.__new__(ForecastData)
    data.steps_per_day = env.steps_per_day
    data.load, data.pv, data.temp, data.solar = (state[k] for k in ('home_load', 'pv_generation', 'temperature', 'solar_irradiance_norm'))
    data.today = env.current_time
    di = env.data_interface
    for key in ('price_profile', 'tariff_params', 'tou_hourly_values', 'ev_schedule_params', '_tou_price_cache'):
        setattr(data, key, getattr(di, key))
    plan = env.declared_ev_plan
    departure = plan['departure_step']
    data.ev_schedule = {data.today: (plan['arrival_step'], None if departure is None else departure + 1)}
    data.trip_km = plan['trip_kwh'] / env.ev_trip_config['energy_consumption_per_km']
    sim = copy.copy(env)
    for key, value in vars(env).items():
        if isinstance(value, list):
            setattr(sim, key, list(value) if key in ('_current_block_dates', 'episode_schedule') else [])
        elif isinstance(value, deque):
            setattr(sim, key, copy.copy(value))
        elif isinstance(value, dict):
            if key in ('records', 'reward_components'):
                setattr(sim, key, {name: [] for name in value})
            else:
                setattr(sim, key, value.copy())
    sim.data_interface = data
    sim._carbon_df = None
    carbon = env._carbon_intensity_kg_per_kwh_at_now()
    sim._carbon_intensity_kg_per_kwh_at_offset = lambda offset_steps=0: carbon
    sim._episode_mean_carbon_intensity = carbon
    sim._ewh_draw_profile_cache = {}
    sim._ewh_service_context_cache = None
    profile = np.array([expected_hot_water_l(i, sim.steps_per_day, sim.ewh_params['daily_mixed_water_mean_l']) for i in range(sim.steps_per_day)])
    sim._generate_ewh_draw_profile = lambda day: (profile, profile / 7.0)
    sim._ewh_draw_profile_l, sim._ewh_draw_flow_lpm = sim._generate_ewh_draw_profile(data.today)
    return sim

class MPC:

    def __init__(self, env, horizon=4, budget=8):
        if horizon < 2 or budget < 3:
            raise ValueError('MPC requires at least two slots and three candidate sequences')
        self.env = env
        self.horizon, self.budget = (horizon, budget)
        self.previous = None

    def take_action(self, state):
        env = self.env
        masks, _ = env.get_action_mask(state, mask_mode='physical')
        legal = {n: [v for v, ok in zip(opts, masks[n]) if ok] for n, opts in env.action_space.items()}
        rule = RuleBasedController(env).take_action(state)

        def project(action):
            return {n: min(vals, key=lambda x: abs(x - action[n])) for n, vals in legal.items()}
        base = project(rule)
        horizon = min(self.horizon, env.current_episode_length_steps - env._current_block_step)
        rng = np.random.default_rng(1009 * date.fromisoformat(env.current_time).toordinal() + env.current_time_index)
        candidates = [[base] * horizon]
        candidates.append([project(dict(base, ev_power=0, battery_power=0))] * horizon)
        branches = list(legal)
        while len(candidates) < self.budget:
            first, second = (base.copy(), base.copy())
            for name in rng.choice(branches, size=3, replace=False):
                first[name] = float(rng.choice(legal[name]))
                second[name] = float(rng.choice(legal[name]))
            cut = horizon
            candidates.append([first] * cut + [second] * (horizon - cut))
        scores = []
        for plan in candidates:
            sim = forecast(env, state)
            observation, score = (state.copy(), 0.0)
            for request in plan:
                observation, reward, done = sim.step(observation, request)
                score += reward
                if done:
                    break
            scores.append(score)
        if not np.isfinite(scores).all():
            raise ValueError('Non-finite planning reward')
        self.previous = candidates[int(np.argmax(scores))]
        return self.previous[0].copy()
