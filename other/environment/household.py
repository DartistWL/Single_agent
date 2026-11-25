"""Connect household dynamics to measured data."""
import copy
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from .config import CONFIG
from .physics import HomeEnergyManagementEnv
from .interface import DataInterface
from .weather import readings
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/storenet_ireland_2020'
CARBON = ROOT / 'data/carbon_2020/eirgrid_roi_co2_2020_30min.csv'
STEPS_PER_DAY = 48
SEED = 11

@lru_cache(maxsize=1)
def carbon_days():
    df = pd.read_csv(CARBON)
    stamps = pd.DatetimeIndex(pd.to_datetime(df.timestamp_utc, utc=True))
    values = df.carbon_intensity_gco2_per_kwh.to_numpy(dtype=float) / 1000.0
    if stamps.has_duplicates or not np.isfinite(values).all() or min(values) < 0:
        raise ValueError('Invalid carbon measurements')
    result = {}
    for d in pd.date_range('2020-01-01', '2020-12-31'):
        slots = pd.date_range(d, periods=STEPS_PER_DAY, freq='30min').tz_localize('Europe/Dublin', ambiguous=False, nonexistent='shift_forward').tz_convert('UTC')
        locations = stamps.get_indexer(slots)
        if len(set(slots)) == STEPS_PER_DAY and (locations >= 0).all() and (slots[-1] - slots[0] == pd.Timedelta(minutes=30 * (STEPS_PER_DAY - 1))):
            result[d.date().isoformat()] = values[locations]
    return result

class MeasuredData(DataInterface):

    def get_outdoor_temp(self, current_time, current_time_index):
        return readings()[str(current_time)[:10], int(current_time_index)]

    def get_home_load(self, current_date, current_time_index):
        return float(self.load_days[str(current_date)[:10]][current_time_index])

    def get_pv_generation(self, current_date, current_time_index):
        return float(self.pv_days[str(current_date)[:10]][current_time_index])

    @classmethod
    def from_storenet_ireland_2020(cls, **kwargs):
        template = measured_template(kwargs['house_id'], kwargs['base_dir'], kwargs['price_profile'], kwargs['steps_per_day'])
        data = copy.copy(template)
        for key, value in vars(data).copy().items():
            if isinstance(value, dict) and key not in ('load_days', 'pv_days'):
                setattr(data, key, value.copy())
        return data

@lru_cache(maxsize=4)
def measured_template(house, directory, price, steps):
    di = super(MeasuredData, MeasuredData).from_storenet_ireland_2020(house, directory, price, steps, tariff_params=CONFIG['environment']['tariff'], ev_schedule_params=CONFIG['environment']['ev_trip'])
    di.load_days, di.pv_days = [dict(zip(frame.index.strftime('%Y-%m-%d'), frame.to_numpy())) for frame in (di.cons_data, di.prod_data)]
    di.weather_source = 'StoreNet weather_2020.csv'
    return di

class Household(HomeEnergyManagementEnv):
    data_class = MeasuredData

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state_space.update(declared_departure=(-1, 1), declared_arrival=(-1, 1), declared_trip_fraction=(0, 1), wash_finished=(0, 1))

    def _scenario_observation_dict(self):
        values = super()._scenario_observation_dict()
        plan = self.declared_ev_plan
        for label, name in [('declared_departure', 'departure_step'), ('declared_arrival', 'arrival_step')]:
            values[label] = -1.0 if plan[name] is None else plan[name] / self.steps_per_day
        values['declared_trip_fraction'] = plan['trip_kwh'] / self.ev_capacity
        values['wash_finished'] = float(self.has_run_in_current_period and self.remaining_runtime == 0)
        return values

    def _carbon_intensity_kg_per_kwh_at_offset(self, offset_steps=0):
        day, slot = self._date_index_with_offset(offset_steps)
        return float(carbon_days()[day][slot])

def environment(schedule, threshold=None):
    env = Household(episode_schedule=list(schedule), storenet_base_dir=str(DATA), carbon_intensity_csv_path=str(CARBON), peak_calibration_bootstrap=True)
    if threshold is not None:
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError('Invalid train-only peak threshold')
        env.peak_threshold_kw = float(threshold)
        env.peak_shaving_enabled = True
        env.peak_calibration_bootstrap = False
        env.peak_threshold_source = 'physical_train_daily_peaks'
        env.peak_calibration_baseline_mean_kw = threshold / env.peak_target_ratio
    env.seed(SEED)
    return env
