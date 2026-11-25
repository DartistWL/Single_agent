"""interface."""
import os
import zlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

class DataInterface:
    MAX_PLAUSIBLE_HOME_LOAD_KW = 20.0
    MAX_PLAUSIBLE_PV_POWER_KW = 10.0

    def __init__(self, cons_file, prod_file, price_profile: str='legacy', steps_per_day: int=48, tou_hourly_values: Optional[List[float]]=None, weather_file: Optional[str]=None, tariff_params: Optional[Dict[str, float]]=None, ev_schedule_params: Optional[Dict[str, float]]=None, house_id: Optional[str]=None):
        self.cons_file = cons_file
        self.prod_file = prod_file
        self.cons_data = pd.read_csv(self.cons_file, parse_dates=['date'], index_col='date')
        self.prod_data = pd.read_csv(self.prod_file, parse_dates=['date'], index_col='date')
        self.weather_file = weather_file if weather_file and os.path.isfile(weather_file) else None
        self.weather_data = None
        if self.weather_file:
            self.weather_data = pd.read_csv(self.weather_file, parse_dates=['date'], index_col='date')
        self.weather_source = os.path.basename(self.weather_file) if self.weather_file else 'irish_climatology_v1'
        self.price_profile = price_profile
        self.steps_per_day = int(steps_per_day)
        if self.steps_per_day <= 0:
            self.steps_per_day = 48
        if len(self.cons_data.columns) != self.steps_per_day:
            raise ValueError(f'消费功率表应包含 {self.steps_per_day} 个日内时隙，实际为 {len(self.cons_data.columns)}')
        if list(self.cons_data.columns) != list(self.prod_data.columns):
            raise ValueError('消费功率表与光伏功率表的日内时隙列不一致')
        if not self.cons_data.index.equals(self.prod_data.index):
            raise ValueError('消费功率表与光伏功率表的日期索引不一致')
        cons_values = np.asarray(self.cons_data, dtype=np.float64)
        prod_values = np.asarray(self.prod_data, dtype=np.float64)
        if not np.all(np.isfinite(cons_values)) or not np.all(np.isfinite(prod_values)):
            raise ValueError('StoreNet 功率表含 NaN 或无穷值')
        if np.min(cons_values) < -1e-08 or np.min(prod_values) < -1e-08:
            raise ValueError('StoreNet 消费或光伏功率含负值')
        if float(np.max(cons_values)) > self.MAX_PLAUSIBLE_HOME_LOAD_KW:
            raise ValueError(f'StoreNet 家庭负荷超过单户 kW 量级上限；请检查是否误将 W 当作 kW：max={float(np.max(cons_values)):.3f} kW')
        if float(np.max(prod_values)) > self.MAX_PLAUSIBLE_PV_POWER_KW:
            raise ValueError(f'StoreNet 光伏功率超过单户 kW 量级上限；请检查数据单位：max={float(np.max(prod_values)):.3f} kW')
        self.tou_hourly_values = tou_hourly_values
        self.tariff_params = dict(tariff_params or {})
        self.ev_schedule_params = dict(ev_schedule_params or {})
        self._tou_price_cache = None
        inferred_house = os.path.basename(str(cons_file)).rsplit('_', 1)[-1].split('.')[0]
        self.house_id = str(house_id or inferred_house).strip().upper()
        pv_values = prod_values
        finite_pv = pv_values[np.isfinite(pv_values)]
        self.pv_capacity_kw = float(np.percentile(finite_pv, 99.5)) if finite_pv.size else 0.0
        self.has_pv = bool(self.pv_capacity_kw > 0.05)
        self.pv_capacity_norm = float(np.clip(self.pv_capacity_kw / 5.0, 0.0, 1.0))
        self.ev_schedule = {}
        self.weather_cache = {}
        self.ev_trip_km_cache = {}
        self.seed_value = 0
        source_id = os.path.basename(str(self.cons_file)).encode('utf-8', errors='replace')
        self._source_seed = int(zlib.crc32(source_id) & 4294967295)
        self.np_random = np.random.RandomState(0)

    def _build_daily_tou_price(self) -> np.ndarray:
        steps = self.steps_per_day
        step_h = 24.0 / steps
        if self.price_profile == 'legacy':
            time_based_prices = {'valley': 0.2, 'flat': 0.5, 'peak': 0.8}
            prices = np.zeros(steps, dtype=np.float32)
            for idx in range(steps):
                if steps == 48:
                    i = idx
                    if 0 <= i < 20 or 44 <= i < 48:
                        prices[idx] = time_based_prices['valley']
                    elif 8 <= i < 34:
                        prices[idx] = time_based_prices['flat']
                    else:
                        prices[idx] = time_based_prices['peak']
                else:
                    t = idx * step_h
                    if 0.0 <= t < 10.0 or 22.0 <= t < 24.0:
                        prices[idx] = time_based_prices['valley']
                    elif 4.0 <= t < 17.0:
                        prices[idx] = time_based_prices['flat']
                    else:
                        prices[idx] = time_based_prices['peak']
            return prices
        if self.price_profile == 'lee2020':
            if self.tou_hourly_values is not None:
                if len(self.tou_hourly_values) != 24:
                    raise ValueError('tou_hourly_values must be a list of length 24 (hourly TOU prices).')
                hourly = np.array(self.tou_hourly_values, dtype=np.float32)
            else:
                hourly = np.array([0.06] * 8 + [0.12] * 1 + [0.14] * 2 + [0.12] * 1 + [0.14] * 4 + [0.12] * 6 + [0.06] * 2, dtype=np.float32)
            factor = steps / 24.0
            prices = np.zeros(steps, dtype=np.float32)
            for idx in range(steps):
                hour_idx = int(np.floor(idx / factor))
                hour_idx = max(0, min(23, hour_idx))
                prices[idx] = hourly[hour_idx]
            return prices.astype(np.float32)
        if self.price_profile == 'ireland_tou_2020':
            offpeak = float(self.tariff_params.get('offpeak_eur_per_kwh', 0.13))
            day = float(self.tariff_params.get('day_eur_per_kwh', 0.18))
            peak = float(self.tariff_params.get('peak_eur_per_kwh', 0.24))
            prices = np.zeros(steps, dtype=np.float32)
            for idx in range(steps):
                hour = idx * step_h
                if hour < 8.0 or hour >= 23.0:
                    prices[idx] = offpeak
                elif 17.0 <= hour < 19.0:
                    prices[idx] = peak
                else:
                    prices[idx] = day
            return prices
        self.price_profile = 'legacy'
        return self._build_daily_tou_price()

    def set_price_profile(self, price_profile: str, tou_hourly_values: Optional[List[float]]=None):
        self.price_profile = price_profile
        self.tou_hourly_values = tou_hourly_values
        self._tou_price_cache = None

    def seed(self, seed: int) -> None:
        self.seed_value = int(seed)
        self.np_random = np.random.RandomState(self.seed_value)
        self.ev_schedule.clear()
        self.weather_cache.clear()
        self.ev_trip_km_cache.clear()

    def _rng_for_date(self, current_date: str, salt: int) -> np.random.RandomState:
        ordinal = datetime.strptime(str(current_date)[:10], '%Y-%m-%d').date().toordinal()
        seed = (self.seed_value * 1000003 + ordinal * 9176 + self._source_seed * 37 + int(salt)) % (2 ** 32 - 1)
        return np.random.RandomState(seed)

    def scenario_rng(self, current_date: str, current_time_index: int=0, salt: int=0) -> np.random.RandomState:
        step_salt = int(salt) + 10007 * int(current_time_index)
        return self._rng_for_date(str(current_date)[:10], step_salt)

    @classmethod
    def from_storenet_ireland_2020(cls, house_id: str, base_dir: str='data/storenet_ireland_2020', price_profile: str='legacy', steps_per_day: int=48, tou_hourly_values: Optional[List[float]]=None, weather_file: Optional[str]=None, tariff_params: Optional[Dict[str, float]]=None, ev_schedule_params: Optional[Dict[str, float]]=None):
        hid = str(house_id).strip().upper()
        cons_file = f'{base_dir}/daily_pivot_cons_2020_{hid}.csv'
        prod_file = f'{base_dir}/daily_pivot_prod_2020_{hid}.csv'
        if weather_file is None:
            weather_candidates = (f'{base_dir}/daily_pivot_temp_2020_{hid}.csv', f'{base_dir}/daily_pivot_weather_2020_{hid}.csv')
            weather_file = next((p for p in weather_candidates if os.path.isfile(p)), None)
        return cls(cons_file, prod_file, price_profile=price_profile, steps_per_day=steps_per_day, tou_hourly_values=tou_hourly_values, weather_file=weather_file, tariff_params=tariff_params, ev_schedule_params=ev_schedule_params, house_id=hid)

    def get_home_load(self, current_date, current_time_index):
        idx = int(current_time_index) % self.steps_per_day
        return float(self.cons_data.loc[current_date, self.cons_data.columns[idx]])

    def get_pv_generation(self, current_date, current_time_index):
        idx = int(current_time_index) % self.steps_per_day
        return float(self.prod_data.loc[current_date, self.prod_data.columns[idx]])

    def get_electricity_price(self, current_date, current_time_index):
        if self._tou_price_cache is None:
            self._tou_price_cache = self._build_daily_tou_price()
        idx = int(current_time_index) % int(self.steps_per_day)
        return float(self._tou_price_cache[idx])

    def get_grid_export_price(self, current_date, current_time_index) -> float:
        import_price = max(0.0, float(self.get_electricity_price(current_date, current_time_index)))
        mode = str(self.tariff_params.get('export_price_mode', 'import_ratio')).strip().lower()
        if mode == 'fixed':
            if 'fixed_export_eur_per_kwh' not in self.tariff_params:
                raise ValueError('fixed export tariff mode requires fixed_export_eur_per_kwh')
            candidate = max(0.0, float(self.tariff_params['fixed_export_eur_per_kwh']))
        elif mode == 'import_ratio':
            ratio = max(0.0, float(self.tariff_params.get('export_import_ratio', 0.9)))
            candidate = import_price * ratio
        else:
            raise ValueError(f'Unsupported export_price_mode={mode!r}')
        cap_ratio = min(0.999999, max(0.0, float(self.tariff_params.get('export_price_cap_ratio', 0.95))))
        return float(min(candidate, import_price * cap_ratio))

    def get_date_time(self, current_date, current_time_index):
        return current_date

    def is_ev_at_home(self, current_date, current_time_index):
        current_date = str(current_date)[:10]
        if current_date not in self.ev_schedule:
            self.generate_daily_ev_schedule(current_date)
        arrival_step, departure_step = self.ev_schedule[current_date]
        if arrival_step is None or departure_step is None:
            return True
        current_step = float(current_time_index)
        return not float(departure_step) <= current_step < float(arrival_step)

    def generate_daily_ev_schedule(self, current_date):
        current_date = str(current_date)[:10]
        current_weekday = datetime.strptime(current_date, '%Y-%m-%d').weekday()
        rng = self._rng_for_date(current_date, salt=11)
        cfg = self.ev_schedule_params
        if current_weekday >= 5:
            trip_probability = float(cfg.get('weekend_trip_probability', 0.45))
            if rng.rand() >= trip_probability:
                self.ev_schedule[current_date] = (None, None)
                return
            departure_hour = float(np.clip(rng.normal(float(cfg.get('weekend_departure_mean_hour', 10.5)), float(cfg.get('weekend_departure_std_hour', 1.0))), 8.0, 13.0))
            arrival_hour = float(np.clip(rng.normal(float(cfg.get('weekend_arrival_mean_hour', 16.0)), float(cfg.get('weekend_arrival_std_hour', 1.5))), departure_hour + 2.0, 21.5))
        else:
            departure_hour = float(np.clip(rng.normal(float(cfg.get('weekday_departure_mean_hour', 7.75)), float(cfg.get('weekday_departure_std_hour', 0.5))), float(cfg.get('weekday_departure_min_hour', 6.5)), float(cfg.get('weekday_departure_max_hour', 9.0))))
            arrival_hour = float(np.clip(rng.normal(float(cfg.get('weekday_arrival_mean_hour', 18.0)), float(cfg.get('weekday_arrival_std_hour', 0.75))), float(cfg.get('weekday_arrival_min_hour', 16.5)), float(cfg.get('weekday_arrival_max_hour', 21.0))))
        factor = float(self.steps_per_day) / 24.0
        self.ev_schedule[current_date] = (arrival_hour * factor, departure_hour * factor)

    def get_ev_schedule_steps(self, current_date: str) -> Tuple[Optional[float], Optional[float]]:
        key = str(current_date)[:10]
        if key not in self.ev_schedule:
            self.generate_daily_ev_schedule(key)
        return self.ev_schedule[key]

    def get_ev_departure_step(self, current_date: str) -> Optional[int]:
        _arrival, departure = self.get_ev_schedule_steps(current_date)
        if departure is None:
            return None
        return int(np.clip(np.ceil(float(departure)) - 1, 0, self.steps_per_day - 1))

    def get_ev_arrival_step(self, current_date: str) -> Optional[int]:
        arrival, _departure = self.get_ev_schedule_steps(current_date)
        if arrival is None:
            return None
        return int(np.clip(np.ceil(float(arrival)), 0, self.steps_per_day - 1))

    def is_ev_departing_soon(self, current_date, current_time_idx):
        return self.get_hours_until_departure(current_date, current_time_idx) <= 2

    def get_hours_until_departure(self, current_date, current_time_idx):
        start_date = datetime.strptime(str(current_date)[:10], '%Y-%m-%d')
        current_step = float(current_time_idx)
        step_h = 24.0 / max(1.0, float(self.steps_per_day))
        for day_offset in range(8):
            date_str = (start_date + timedelta(days=day_offset)).strftime('%Y-%m-%d')
            departure = self.get_ev_departure_step(date_str)
            if departure is None:
                continue
            absolute_step = day_offset * self.steps_per_day + departure
            if absolute_step >= current_step:
                return float(absolute_step - current_step + 1.0) * step_h
        return 7.0 * 24.0

    def get_zone_occupancy(self, current_date: str, current_time_index: int) -> Tuple[float, float]:
        key = str(current_date)[:10]
        hour = float(current_time_index) * 24.0 / max(1.0, float(self.steps_per_day))
        weekday = datetime.strptime(key, '%Y-%m-%d').weekday()
        if weekday < 5:
            arrival, departure = self.get_ev_schedule_steps(key)
            dep_hour = float(departure or 0.0) * 24.0 / self.steps_per_day
            arr_hour = float(arrival or self.steps_per_day) * 24.0 / self.steps_per_day
            if hour < 6.5 or hour >= 22.5:
                return (0.0, 2.0)
            if 6.5 <= hour < dep_hour or arr_hour <= hour < 22.5:
                return (2.0, 0.0)
            return (0.0, 0.0)
        if not self.is_ev_at_home(key, current_time_index):
            return (0.0, 0.0)
        if hour < 8.0 or hour >= 23.0:
            return (0.0, 2.0)
        return (2.0, 0.0)

    def get_solar_irradiance_norm(self, current_date: str, current_time_index: int) -> float:
        key = str(current_date)[:10]
        dt = datetime.strptime(key, '%Y-%m-%d')
        day_of_year = dt.timetuple().tm_yday
        hour = float(current_time_index) * 24.0 / max(1.0, float(self.steps_per_day))
        daylight_h = float(np.clip(12.0 + 4.0 * np.sin(2.0 * np.pi * (day_of_year - 80) / 365.25), 7.5, 16.5))
        sunrise = 12.0 - 0.5 * daylight_h
        sunset = 12.0 + 0.5 * daylight_h
        if hour <= sunrise or hour >= sunset:
            return 0.0
        clear_sky = np.sin(np.pi * (hour - sunrise) / max(1e-08, daylight_h))
        daily_cloud = float(np.clip(self._rng_for_date(key, salt=1201).normal(0.68, 0.16), 0.25, 1.0))
        step_cloud = float(np.clip(self._rng_for_date(key, salt=1300 + int(current_time_index)).normal(1.0, 0.06), 0.82, 1.08))
        return float(np.clip(clear_sky * daily_cloud * step_cloud, 0.0, 1.0))

    def get_outdoor_temp(self, current_time, current_time_index):
        key = (str(current_time)[:10], int(current_time_index) % int(self.steps_per_day))
        if key in self.weather_cache:
            return self.weather_cache[key]
        if self.weather_data is not None:
            try:
                value = float(self.weather_data.loc[key[0], self.weather_data.columns[key[1]]])
                if np.isfinite(value):
                    self.weather_cache[key] = value
                    return value
            except (KeyError, IndexError, TypeError, ValueError):
                pass
        current_datetime = datetime.strptime(key[0], '%Y-%m-%d')
        day_of_year = current_datetime.timetuple().tm_yday
        hour = float(key[1]) * 24.0 / max(1, int(self.steps_per_day))
        annual_mean = 10.0 + 5.0 * np.cos(2.0 * np.pi * (day_of_year - 200) / 365.0)
        diurnal = 2.0 * np.cos(2.0 * np.pi * (hour - 15.0) / 24.0)
        day_anomaly = self._rng_for_date(key[0], salt=901).normal(0.0, 1.5)
        step_noise = self._rng_for_date(key[0], salt=1000 + key[1]).normal(0.0, 0.25)
        outdoor_temp = np.clip(annual_mean + diurnal + day_anomaly + step_noise, -8.0, 28.0)
        value = round(float(outdoor_temp), 1)
        self.weather_cache[key] = value
        return value

    def get_daily_ev_trip_km(self, current_date: str, mean_km: float, sigma: float) -> float:
        key = str(current_date)[:10]
        if key not in self.ev_trip_km_cache:
            rng = self._rng_for_date(key, salt=29)
            self.ev_trip_km_cache[key] = float(rng.lognormal(mean=np.log(max(1e-06, float(mean_km))), sigma=float(sigma)))
        return float(self.ev_trip_km_cache[key])
