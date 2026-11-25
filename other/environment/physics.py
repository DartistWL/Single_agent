"""Household device dynamics."""
from datetime import datetime, timedelta
from collections import deque
import numpy as np
if __package__:
    from .interface import DataInterface
    from .config import CONFIG
    from .information_protocol import expected_hot_water_l
else:
    from interface import DataInterface
    from config import CONFIG
    from information_protocol import expected_hot_water_l
import csv
import json
import os
import sys
import pandas as pd
from typing import Optional, List, Tuple, Dict, Union

def build_wash_schedule_action_tuple(steps_per_day: int, pref: Tuple[int, int], wash_duration_hours: float) -> Tuple[int, ...]:
    spd = int(steps_per_day)
    earliest = int(pref[0] * spd / 24)
    dur_steps = max(1, int(round(float(wash_duration_hours) * spd / 24)))
    latest = int(pref[1] * spd / 24) - dur_steps
    latest = max(earliest, min(spd - 1, latest))
    n_slots = latest - earliest + 1
    if n_slots < 1:
        raise ValueError(f'洗衣机偏好窗过窄：earliest={earliest}, latest={latest}, steps_per_day={spd}')
    return (0,) + tuple(range(1, n_slots + 1))

def _calendar_day_label_cn(date_str: str) -> str:
    if not date_str or len(str(date_str)) < 10:
        return ''
    try:
        d = datetime.strptime(str(date_str)[:10], '%Y-%m-%d').date()
        wd = d.weekday()
        names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        if wd >= 5:
            return f'{names[wd]}·双休日（EV 全天在家可调度）'
        return f'{names[wd]}·工作日（EV 按离/返家，非全天可调度）'
    except ValueError:
        return ''

class FixedRewardScaler:

    def __init__(self, scaling_params: dict=None):
        self._params: Dict[str, Dict[str, float]] = {}
        if scaling_params:
            for key, p in scaling_params.items():
                self._params[key] = {'mean': float(p.get('mean', 0.0)), 'std': float(max(float(p.get('std', 1.0)), 1e-08))}

    @classmethod
    def from_config(cls):
        _rw = CONFIG.get('reward_weights', {})
        _scaling = _rw.get('FIXED_REWARD_SCALING', {})
        return cls(_scaling)

    def scale(self, key: str, value: float) -> float:
        if key not in self._params:
            return float(value)
        p = self._params[key]
        return float(float(value) / p['std'])

    @property
    def params(self) -> dict:
        return dict(self._params)

def calibrate_reward_scaling(env, num_episodes: int=30, seed: int=42) -> Dict[str, Dict[str, float]]:
    import random
    random.seed(seed)
    np.random.seed(seed)
    collectors: Dict[str, List[float]] = {'cost': [], 'comfort': [], 'carbon': [], 'aging': [], 'battery_health': [], 'pv_self_use': [], 'peak_shave': []}
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = {k: random.choice(list(v)) for k, v in env.action_space.items() if isinstance(v, (list, tuple))}
            next_state, reward, done = env.step(state, action)
            comps = getattr(env, '_last_reward_components', None)
            if comps:
                collectors['cost'].append(float(comps.get('cost_raw', 0)))
                collectors['comfort'].append(float(comps.get('comfort_unnorm', 0)))
                collectors['carbon'].append(float(comps.get('carbon_unnorm', 0)))
                collectors['aging'].append(float(max(comps.get('aging_fec_ess', 0), comps.get('aging_fec_ev', 0))))
                collectors['battery_health'].append(float(comps.get('battery_health_unnorm', 0)))
                collectors['pv_self_use'].append(float(comps.get('pv_self_raw', 0)))
                collectors['peak_shave'].append(float(comps.get('peak_shave_raw', 0)))
            state = next_state
    result = {}
    for key, values in collectors.items():
        if values:
            arr = np.array(values, dtype=np.float64)
            result[key] = {'mean': float(np.mean(arr)), 'std': max(float(np.std(arr)), 1e-08)}
        else:
            result[key] = {'mean': 0.0, 'std': 1.0}
    return result

class HomeEnergyManagementEnv:
    data_class = DataInterface

    def __init__(self, ev_capacity=None, ess_capacity=None, charge_efficiency=0.95, discharge_efficiency=0.95, data_interface: Optional[DataInterface]=None, episode_length_steps: Optional[int]=None, episode_schedule: Optional[List[Union[Tuple[str, str], Tuple[str, str, int]]]]=None, storenet_base_dir: str='data/storenet_ireland_2020', price_profile: str='ireland_tou_2020', steps_per_day: int=48, ess_aging_weight: Optional[float]=None, ev_aging_weight: Optional[float]=None, use_carbon_in_reward: bool=True, carbon_intensity_csv_path: Optional[str]='data/carbon_2020/eirgrid_roi_co2_2020_30min.csv', carbon_weight: float=0.5, reward_norm_comfort_scale: float=15.0, ac_user_behavior_std: Optional[float]=None, ev_daily_km_mean: float=50.0, ev_daily_km_sigma: float=0.2, peak_calibration_bootstrap: bool=False):
        self.ev_capacity = float(ev_capacity if ev_capacity is not None else CONFIG['environment']['ev']['ev_capacity'])
        self.ess_capacity = float(ess_capacity if ess_capacity is not None else CONFIG['environment']['battery']['ess_capacity'])
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.ev_min_charge = float(CONFIG['environment']['ev']['ev_min_charge'])
        self.ev_no_trip_reserve_kwh = float(np.clip(CONFIG['environment']['ev'].get('ev_no_trip_reserve_kwh', self.ev_min_charge), self.ev_min_charge, self.ev_capacity))
        _init_cfg = CONFIG['environment'].get('initial_state', {})
        self.randomize_initial_soc = bool(_init_cfg.get('randomize_initial_soc', False))
        self.ess_initial_soc = float(_init_cfg.get('ess_initial_soc', 0.5))
        self.ev_initial_soc = float(_init_cfg.get('ev_initial_soc', 0.4))
        self.ess_initial_soc_low = float(_init_cfg.get('ess_initial_soc_low', self.ess_initial_soc))
        self.ess_initial_soc_high = float(_init_cfg.get('ess_initial_soc_high', self.ess_initial_soc))
        self.ev_initial_soc_low = float(_init_cfg.get('ev_initial_soc_low', self.ev_initial_soc))
        self.ev_initial_soc_high = float(_init_cfg.get('ev_initial_soc_high', self.ev_initial_soc))
        self.steps_per_day = int(steps_per_day)
        self.protocol_id = str(CONFIG.get('protocol', {}).get('protocol_id', 'legacy'))
        self.environment_model_version = str(CONFIG.get('protocol', {}).get('environment_model_version', 'legacy'))
        self.tariff_config = dict(CONFIG['environment'].get('tariff', {}))
        self.ev_trip_config = dict(CONFIG['environment'].get('ev_trip', {}))
        self.export_price_mode = str(self.tariff_config.get('export_price_mode', 'import_ratio')).strip().lower()
        if self.export_price_mode == 'import_ratio':
            export_ratio = float(self.tariff_config.get('export_import_ratio', 0.0))
            export_cap_ratio = float(self.tariff_config.get('export_price_cap_ratio', 0.0))
            if not (0.0 < export_ratio < 1.0 and 0.0 < export_cap_ratio < 1.0):
                raise ValueError(f'Dynamic export tariff ratios must both lie strictly in (0, 1); got ratio={export_ratio}, cap={export_cap_ratio}')
        self.current_step_grid_export_price = 0.0
        _cross_day_cfg = dict(CONFIG.get('cross_day', {}))
        self.cross_day_enabled = bool(_cross_day_cfg.get('ENABLED', True))
        self.block_min_days = max(1, int(_cross_day_cfg.get('BLOCK_MIN_DAYS', 2)))
        self.block_max_days = max(self.block_min_days, int(_cross_day_cfg.get('BLOCK_MAX_DAYS', 4)))
        self.boundary_recovery_enabled = bool(_cross_day_cfg.get('BOUNDARY_RECOVERY_ENABLED', True))
        self.boundary_reference_mode = str(_cross_day_cfg.get('BOUNDARY_REFERENCE_MODE', 'block_initial_mobility_adjusted')).strip().lower()
        if self.boundary_reference_mode != 'block_initial_mobility_adjusted':
            raise ValueError(f'Unsupported cross-day boundary reference mode: {self.boundary_reference_mode!r}')
        self.boundary_risk_start = float(_cross_day_cfg.get('BOUNDARY_RISK_START', 0.55))
        self.boundary_ewh_reference_temp_c = float(_cross_day_cfg.get('EWH_REFERENCE_TEMP_C', 45.0))
        self.boundary_hvac_reference_c = (float(_cross_day_cfg.get('HVAC_ZONE1_REFERENCE_C', 18.0)), float(_cross_day_cfg.get('HVAC_ZONE2_REFERENCE_C', 18.0)))
        self.carry_ess_across_days = bool(_cross_day_cfg.get('CARRY_ESS', True))
        self.carry_ev_across_days = bool(_cross_day_cfg.get('CARRY_EV', True))
        self.carry_ewh_across_days = bool(_cross_day_cfg.get('CARRY_EWH', True))
        self.carry_hvac_across_days = bool(_cross_day_cfg.get('CARRY_HVAC', True))
        self.carry_motion_history_across_days = bool(_cross_day_cfg.get('CARRY_MOTION_HISTORY', True))
        if self.cross_day_enabled and (not all((self.carry_ess_across_days, self.carry_ev_across_days, self.carry_ewh_across_days, self.carry_hvac_across_days, self.carry_motion_history_across_days))):
            raise ValueError('v9 cross-day protocol requires all controllable physical and motion states to carry within a block; resetting any of them would inject or remove unmetered energy/history.')
        self._current_block_dates: List[str] = []
        self._current_block_days = self.block_min_days
        self._current_block_day_index = 0
        self._current_block_step = 0
        self.current_episode_length_steps = self.block_min_days * self.steps_per_day
        self._block_trip_energy_consumed_kwh = 0.0
        self.energy_weight = 1.0
        self.user_satisfaction_weight0 = 0.1
        self.user_satisfaction_weight1 = 0.1
        self.user_satisfaction_weight2 = 0.1
        _wash_cfg = CONFIG['environment']['wash']
        self.wash_time_penalty_coef = float(_wash_cfg['time_penalty_coef'])
        self.wash_price_penalty_coef = float(_wash_cfg['price_penalty_coef'])
        self.violation_weight = 1.0
        self.reward_scaler = FixedRewardScaler.from_config()
        _rw = CONFIG.get('reward_weights', {})
        self.w_comfort = float(_rw.get('w_comfort', 1.5))
        self.w_cost = float(_rw.get('w_cost', 1.0))
        self.w_carbon = float(_rw.get('w_carbon', 0.8))
        self.w_aging = float(_rw.get('w_aging', 0.5))
        self.w_battery_health = float(_rw.get('w_battery_health', 0.0))
        self.w_pv_self_use = float(_rw.get('w_pv_self_use', 0.3))
        self.w_peak_shave = float(_rw.get('w_peak_shave', 0.0))
        self.w_violation = float(_rw.get('w_violation', 1.0))
        self.w_smoothness = float(_rw.get('w_smoothness', 0.5))
        self.w_invalid = float(_rw.get('w_invalid', 0.2))
        self.w_wash = float(_rw.get('w_wash', 0.1))
        self.reward_design_version = str(_rw.get('reward_design_version', 'legacy'))
        self.comfort_target_score = float(_rw.get('comfort_target_score', 0.96))
        self.comfort_score_scale = float(_rw.get('comfort_score_scale', 30.0))
        self.comfort_deficit_linear_weight = float(_rw.get('comfort_deficit_linear_weight', 2.0))
        self.comfort_deficit_quadratic_weight = float(_rw.get('comfort_deficit_quadratic_weight', 4.0))
        self.comfort_deficit_barrier_weight = float(_rw.get('comfort_deficit_barrier_weight', 0.0))
        self.comfort_deficit_barrier_scale = float(_rw.get('comfort_deficit_barrier_scale', 8.0))
        self.comfort_saturation_score = float(_rw.get('comfort_saturation_score', 0.98))
        self.comfort_mid_reward_weight = float(_rw.get('comfort_mid_reward_weight', 0.18))
        self.comfort_saturation_reward_weight = float(_rw.get('comfort_saturation_reward_weight', 0.03))
        self.ac_comfort_target_score = float(_rw.get('ac_comfort_target_score', self.comfort_target_score))
        self.ac_comfort_saturation_score = float(_rw.get('ac_comfort_saturation_score', self.comfort_saturation_score))
        self.ac_comfort_penalty_cap = float(_rw.get('ac_comfort_penalty_cap', 2.0))
        self.w_hot_water_service = float(_rw.get('w_hot_water_service', 0.45))
        self.w_boundary_recovery = float(_rw.get('w_boundary_recovery', 0.0))
        self.boundary_terminal_multiplier = float(_rw.get('boundary_terminal_multiplier', 2.0))
        self.hot_water_service_temp_c = float(_rw.get('hot_water_service_temp_c', 42.0))
        self.hot_water_reserve_temp_c = float(_rw.get('hot_water_reserve_temp_c', 42.0))
        self.hot_water_shortfall_scale_c = max(1e-08, float(_rw.get('hot_water_shortfall_scale_c', 10.0)))
        self.hot_water_service_window_weight = float(_rw.get('hot_water_service_window_weight', 1.0))
        self.hot_water_draw_event_weight = float(_rw.get('hot_water_draw_event_weight', 1.25))
        self.hot_water_offpeak_weight = float(_rw.get('hot_water_offpeak_weight', 0.2))
        self.hot_water_deficit_linear_weight = float(_rw.get('hot_water_deficit_linear_weight', 2.0))
        self.hot_water_deficit_quadratic_weight = float(_rw.get('hot_water_deficit_quadratic_weight', 2.0))
        self.hot_water_penalty_cap_per_step = float(_rw.get('hot_water_penalty_cap_per_step', 1.5))
        self.hot_water_service_bonus_weight = float(_rw.get('hot_water_service_bonus_weight', 0.08))
        self.hot_water_overheat_temp_c = float(_rw.get('hot_water_overheat_temp_c', 60.0))
        self.hot_water_overheat_weight = float(_rw.get('hot_water_overheat_weight', 0.1))
        self.raw_action_event_weight = float(_rw.get('raw_action_event_weight', self.w_invalid))
        self.raw_action_distance_weight = float(_rw.get('raw_action_distance_weight', self.w_invalid))
        self.raw_action_distance_normalization = str(_rw.get('raw_action_distance_normalization', 'per_device_max_abs_action'))
        self.raw_action_distance_cap = max(0.0, float(_rw.get('raw_action_distance_cap', 2.0)))
        self.temp_weight = float(CONFIG['environment']['comfort']['temp_weight'])
        _peak_cfg = CONFIG.get('peak_shaving', {})
        self.peak_calibration_bootstrap = bool(peak_calibration_bootstrap)
        self.peak_shaving_enabled = bool(_peak_cfg.get('enabled', False)) and (not self.peak_calibration_bootstrap)
        self.peak_threshold_kw = float(_peak_cfg.get('threshold_kw', 0.0) or 0.0)
        self.peak_threshold_source = str(_peak_cfg.get('threshold_source', 'config'))
        self.peak_target_ratio = float(_peak_cfg.get('target_ratio', 0.9) or 0.9)
        self.peak_max_penalty_weight = float(_peak_cfg.get('peak_max_penalty_weight', 1.0) or 1.0)
        self.peak_auc_penalty_weight = float(_peak_cfg.get('peak_auc_penalty_weight', 0.25) or 0.25)
        self.peak_relief_reward_weight = float(_peak_cfg.get('peak_relief_reward_weight', 0.0) or 0.0)
        self.peak_relief_cap_kw = max(1e-08, float(_peak_cfg.get('peak_relief_cap_kw', 4.0) or 4.0))
        self.peak_relief_min_comfort_score = float(_peak_cfg.get('peak_relief_min_comfort_score', 0.93))
        self.peak_relief_full_comfort_score = float(_peak_cfg.get('peak_relief_full_comfort_score', 0.96))
        self.peak_relief_ev_reserve_margin_kwh = max(1e-08, float(_peak_cfg.get('peak_relief_ev_reserve_margin_kwh', 1.0) or 1.0))
        self.peak_calibration_baseline_mean_kw = 0.0
        self.current_episode_baseline_peak_kw = 0.0
        self.current_episode_peak_target_kw = self.peak_threshold_kw
        self.current_episode_peak_import_kw = 0.0
        self.current_episode_peak_potential = 0.0
        if bool(_peak_cfg.get('use_calibration_file', True)) and (not self.peak_calibration_bootstrap):
            self._load_peak_calibration(_peak_cfg.get('calibration_file', ''))
            if self.w_peak_shave > 0.0 and (not self.peak_shaving_enabled):
                raise FileNotFoundError('统一 Reward 需要 train split 生成的 peak_calibration.json；请先运行 Physical-Baseline calibration 命令。')
        _mask_cfg = CONFIG.get('mask', {})
        self.mask_free_low = float(_mask_cfg.get('free_low', 0.25))
        self.mask_free_high = float(_mask_cfg.get('free_high', 0.75))
        self.mask_remind_low = float(_mask_cfg.get('remind_low', 0.15))
        self.mask_remind_high = float(_mask_cfg.get('remind_high', 0.85))
        self.mask_warn_low = float(_mask_cfg.get('warn_low', 0.05))
        self.mask_warn_high = float(_mask_cfg.get('warn_high', 0.95))
        self.mask_remind_pen = float(_mask_cfg.get('remind_penalty', -1.5))
        self.mask_warn_pen = float(_mask_cfg.get('warn_penalty', -4.0))
        self.mask_hard_val = float(_mask_cfg.get('hard_mask_value', -20.0))
        self.mask_soc_recovery_bonus = float(_mask_cfg.get('soc_recovery_bonus', 0.8))
        self.mask_idle_bias = float(_mask_cfg.get('idle_bias', 0.25))
        self.mask_reversal_penalty = float(_mask_cfg.get('reversal_penalty', -1.2))
        self.mask_mid_soc_idle_bias = float(_mask_cfg.get('mid_soc_idle_bias', 0.2))
        self.mask_economic_bias = float(_mask_cfg.get('economic_bias', 0.9))
        self.mask_low_price_charge_threshold = float(_mask_cfg.get('low_price_charge_threshold', 0.3))
        self.mask_high_price_discharge_threshold = float(_mask_cfg.get('high_price_discharge_threshold', 0.7))
        self.mask_pv_surplus_charge_bonus = float(_mask_cfg.get('pv_surplus_charge_bonus', 0.8))
        self.mask_non_economic_power_penalty = float(_mask_cfg.get('non_economic_power_penalty', -0.45))
        self.mask_ev_deadline_soft_penalty = float(_mask_cfg.get('ev_deadline_soft_penalty', -3.0))
        self.mask_ev_deadline_hard_margin_kwh = float(_mask_cfg.get('ev_deadline_hard_margin_kwh', 0.25))
        self.mask_mode = str(_mask_cfg.get('mode', 'full') or 'full').lower()
        _prior_cfg = CONFIG.get('action_prior', {})
        self.action_prior_enabled = bool(_prior_cfg.get('ENABLED', True))
        self.prior_forecast_horizon_steps = max(1, int(_prior_cfg.get('FORECAST_HORIZON_STEPS', 8)))
        self.prior_strength = float(_prior_cfg.get('STRENGTH', 1.0))
        self.prior_max_abs_bias = float(_prior_cfg.get('MAX_ABS_BIAS', 4.0))
        self.prior_dominated_bias = float(_prior_cfg.get('DOMINATED_BIAS', -2.5))
        self.prior_idle_bias = float(_prior_cfg.get('IDLE_BIAS', 0.2))
        self.prior_ess_price_weight = float(_prior_cfg.get('ESS_PRICE_WEIGHT', 1.0))
        self.prior_ess_carbon_weight = float(_prior_cfg.get('ESS_CARBON_WEIGHT', 0.7))
        self.prior_ess_pv_weight = float(_prior_cfg.get('ESS_PV_WEIGHT', 1.0))
        self.prior_ess_peak_weight = float(_prior_cfg.get('ESS_PEAK_WEIGHT', 1.2))
        self.prior_ess_degradation_weight = float(_prior_cfg.get('ESS_DEGRADATION_WEIGHT', 1.5))
        self.prior_ess_edge_soc_weight = float(_prior_cfg.get('ESS_EDGE_SOC_WEIGHT', 0.8))
        self.prior_ess_reversal_weight = float(_prior_cfg.get('ESS_REVERSAL_WEIGHT', 1.2))
        self.prior_ess_idle_margin = float(_prior_cfg.get('ESS_IDLE_MARGIN', 0.15))
        self.prior_peak_soft_threshold_kw = float(_prior_cfg.get('PEAK_SOFT_THRESHOLD_KW', 4.0))
        self.prior_ess_reserve_soc = float(_prior_cfg.get('ESS_RESERVE_SOC', 0.35))
        self.prior_ess_reserve_weight = float(_prior_cfg.get('ESS_RESERVE_WEIGHT', 0.8))
        self.prior_ess_peak_discharge_soc_floor = float(_prior_cfg.get('ESS_PEAK_DISCHARGE_SOC_FLOOR', 0.28))
        self.prior_ev_deadline_weight = float(_prior_cfg.get('EV_DEADLINE_WEIGHT', 3.0))
        self.prior_ev_cheap_charge_weight = float(_prior_cfg.get('EV_CHEAP_CHARGE_WEIGHT', 0.8))
        self.prior_ev_v2h_weight = float(_prior_cfg.get('EV_V2H_WEIGHT', 0.4))
        self.prior_ev_mobility_risk_weight = float(_prior_cfg.get('EV_MOBILITY_RISK_WEIGHT', 2.0))
        self.prior_ev_degradation_weight = float(_prior_cfg.get('EV_DEGRADATION_WEIGHT', 0.8))
        self.prior_ev_reserve_margin_kwh = float(_prior_cfg.get('EV_RESERVE_MARGIN_KWH', 2.0))
        self.prior_ev_deadline_business_weight = float(_prior_cfg.get('EV_DEADLINE_BUSINESS_WEIGHT', 0.45))
        self.prior_wash_price_weight = float(_prior_cfg.get('WASH_PRICE_WEIGHT', 1.2))
        self.prior_wash_carbon_weight = float(_prior_cfg.get('WASH_CARBON_WEIGHT', 0.5))
        self.prior_wash_deviation_weight = float(_prior_cfg.get('WASH_DEVIATION_WEIGHT', 0.6))
        self.prior_wash_deadline_weight = float(_prior_cfg.get('WASH_DEADLINE_WEIGHT', 2.0))
        self.prior_thermal_comfort_weight = float(_prior_cfg.get('THERMAL_COMFORT_WEIGHT', 1.0))
        self.prior_thermal_econ_weight = float(_prior_cfg.get('THERMAL_ECON_WEIGHT', 0.4))
        self.prior_ewh_preheat_weight = float(_prior_cfg.get('EWH_PREHEAT_WEIGHT', 0.8))
        self.prior_branch_multipliers = dict(_prior_cfg.get('BRANCH_MULTIPLIERS', {}) or {})
        self.current_action_prior_abs_mean = 0.0
        self.current_action_prior_max_abs = 0.0
        self.current_action_prior_branch_mean = {}
        _ev_deadline_cfg = CONFIG.get('ev_deadline', {})
        self.ev_deadline_enabled = bool(_ev_deadline_cfg.get('ENABLED', False))
        self.ev_deadline_target_margin_kwh = float(_ev_deadline_cfg.get('TARGET_MARGIN_KWH', 1.0))
        self.ev_deadline_hard_target_margin_kwh = float(_ev_deadline_cfg.get('HARD_TARGET_MARGIN_KWH', 0.0))
        self.ev_deadline_risk_start = float(_ev_deadline_cfg.get('RISK_START', 0.2))
        self.ev_deadline_risk_full = float(_ev_deadline_cfg.get('RISK_FULL', 0.95))
        self.ev_deadline_reward_weight = float(_ev_deadline_cfg.get('REWARD_WEIGHT', 3.0))
        self.ev_deadline_reward_cap = float(_ev_deadline_cfg.get('REWARD_CAP', 1.2))
        self.ev_deadline_lag_penalty_weight = float(_ev_deadline_cfg.get('LAG_PENALTY_WEIGHT', 1.2))
        self.ev_deadline_lag_penalty_cap = float(_ev_deadline_cfg.get('LAG_PENALTY_CAP', 1.0))
        self.ev_deadline_terminal_penalty_weight = float(_ev_deadline_cfg.get('TERMINAL_PENALTY_WEIGHT', 8.0))
        self.ev_deadline_terminal_penalty_cap = float(_ev_deadline_cfg.get('TERMINAL_PENALTY_CAP', 10.0))
        self.ev_deadline_business_penalty_weight = float(_ev_deadline_cfg.get('BUSINESS_PENALTY_WEIGHT', 0.45))
        self.ev_deadline_urgent_business_floor = float(_ev_deadline_cfg.get('URGENT_BUSINESS_FLOOR', 0.75))
        self.ev_deadline_price_weight = float(_ev_deadline_cfg.get('PRICE_WEIGHT', 0.35))
        self.ev_deadline_carbon_weight = float(_ev_deadline_cfg.get('CARBON_WEIGHT', 0.25))
        self.ev_deadline_peak_weight = float(_ev_deadline_cfg.get('PEAK_WEIGHT', 0.45))
        self.ev_deadline_pv_relief_weight = float(_ev_deadline_cfg.get('PV_RELIEF_WEIGHT', 0.35))
        self.ev_deadline_within_episode_only = bool(_ev_deadline_cfg.get('WITHIN_EPISODE_ONLY', True))
        self.ev_shield_enabled = bool(_ev_deadline_cfg.get('SHIELD_ENABLED', True))
        self.ev_shield_intervention_weight = float(_ev_deadline_cfg.get('SHIELD_INTERVENTION_WEIGHT', 0.2))
        self.ev_deadline_uncertainty_margin_kwh = float(_ev_deadline_cfg.get('UNCERTAINTY_MARGIN_KWH', 1.0))
        self.current_ev_deadline_risk = 0.0
        self.current_ev_deadline_required_kw = 0.0
        self.current_ev_deadline_gap_kwh = 0.0
        self.current_ev_deadline_reward = 0.0
        self.current_ev_deadline_business_penalty = 0.0
        self.current_ev_shield_intervened = 0
        self.current_ev_shield_action_delta_kw = 0.0
        self.current_ev_forced_charge_kwh = 0.0
        _terminal_cfg = CONFIG.get('terminal_energy', {})
        self.terminal_energy_enabled = bool(_terminal_cfg.get('ENABLED', True))
        self.terminal_settle_surplus = bool(_terminal_cfg.get('SETTLE_SURPLUS', False))
        self.terminal_ewh_enabled = bool(_terminal_cfg.get('EWH_ENABLED', True))
        self.terminal_ewh_reference_temp_c = float(_terminal_cfg.get('EWH_REFERENCE_TEMP_C', 45.0))
        self.terminal_ewh_surplus_credit_cap_temp_c = float(_terminal_cfg.get('EWH_SURPLUS_CREDIT_CAP_TEMP_C', 45.0))
        self.terminal_hvac_enabled = bool(_terminal_cfg.get('HVAC_ENABLED', True))
        self.terminal_hvac_settle_surplus = bool(_terminal_cfg.get('HVAC_SETTLE_SURPLUS', False))
        self.terminal_hvac_surplus_credit_cap_c = float(_terminal_cfg.get('HVAC_SURPLUS_CREDIT_CAP_C', 24.0))
        self._episode_initial_ess_kwh = 0.0
        self._episode_initial_ev_kwh = 0.0
        self._episode_trip_kwh = 0.0
        self._episode_mean_price = 0.0
        self._episode_mean_carbon_intensity = 0.0
        self._episode_initial_ewh_temp_c = float(CONFIG['environment']['ewh'].get('temp_init', 40.0))
        self._episode_initial_indoor_temp_c = float(CONFIG['environment']['comfort'].get('indoor_temp_init', 20.5))
        self._episode_initial_indoor_temp2_c = float(CONFIG['environment']['comfort'].get('indoor_temp2_init', 20.0))
        self.current_terminal_ewh_deficit_kwh_th = 0.0
        self.current_terminal_hvac_deficit_kwh_th = 0.0
        self.current_terminal_grid_equivalent_kwh = 0.0
        self.current_terminal_settlement_cost_eur = 0.0
        self.current_terminal_settlement_carbon_kg = 0.0
        self._last_action_info: Dict[str, object] = {}
        self.action_smoothness_weight = 1.0
        self.invalid_action_weight = self.raw_action_distance_weight
        _bat_motion_cfg = CONFIG.get('battery_aware', {})
        self.battery_aware_enabled = bool(_bat_motion_cfg.get('ENABLED', True))
        self.battery_ramp_free = float(_bat_motion_cfg.get('RAMP_FREE', 0.5))
        self.battery_ramp_weight = float(_bat_motion_cfg.get('RAMP_WEIGHT', 0.08))
        self.battery_switch_weight = float(_bat_motion_cfg.get('SWITCH_WEIGHT', 0.22))
        self.battery_micro_cycle_weight = float(_bat_motion_cfg.get('MICRO_CYCLE_WEIGHT', 0.08))
        self.battery_high_power_switch_weight = float(_bat_motion_cfg.get('HIGH_POWER_SWITCH_WEIGHT', 0.18))
        self.battery_micro_cycle_window_steps = max(1, int(_bat_motion_cfg.get('MICRO_CYCLE_WINDOW_STEPS', 4)))
        self.legacy_ess_aging_weight = ess_aging_weight
        self.legacy_ev_aging_weight = ev_aging_weight
        ess_aging_model = str(CONFIG['environment']['battery'].get('aging_cost_model', 'replacement_cost_fec'))
        ev_aging_model = str(CONFIG['environment']['ev'].get('aging_cost_model', 'replacement_cost_fec'))
        if ess_aging_model != 'replacement_cost_fec' or ev_aging_model != 'replacement_cost_fec':
            raise ValueError(f'v9 requires replacement_cost_fec aging for both ESS and EV; got ESS={ess_aging_model!r}, EV={ev_aging_model!r}')
        self.aging_cost_model = 'replacement_cost_fec'
        self.ess_replacement_cost_eur = float(CONFIG['environment']['battery'].get('replacement_cost_eur_per_kwh', 350.0)) * self.ess_capacity
        self.ev_replacement_cost_eur = float(CONFIG['environment']['ev'].get('replacement_cost_eur_per_kwh', 150.0)) * self.ev_capacity
        self.ess_cycle_life_fec = max(1.0, float(CONFIG['environment']['battery'].get('cycle_life_fec', 4000.0)))
        self.ev_cycle_life_fec = max(1.0, float(CONFIG['environment']['ev'].get('cycle_life_fec', 2000.0)))
        _ess_health_cfg = CONFIG['environment'].get('ess_health', {})
        self.ess_healthy_soc_low = float(_ess_health_cfg.get('healthy_soc_low', 0.25))
        self.ess_healthy_soc_high = float(_ess_health_cfg.get('healthy_soc_high', 0.8))
        self.ess_low_soc_weight = float(_ess_health_cfg.get('low_soc_weight', 0.15))
        self.ess_high_soc_weight = float(_ess_health_cfg.get('high_soc_weight', 0.05))
        self.ess_terminal_soc_min = float(_ess_health_cfg.get('terminal_soc_min', 0.3))
        self.ess_terminal_soc_weight = float(_ess_health_cfg.get('terminal_soc_weight', 0.2))
        self.ess_throughput_weight = float(_ess_health_cfg.get('throughput_weight', 0.0))
        _ev_cfg = CONFIG['environment'].get('ev', {})
        self.ev_healthy_soc_low = float(_ev_cfg.get('ev_healthy_soc_low', 0.2))
        self.ev_healthy_soc_high = float(_ev_cfg.get('ev_healthy_soc_high', 0.9))
        self.ev_low_soc_weight = float(_ev_cfg.get('ev_low_soc_weight', 0.0))
        self.ev_high_soc_weight = float(_ev_cfg.get('ev_high_soc_weight', 0.0))
        self.ev_throughput_weight = float(_ev_cfg.get('ev_throughput_weight', 0.0))
        self.use_carbon_in_reward = bool(use_carbon_in_reward)
        self.carbon_weight = float(CONFIG['environment']['carbon']['carbon_weight'])
        self.reward_norm_comfort_scale = float(CONFIG['environment']['comfort']['reward_norm_comfort_scale'])
        self._carbon_intensity_csv_path = carbon_intensity_csv_path
        self._carbon_df: Optional[pd.DataFrame] = None
        self._carbon_fallback_kg_per_kwh: float = 0.25
        self._carbon_kg_per_kwh_max: float = 0.5
        self.carbon_source = 'disabled' if not self.use_carbon_in_reward else 'fixed_fallback_0.25'
        self.prev_battery_power = 0.0
        self.prev_ev_power = 0.0
        self._prev_ev_power_sign = 0
        self._ess_steps_since_switch = self.steps_per_day
        self._ev_steps_since_switch = self.steps_per_day
        self._ess_recent_switch_flags = deque(maxlen=self.battery_micro_cycle_window_steps)
        self._ev_recent_switch_flags = deque(maxlen=self.battery_micro_cycle_window_steps)
        self.current_step_ess_ramp_norm = 0.0
        self.current_step_ev_ramp_norm = 0.0
        self.current_step_battery_switching_cost = 0.0
        self.current_step_ess_switching_cost = 0.0
        self.current_step_ev_switching_cost = 0.0
        self.current_step_high_power_reversal_count = 0.0
        self.current_step_micro_cycle_count = 0.0
        self.current_invalid_action_penalty = 0.0
        self.current_invalid_action_dist = 0.0
        self.current_invalid_action_dist_kw = 0.0
        self.current_ess_repair_distance_norm = 0.0
        self.current_ev_repair_distance_norm = 0.0
        self.current_smoothness_penalty = 0.0
        self.current_step_raw_violations = 0
        self.current_step_chattering_events = 0
        self.current_step_ess_chattering_events = 0
        self.current_step_ev_chattering_events = 0
        _comfort_cfg = CONFIG['environment']['comfort']
        self.ac_user_behavior_std = float(_comfort_cfg.get('ac_user_behavior_std', 0.05) if ac_user_behavior_std is None else ac_user_behavior_std)
        self.zone_thermal_capacity_kwh_per_c = (max(1e-06, float(_comfort_cfg.get('zone1_thermal_capacity_kwh_per_c', 3.2))), max(1e-06, float(_comfort_cfg.get('zone2_thermal_capacity_kwh_per_c', 2.4))))
        self.zone_envelope_conductance_kw_per_c = (max(0.0, float(_comfort_cfg.get('zone1_envelope_conductance_kw_per_c', 0.1))), max(0.0, float(_comfort_cfg.get('zone2_envelope_conductance_kw_per_c', 0.075))))
        self.interzone_conductance_kw_per_c = max(0.0, float(_comfort_cfg.get('interzone_conductance_kw_per_c', 0.045)))
        self.hvac_shared_max_power_kw = max(1e-06, float(_comfort_cfg.get('hvac_shared_max_power_kw', 3.0)))
        self.hvac_zone_max_power_kw = max(1e-06, float(_comfort_cfg.get('hvac_zone_max_power_kw', 2.2)))
        self.hvac_max_power_kw = self.hvac_shared_max_power_kw
        self.hvac_deadband_c = max(0.0, float(_comfort_cfg.get('hvac_deadband_c', 0.5)))
        self.hvac_full_power_error_c = max(1e-06, float(_comfort_cfg.get('hvac_full_power_error_c', 4.0)))
        self.heating_cop_at_7c = float(_comfort_cfg.get('heating_cop_at_7c', 3.2))
        self.heating_cop_slope_per_c = float(_comfort_cfg.get('heating_cop_slope_per_c', 0.06))
        self.heating_cop_min = float(_comfort_cfg.get('heating_cop_min', 1.8))
        self.heating_cop_max = float(_comfort_cfg.get('heating_cop_max', 4.5))
        self.cooling_eer = max(1.0, float(_comfort_cfg.get('cooling_eer', 3.0)))
        self.internal_gain_base_kw = max(0.0, float(_comfort_cfg.get('internal_gain_base_kw', 0.18)))
        self.internal_gain_per_occupant_kw = max(0.0, float(_comfort_cfg.get('internal_gain_per_occupant_kw', 0.08)))
        self.internal_load_heat_fraction = max(0.0, float(_comfort_cfg.get('internal_load_heat_fraction', 0.35)))
        self.internal_load_heat_cap_kw = max(0.0, float(_comfort_cfg.get('internal_load_heat_cap_kw', 0.7)))
        self.zone_solar_gain_peak_kw = (max(0.0, float(_comfort_cfg.get('zone1_solar_gain_peak_kw', 0.65))), max(0.0, float(_comfort_cfg.get('zone2_solar_gain_peak_kw', 0.35))))
        self.zone_occupied_bounds_c = ((float(_comfort_cfg.get('zone1_occupied_lower_c', 20.0)), float(_comfort_cfg.get('zone1_occupied_upper_c', 24.0))), (float(_comfort_cfg.get('zone2_occupied_lower_c', 18.0)), float(_comfort_cfg.get('zone2_occupied_upper_c', 23.0))))
        self.zone_acceptable_bounds_c = ((float(_comfort_cfg.get('zone1_acceptable_lower_c', 18.0)), float(_comfort_cfg.get('zone1_acceptable_upper_c', 28.0))), (float(_comfort_cfg.get('zone2_acceptable_lower_c', 16.0)), float(_comfort_cfg.get('zone2_acceptable_upper_c', 27.0))))
        self.acceptable_low_penalty_scale = max(0.0, float(_comfort_cfg.get('acceptable_low_penalty_scale', 0.5)))
        self.acceptable_high_penalty_scale = max(0.0, float(_comfort_cfg.get('acceptable_high_penalty_scale', 1.0)))
        self.unacceptable_low_linear_slope = max(0.0, float(_comfort_cfg.get('unacceptable_low_linear_slope', 1.0)))
        self.unacceptable_high_linear_slope = max(0.0, float(_comfort_cfg.get('unacceptable_high_linear_slope', 2.0)))
        self.occupied_safety_bounds_c = (float(_comfort_cfg.get('occupied_safety_lower_c', 10.0)), float(_comfort_cfg.get('occupied_safety_upper_c', 32.0)))
        self.unoccupied_bounds_c = (float(_comfort_cfg.get('unoccupied_lower_c', 16.0)), float(_comfort_cfg.get('unoccupied_upper_c', 28.0)))
        self.ev_daily_km_mean = float(ev_daily_km_mean) if ev_daily_km_mean is not None else float(CONFIG['environment']['ev_trip'].get('daily_km_mean', 50.0))
        self.ev_daily_km_sigma = float(ev_daily_km_sigma) if ev_daily_km_sigma is not None else float(CONFIG['environment']['ev_trip'].get('daily_km_sigma', 0.2))
        self.total_cost = 0
        self.episode_costs = []
        self.current_step_cost = 0
        self.current_core_reward = 0.0
        self.current_step_emission_kg = 0.0
        self.current_step_grid_import_kwh = 0.0
        self.current_step_grid_export_kwh = 0.0
        self.current_step_grid_import_kw = 0.0
        self.current_step_grid_export_kw = 0.0
        self.current_step_net_load_kw = 0.0
        self.current_step_grid_import_cost = 0.0
        self.current_step_grid_export_revenue = 0.0
        self.current_step_net_energy_cost = 0.0
        self.last_step_carbon_intensity_kg_per_kwh = 0.0
        self.current_step_ev_departure_shortfall = 0.0
        self.current_step_ev_departure_event = 0
        self.current_step_ev_departure_success = 0
        self.current_step_ev_trip_consumption_kwh = 0.0
        self.current_step_user_flow = 0.0
        self.current_step_hot_water_draw_l = 0.0
        self.current_step_hot_water_delivery_temp_c = float(CONFIG['environment']['ewh'].get('temp_init', 40.0))
        self.current_ev_deadline_risk = 0.0
        self.current_ev_deadline_required_kw = 0.0
        self.current_ev_deadline_gap_kwh = 0.0
        self.current_ev_deadline_reward = 0.0
        self.current_ev_deadline_business_penalty = 0.0
        self.current_ev_shield_intervened = 0
        self.current_ev_shield_action_delta_kw = 0.0
        self.current_ev_forced_charge_kwh = 0.0
        self.current_terminal_energy_penalty = 0.0
        self.current_terminal_ess_deficit_kwh = 0.0
        self.current_terminal_ev_deficit_kwh = 0.0
        self.current_terminal_ewh_deficit_kwh_th = 0.0
        self.current_terminal_hvac_deficit_kwh_th = 0.0
        self.current_terminal_grid_equivalent_kwh = 0.0
        self.current_terminal_settlement_cost_eur = 0.0
        self.current_terminal_settlement_carbon_kg = 0.0
        self.current_initial_ev_feasibility_adjustment_kwh = 0.0
        self.current_ev_reserve_floor_kwh = self.ev_min_charge
        self.current_ev_reserve_gap_kwh = 0.0
        self._last_action_info = {}
        self.current_step_wash_started = 0
        self.current_step_wash_completed = 0
        self.current_step_wash_start_price = 0.0
        self.current_step_wash_on_window = 0
        self.current_step_wash_price_rank = 0.0
        self.current_action_prior_abs_mean = 0.0
        self.current_action_prior_max_abs = 0.0
        self.current_action_prior_branch_mean = {}
        self.current_step_ess_charge_input_kwh = 0.0
        self.current_step_ess_discharge_output_kwh = 0.0
        self.current_step_ess_charge_from_pv_kwh = 0.0
        self.current_step_ess_charge_non_pv_kwh = 0.0
        self.current_step_ess_discharge_to_load_kwh = 0.0
        self.current_step_ess_discharge_to_grid_kwh = 0.0
        self.current_step_pv_generation_kwh = 0.0
        self.current_step_pv_self_consumed_kwh = 0.0
        self.current_step_pv_export_kwh = 0.0
        self.current_step_base_load_kwh = 0.0
        self.current_step_hvac_energy_kwh = 0.0
        self.current_step_ewh_energy_kwh = 0.0
        self.current_step_wash_energy_kwh = 0.0
        self.current_step_ev_charge_input_kwh = 0.0
        self.current_step_ev_discharge_output_kwh = 0.0
        self.current_step_ev_discharge_to_load_kwh = 0.0
        self.current_step_ev_discharge_to_grid_kwh = 0.0
        self.current_step_energy_balance_residual_kwh = 0.0
        self.current_step_hvac_safety_interventions = 0
        self._ess_stored_from_pv_kwh = 0.0
        self._ess_stored_from_grid_kwh = 0.0
        self._ess_cumulative_fec_today = 0.0
        self._ev_cumulative_fec_today = 0.0
        self._ess_cycle_count_today = 0
        self._ess_cumulative_fec_block = 0.0
        self._ev_cumulative_fec_block = 0.0
        self._ess_cycle_count_block = 0
        self._prev_battery_power_sign = 0
        self.wash_machine_preferred_time = (int(_wash_cfg['preferred_time_start']), int(_wash_cfg['preferred_time_end']))
        self.wash_machine_duration = float(_wash_cfg['duration_hours'])
        self.state_space = {'home_load': (0, DataInterface.MAX_PLAUSIBLE_HOME_LOAD_KW), 'pv_generation': (0, DataInterface.MAX_PLAUSIBLE_PV_POWER_KW), 'ess_state': (0, self.ess_capacity), 'ev_battery_state': (0, self.ev_capacity), 'time_index': (0, max(1, self.steps_per_day)), 'electricity_price': (0, 2), 'grid_export_price': (0, 2), 'future_price_1': (0, 2), 'future_price_2': (0, 2), 'future_price_3': (0, 2), 'future_price_4': (0, 2), 'future_price_mean': (0, 2), 'future_load_mean': (0, DataInterface.MAX_PLAUSIBLE_HOME_LOAD_KW), 'future_pv_mean': (0, DataInterface.MAX_PLAUSIBLE_PV_POWER_KW), 'future_net_load_mean': (-DataInterface.MAX_PLAUSIBLE_PV_POWER_KW, DataInterface.MAX_PLAUSIBLE_HOME_LOAD_KW), 'future_carbon_intensity_mean': (0.0, 1.0), 'price_rank_lookahead': (0.0, 1.0), 'solar_irradiance_norm': (0.0, 1.0), 'zone1_occupied': (0.0, 1.0), 'zone2_occupied': (0.0, 1.0), 'hot_water_draw_forecast_norm': (0.0, 1.0), 'temperature': (-10, 35), 'indoor_temp': (10, 40), 'indoor_temp2': (10, 40), 'wash_machine_state': (0, 1), 'Air_conditioner_power': (0, self.hvac_zone_max_power_kw), 'Air_conditioner_power2': (0, self.hvac_zone_max_power_kw), 'ewh_temp': (float(CONFIG['environment']['ewh'].get('temp_min', 10.0)), float(CONFIG['environment']['ewh'].get('temp_max', 100.0))), 'ewh_power': (0, float(CONFIG['environment']['ewh'].get('heater_max_power_kw', 1.8))), 'weekday_sin': (-1, 1), 'weekday_cos': (-1, 1), 'is_weekend': (0, 1), 'day_of_year_sin': (-1, 1), 'day_of_year_cos': (-1, 1), 'ev_at_home': (0, 1), 'ev_steps_to_departure_norm': (0.0, 1.0), 'ev_deadline_gap_norm': (0.0, 1.0), 'ev_required_charge_norm': (0.0, 1.0), 'ev_operational_reserve_norm': (0.0, 1.0), 'block_day_index_norm': (0.0, 1.0), 'block_length_days_norm': (0.0, 1.0), 'block_step_norm': (0.0, 1.0), 'steps_to_block_end_norm': (0.0, 1.0), 'is_day_boundary': (0.0, 1.0), 'ess_boundary_gap_norm': (0.0, 1.0), 'ev_boundary_gap_norm': (0.0, 1.0), 'ewh_boundary_gap_norm': (0.0, 1.0), 'hvac_boundary_gap_norm': (0.0, 1.0), 'boundary_recovery_risk': (0.0, 1.0), 'time_index_sin': (-1, 1), 'time_index_cos': (-1, 1), 'wash_pending': (0, 1), 'wash_planned_start_norm': (0, 1), 'wash_steps_until_start_norm': (0, 1), 'wash_remaining_runtime_norm': (0, 1), 'carbon_intensity_norm': (0.0, 1.0), 'ess_charge_pv_fraction': (0.0, 1.0), 'episode_peak_import_norm': (0.0, 3.0), 'peak_target_margin_norm': (-3.0, 3.0), 'peak_pressure_norm': (0.0, 3.0), 'ess_cumulative_fec_today': (0.0, 10.0), 'ev_cumulative_fec_today': (0.0, 10.0), 'ess_cycle_count_today': (0.0, 100.0), 'ess_prev_power_norm': (-1.0, 1.0), 'ev_prev_power_norm': (-1.0, 1.0), 'ess_power_sign': (-1.0, 1.0), 'ev_power_sign': (-1.0, 1.0), 'ess_steps_since_switch_norm': (0.0, 1.0), 'ev_steps_since_switch_norm': (0.0, 1.0), 'ess_ramp_norm': (0.0, 2.0), 'ev_ramp_norm': (0.0, 2.0)}
        ess_max_kw = float(CONFIG['environment']['battery']['ess_max_charge_power'])
        ev_max_kw = float(CONFIG['environment']['ev']['ev_max_charge_power'])
        self.action_space = {'ev_power': (-ev_max_kw, -ev_max_kw / 2.0, 0, ev_max_kw / 2.0, ev_max_kw), 'battery_power': (-ess_max_kw, -ess_max_kw / 2.0, 0, ess_max_kw / 2.0, ess_max_kw), 'wash_machine_schedule': build_wash_schedule_action_tuple(self.steps_per_day, self.wash_machine_preferred_time, self.wash_machine_duration), 'Air_conditioner_set_temp': (16, 18, 20, 22, 24, 26, 28, 30), 'Air_conditioner_set_temp2': (16, 18, 20, 22, 24, 26, 28, 30), 'ewh_set_temp': (0, 40, 42, 45, 50, 55, 60, 65)}
        protocol_min_steps = int(CONFIG.get('protocol', {}).get('episode_length_steps', self.block_min_days * self.steps_per_day))
        self.episode_length_steps = max(self.block_min_days * self.steps_per_day, protocol_min_steps)
        if episode_length_steps not in (None, self.steps_per_day, self.episode_length_steps):
            raise ValueError('episode_length_steps 由 v9 跨日协议决定；请勿为单个算法改写。')
        self.episode_schedule = episode_schedule or None
        self.episode_schedule_cursor = 0
        self.storenet_base_dir = storenet_base_dir
        self.price_profile = price_profile
        self.current_time_index = 0
        _project_root = os.path.dirname(os.path.abspath(__file__))
        self._project_root = _project_root
        if self.storenet_base_dir and (not os.path.isabs(self.storenet_base_dir)):
            self.storenet_base_dir = os.path.normpath(os.path.join(_project_root, self.storenet_base_dir))
        self._load_carbon_intensity_table(_project_root)
        if data_interface is not None:
            self.data_interface = data_interface
        elif self.episode_schedule is not None and len(self.episode_schedule) > 0:
            _house_id, _, _ = self._parse_episode_schedule_item(self.episode_schedule[0])
            self.data_interface = self.data_class.from_storenet_ireland_2020(house_id=_house_id, base_dir=self.storenet_base_dir, price_profile=self.price_profile, steps_per_day=self.steps_per_day, tariff_params=self.tariff_config, ev_schedule_params=self.ev_trip_config)
        else:
            raise ValueError('HomeEnergyManagementEnv 需要 `data_interface=...` 或非空的 `episode_schedule`（StoreNet）。')
        if self.episode_schedule is not None and len(self.episode_schedule) > 0:
            _, _start_date, _ = self._parse_episode_schedule_item(self.episode_schedule[0])
            self.current_time = HomeEnergyManagementEnv._normalize_date_str(_start_date)
        else:
            self.current_time = HomeEnergyManagementEnv._first_cons_date_str(self.data_interface)
        self._seed_value: Optional[int] = None
        self.current_ev_power = 0
        self.current_battery_power = 0
        self.ev_battery_record = []
        self.ess_state_record = []
        self.home_load_record = []
        self.pv_generation_record = []
        self.electricity_price_record = []
        self.ev_at_home_record = []
        self.wash_machine_record = []
        self.air_conditioner_power_record = []
        self.air_conditioner_power_record2 = []
        self.ess_actions = []
        self.wash_machine_actions = []
        self.air_conditioner_actions = []
        self.ess_charge_pv = []
        self.ess_charge_grid = []
        self.ess_discharge_ev = []
        self.ess_discharge_house = []
        self.ess_discharge_grid = []
        self.records = {'timestamps': [], 'ev_soc': [], 'ess_soc': [], 'grid_power': [], 'energy_cost': [], 'daily_costs': [], 'indoor_temp': [], 'indoor_temp2': [], 'outdoor_temp': [], 'current_daily_cost': 0, 'total_load': [], 'user_dissatisfaction': [], 'requested_battery_power': [], 'executed_battery_power': [], 'requested_ev_power': [], 'executed_ev_power': [], 'ev_shield_intervened': [], 'solar_irradiance_norm': [], 'zone1_occupied': [], 'zone2_occupied': [], 'hot_water_draw_l': [], 'hvac_cop': []}
        self.reward_components = {'total': [], 'energy_cost': [], 'violation_penalty': [], 'aging_penalty': [], 'battery_health_penalty': [], 'smoothness_penalty': [], 'invalid_action_penalty': [], 'user_penalty': [], 'temp_reward': [], 'ac_comfort_reward': [], 'hot_water_service_reward': [], 'comfort_bonus': [], 'comfort_penalty': [], 'carbon_penalty': [], 'pv_self_use_reward': [], 'peak_shave_reward': [], 'ev_deadline_reward': [], 'ev_shield_reward': [], 'terminal_energy_reward': [], 'boundary_recovery_reward': [], 'core_reward': [], 'raw_violation_count': [], 'chattering_count': []}
        self.ewh_params = dict(CONFIG['environment']['ewh'])
        self.ewh_temp_record = []
        self.ewh_power_record = []
        self.user_flow_record = []
        self.hot_water_draw_volume_record = []
        self._ewh_draw_profile_l = np.zeros(self.steps_per_day, dtype=np.float64)
        self._ewh_draw_flow_lpm = np.zeros(self.steps_per_day, dtype=np.float64)
        self._ewh_draw_profile_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.current_hvac_cop = self.heating_cop_at_7c
        self.current_step_user_flow = 0.0
        self.current_step_hot_water_draw_l = 0.0
        self.current_step_hot_water_delivery_temp_c = float(self.ewh_params['temp_init'])
        self.wash_machine_power = float(CONFIG['environment']['wash']['machine_power_kw'])
        self.current_period_start = None
        self.has_run_in_current_period = False
        self.time_deviation = 0
        self.wash_machine_state = 0
        self.last_action_time = None
        self.remaining_runtime = 0
        self.wash_machine_scheduled_start_step: Optional[int] = None
        self._wash_price_step_for_penalty: Optional[int] = None
        _cmf = CONFIG['environment']['comfort']
        self.indoor_temp = self._episode_initial_indoor_temp_c
        self.indoor_temp2 = self._episode_initial_indoor_temp2_c
        self.outdoor_temp = float(_cmf['indoor_temp_init'])
        self.user_temp_preference = float(_cmf['user_temp_preference_ac1'])
        self.user_temp_preference2 = float(_cmf['user_temp_preference_ac2'])

    @staticmethod
    def _normalize_date_str(d) -> str:
        if isinstance(d, str):
            s = d.strip()
            return s[:10] if len(s) >= 10 else s
        return pd.Timestamp(d).strftime('%Y-%m-%d')

    @staticmethod
    def _parse_episode_schedule_item(item) -> Tuple[str, str, int]:
        if not isinstance(item, (tuple, list)):
            raise TypeError(f'Invalid episode schedule item: {item!r}')
        if len(item) == 2:
            house_id, start_date = item
            block_days = 1
        elif len(item) == 3:
            house_id, start_date, block_days = item
        else:
            raise ValueError('Episode schedule item must be (house, start_date, block_days)')
        return (str(house_id).strip().upper(), HomeEnergyManagementEnv._normalize_date_str(start_date), max(1, int(block_days)))

    @staticmethod
    def _first_cons_date_str(di: DataInterface) -> str:
        return HomeEnergyManagementEnv._normalize_date_str(di.cons_data.index[0])

    @staticmethod
    def _last_cons_date_str(di: DataInterface) -> str:
        return HomeEnergyManagementEnv._normalize_date_str(di.cons_data.index[-1])

    def _get_future_price_dict(self) -> Dict[str, float]:
        res = {}
        for i in range(1, 5):
            res[f'future_price_{i}'] = self._price_at_offset(i, fallback=self.state.get('electricity_price', 0.1) if hasattr(self, 'state') else 0.1)
        return res

    def _forecast_vectors(self, horizon_steps: Optional[int]=None) -> Dict[str, List[float]]:
        horizon = max(1, int(horizon_steps or self.prior_forecast_horizon_steps))
        cur_price = self._price_at_offset(0)
        cur_load = self._load_at_offset(0)
        cur_pv = self._pv_at_offset(0)
        cur_carbon = self._carbon_intensity_kg_per_kwh_at_offset(0)
        prices, loads, pvs, carbons = ([], [], [], [])
        for offset in range(0, horizon + 1):
            prices.append(self._price_at_offset(offset, fallback=cur_price))
            loads.append(cur_load)
            pvs.append(cur_pv)
            carbons.append(cur_carbon)
        return {'price': prices, 'load': loads, 'pv': pvs, 'carbon': carbons}

    def _forecast_observation_dict(self) -> Dict[str, float]:
        vec = self._forecast_vectors(self.prior_forecast_horizon_steps)
        future_prices = vec['price'][1:] or vec['price'][:1]
        future_loads = vec['load'][1:] or vec['load'][:1]
        future_pvs = vec['pv'][1:] or vec['pv'][:1]
        future_carbons = vec['carbon'][1:] or vec['carbon'][:1]
        price_now = vec['price'][0] if vec['price'] else 0.0
        sorted_prices = sorted(vec['price'])
        if len(sorted_prices) <= 1:
            price_rank = 0.5
        else:
            cheaper_or_equal = sum((1 for p in sorted_prices if p <= price_now))
            price_rank = cheaper_or_equal / float(len(sorted_prices))
        return {'future_price_mean': float(np.mean(future_prices)), 'future_load_mean': float(np.mean(future_loads)), 'future_pv_mean': float(np.mean(future_pvs)), 'future_net_load_mean': float(np.mean(np.asarray(future_loads) - np.asarray(future_pvs))), 'future_carbon_intensity_mean': self._normalized_carbon_value(float(np.mean(future_carbons))), 'price_rank_lookahead': float(np.clip(price_rank, 0.0, 1.0))}

    def _peak_observation_dict(self, grid_import_kw: Optional[float]=None) -> Dict[str, float]:
        target_kw = float(getattr(self, 'current_episode_peak_target_kw', 0.0) or 0.0)
        peak_so_far = float(getattr(self, 'current_episode_peak_import_kw', 0.0) or 0.0)
        if grid_import_kw is None:
            grid_import_kw = float(getattr(self, 'current_step_grid_import_kw', 0.0) or 0.0)
        grid_import_kw = max(0.0, float(grid_import_kw))
        denom = max(1e-08, target_kw)
        pressure_start = 0.88 * target_kw
        pressure = max(0.0, grid_import_kw - pressure_start) / denom if target_kw > 0.0 else 0.0
        return {'episode_peak_import_norm': float(np.clip(peak_so_far / denom, 0.0, 3.0)), 'peak_target_margin_norm': float(np.clip((target_kw - grid_import_kw) / denom, -3.0, 3.0)), 'peak_pressure_norm': float(np.clip(pressure, 0.0, 3.0))}

    def _calendar_observation_dict(self) -> Dict[str, float]:
        dt = datetime.strptime(self.current_time, '%Y-%m-%d')
        wd = int(dt.weekday())
        doy = int(dt.timetuple().tm_yday)
        sin_wd = float(np.sin(2 * np.pi * wd / 7.0))
        cos_wd = float(np.cos(2 * np.pi * wd / 7.0))
        sin_doy = float(np.sin(2 * np.pi * (doy - 1) / 365.25))
        cos_doy = float(np.cos(2 * np.pi * (doy - 1) / 365.25))
        is_weekend = 1.0 if wd >= 5 else 0.0
        ev_home = 1.0 if self.is_ev_at_home() else 0.0
        return {'weekday_sin': sin_wd, 'weekday_cos': cos_wd, 'is_weekend': is_weekend, 'day_of_year_sin': sin_doy, 'day_of_year_cos': cos_doy, 'ev_at_home': ev_home}

    def _generate_ewh_draw_profile(self, date_str: str) -> Tuple[np.ndarray, np.ndarray]:
        key = self._normalize_date_str(date_str)
        cached = self._ewh_draw_profile_cache.get(key)
        if cached is not None:
            return (cached[0].copy(), cached[1].copy())
        params = self.ewh_params
        rng = self.data_interface.scenario_rng(key, 0, salt=320)
        target_l = float(np.clip(rng.normal(float(params.get('daily_mixed_water_mean_l', 65.0)), float(params.get('daily_mixed_water_std_l', 12.0))), float(params.get('daily_mixed_water_min_l', 35.0)), float(params.get('daily_mixed_water_max_l', 95.0))))
        profile = np.zeros(self.steps_per_day, dtype=np.float64)
        flow = np.zeros(self.steps_per_day, dtype=np.float64)
        events: List[Tuple[int, float, float]] = []
        occupants = max(1, int(params.get('occupants', 2)))
        for _ in range(occupants):
            if rng.rand() > float(params.get('shower_probability_per_person', 0.75)):
                continue
            center_hour = 7.25 if rng.rand() < 0.55 else 19.5
            hour = float(np.clip(rng.normal(center_hour, 0.7), 5.5, 22.0))
            step = int(np.clip(round(hour * self.steps_per_day / 24.0), 0, self.steps_per_day - 1))
            volume_l = float(np.clip(rng.normal(float(params.get('shower_volume_mean_l', 28.0)), float(params.get('shower_volume_std_l', 7.0))), 18.0, 48.0))
            events.append((step, volume_l, float(rng.uniform(5.0, 9.0))))
        if not events:
            fallback_step = int(round(7.25 * self.steps_per_day / 24.0))
            events.append((fallback_step, float(params.get('shower_volume_mean_l', 28.0)), 7.0))
        small_count = int(rng.poisson(float(params.get('small_event_count_mean', 4.0))))
        meal_centers = np.asarray([7.5, 13.0, 19.0], dtype=np.float64)
        for _ in range(small_count):
            center = float(rng.choice(meal_centers, p=[0.3, 0.2, 0.5]))
            hour = float(np.clip(rng.normal(center, 0.9), 5.0, 23.0))
            step = int(np.clip(round(hour * self.steps_per_day / 24.0), 0, self.steps_per_day - 1))
            volume_l = float(rng.uniform(float(params.get('small_event_volume_min_l', 1.0)), float(params.get('small_event_volume_max_l', 5.0))))
            events.append((step, volume_l, float(rng.uniform(0.5, 2.0))))
        raw_total = max(1e-08, sum((event[1] for event in events)))
        scale = target_l / raw_total
        for step, volume_l, duration_min in events:
            scaled_l = max(0.0, volume_l * scale)
            profile[step] += scaled_l
            flow[step] = max(flow[step], scaled_l / max(0.25, duration_min))
        result = (profile.astype(np.float64), flow.astype(np.float64))
        self._ewh_draw_profile_cache[key] = (result[0].copy(), result[1].copy())
        return result

    def _hot_water_draw_at_offset(self, offset_steps: int) -> float:
        _date_str, step = self._date_index_with_offset(offset_steps)
        return expected_hot_water_l(step, self.steps_per_day, self.ewh_params['daily_mixed_water_mean_l'])

    def _scenario_observation_dict(self) -> Dict[str, float]:
        occ1, occ2 = self.data_interface.get_zone_occupancy(self.current_time, self.current_time_index)
        horizon = max(1, int(self.ewh_params.get('draw_forecast_horizon_steps', 8)))
        expected_draw_l = sum((self._hot_water_draw_at_offset(offset) for offset in range(horizon)))
        daily_max_l = max(1e-08, float(self.ewh_params.get('daily_mixed_water_max_l', 95.0)))
        return {'solar_irradiance_norm': float(self.data_interface.get_solar_irradiance_norm(self.current_time, self.current_time_index)), 'zone1_occupied': float(np.clip(occ1 / 2.0, 0.0, 1.0)), 'zone2_occupied': float(np.clip(occ2 / 2.0, 0.0, 1.0)), 'hot_water_draw_forecast_norm': float(np.clip(expected_draw_l / daily_max_l, 0.0, 1.0))}

    def _time_index_observation_dict(self) -> Dict[str, float]:
        spd = max(1, int(self.steps_per_day))
        ti = int(self.current_time_index) % spd
        ang = 2.0 * np.pi * float(ti) / float(spd)
        return {'time_index_sin': float(np.sin(ang)), 'time_index_cos': float(np.cos(ang))}

    def _wash_schedule_observation_dict(self) -> Dict[str, float]:
        spd = max(1, int(self.steps_per_day))
        e = self._wash_abs_earliest_start_step()
        l = self._wash_abs_latest_start_step()
        span = max(1, l - e)
        cur = int(self.current_time_index) % spd
        sch = self.wash_machine_scheduled_start_step
        remaining_norm = float(np.clip(float(self.remaining_runtime) / max(1.0, float(self._wash_duration_steps())), 0.0, 1.0))
        if sch is None:
            return {'wash_pending': 0.0, 'wash_planned_start_norm': 0.0, 'wash_steps_until_start_norm': 0.0, 'wash_remaining_runtime_norm': remaining_norm}
        s = int(sch)
        planned_norm = float(np.clip(float(s - e) / float(span), 0.0, 1.0))
        steps_left = max(0, s - cur)
        until_norm = float(np.clip(float(steps_left) / float(spd), 0.0, 1.0))
        return {'wash_pending': 1.0, 'wash_planned_start_norm': planned_norm, 'wash_steps_until_start_norm': until_norm, 'wash_remaining_runtime_norm': remaining_norm}

    def get_state_vector(self, state_dict):
        self._validate_observation_schema(state_dict, context='get_state_vector')
        ordered_keys = sorted(self.state_space.keys())
        return [state_dict[k] for k in ordered_keys]

    def _validate_observation_schema(self, state_dict: Dict, context: str) -> None:
        missing = sorted(set(self.state_space) - set(state_dict))
        if missing:
            raise RuntimeError(f"环境在 {context} 返回了不完整状态，缺少字段: {', '.join(missing)}")

    def _dt_hours(self) -> float:
        return 24.0 / max(1.0, float(self.steps_per_day))

    def _hour_of_step(self, step_index: int) -> int:
        return int(float(step_index) * self._dt_hours() % 24.0)

    @staticmethod
    def _power_sign(power: float) -> int:
        if power > 0.001:
            return 1
        if power < -0.001:
            return -1
        return 0

    def _max_abs_action_power(self, action_name: str) -> float:
        values = list(self.action_space[action_name])
        return max(1e-08, max((abs(float(v)) for v in values)))

    def _battery_motion_observation_dict(self) -> Dict[str, float]:
        ess_max = self._max_abs_action_power('battery_power')
        ev_max = self._max_abs_action_power('ev_power')
        norm_window = float(max(1, self.battery_micro_cycle_window_steps))
        return {'ess_prev_power_norm': float(np.clip(self.prev_battery_power / ess_max, -1.0, 1.0)), 'ev_prev_power_norm': float(np.clip(self.prev_ev_power / ev_max, -1.0, 1.0)), 'ess_power_sign': float(self._power_sign(self.prev_battery_power)), 'ev_power_sign': float(self._power_sign(self.prev_ev_power)), 'ess_steps_since_switch_norm': float(np.clip(self._ess_steps_since_switch / norm_window, 0.0, 1.0)), 'ev_steps_since_switch_norm': float(np.clip(self._ev_steps_since_switch / norm_window, 0.0, 1.0)), 'ess_ramp_norm': float(np.clip(self.current_step_ess_ramp_norm, 0.0, 2.0)), 'ev_ramp_norm': float(np.clip(self.current_step_ev_ramp_norm, 0.0, 2.0))}

    def _motion_cost_for_power(self, current_power: float, previous_power: float, max_power: float, recent_switch_flags: deque, steps_since_switch: int) -> Tuple[float, float, int, float, float, int]:
        if not self.battery_aware_enabled:
            recent_switch_flags.append(0)
            return (0.0, 0.0, 0, 0.0, 0.0, steps_since_switch + 1)
        prev_norm = float(previous_power) / max(1e-08, float(max_power))
        curr_norm = float(current_power) / max(1e-08, float(max_power))
        ramp_norm = abs(curr_norm - prev_norm)
        ramp_excess = max(0.0, ramp_norm - self.battery_ramp_free)
        prev_sign = self._power_sign(previous_power)
        curr_sign = self._power_sign(current_power)
        switched = int(prev_sign != 0 and curr_sign == -prev_sign)
        high_power_switch = float(switched) * abs(prev_norm) * abs(curr_norm)
        recent_switch_flags.append(switched)
        micro_cycle = max(0, int(sum(recent_switch_flags)) - 1)
        next_steps_since_switch = 0 if switched else steps_since_switch + 1
        cost = self.battery_ramp_weight * ramp_excess ** 2 + self.battery_high_power_switch_weight * high_power_switch + self.battery_micro_cycle_weight * float(micro_cycle)
        return (float(cost), float(ramp_norm), switched, float(high_power_switch), float(micro_cycle), next_steps_since_switch)

    def _update_battery_motion_metrics(self, action: Dict[str, float]) -> None:
        ess_cost, ess_ramp, ess_switch, ess_high, ess_micro, ess_since = self._motion_cost_for_power(float(action.get('battery_power', 0.0)), self.prev_battery_power, self._max_abs_action_power('battery_power'), self._ess_recent_switch_flags, self._ess_steps_since_switch)
        ev_cost, ev_ramp, ev_switch, ev_high, ev_micro, ev_since = self._motion_cost_for_power(float(action.get('ev_power', 0.0)), self.prev_ev_power, self._max_abs_action_power('ev_power'), self._ev_recent_switch_flags, self._ev_steps_since_switch)
        self.current_step_ess_ramp_norm = ess_ramp
        self.current_step_ev_ramp_norm = ev_ramp
        self.current_step_ess_switching_cost = ess_cost
        self.current_step_ev_switching_cost = ev_cost
        self.current_step_battery_switching_cost = ess_cost + ev_cost
        self.current_step_high_power_reversal_count = ess_high + ev_high
        self.current_step_micro_cycle_count = ess_micro + ev_micro
        self._ess_steps_since_switch = ess_since
        self._ev_steps_since_switch = ev_since
        self.current_smoothness_penalty += self.current_step_battery_switching_cost

    def _zero_action_index(self, action_name: str) -> int:
        values = list(self.action_space[action_name])
        close_idx = np.where(np.isclose(values, 0.0, atol=1e-06))[0]
        return int(close_idx[0]) if len(close_idx) else 0

    def _ensure_valid_branch(self, action_name: str, mask: List[bool], prefer_charge: bool=False) -> None:
        if any(mask):
            return
        values = list(self.action_space[action_name])
        if prefer_charge:
            positive = [i for i, v in enumerate(values) if float(v) > 0.0]
            if positive:
                mask[min(positive, key=lambda i: abs(float(values[i])))] = True
                return
        mask[self._zero_action_index(action_name)] = True

    def _date_index_with_offset(self, offset_steps: int) -> Tuple[str, int]:
        spd = max(1, int(self.steps_per_day))
        if self._current_block_dates:
            target_step = int(self._current_block_step) + int(offset_steps)
            target_step = int(np.clip(target_step, 0, max(0, int(self.current_episode_length_steps) - 1)))
            day_index = min(len(self._current_block_dates) - 1, target_step // spd)
            return (self._current_block_dates[day_index], target_step % spd)
        absolute_index = int(self.current_time_index) + int(offset_steps)
        base_date = datetime.strptime(self.current_time, '%Y-%m-%d')
        date = base_date + timedelta(days=absolute_index // spd)
        return (date.strftime('%Y-%m-%d'), absolute_index % spd)

    def _next_step_date_index(self) -> Tuple[str, int]:
        return self._date_index_with_offset(1)

    def _price_at_offset(self, offset_steps: int, fallback: Optional[float]=None) -> float:
        try:
            date_str, idx = self._date_index_with_offset(offset_steps)
            return float(self.data_interface.get_electricity_price(date_str, idx))
        except Exception:
            if fallback is not None:
                return float(fallback)
            return float(getattr(self, 'state', {}).get('electricity_price', 0.1))

    def _load_at_offset(self, offset_steps: int, fallback: Optional[float]=None) -> float:
        try:
            date_str, idx = self._date_index_with_offset(offset_steps)
            return float(self.data_interface.get_home_load(date_str, idx))
        except Exception:
            if fallback is not None:
                return float(fallback)
            return float(getattr(self, 'state', {}).get('home_load', 0.0))

    def _pv_at_offset(self, offset_steps: int, fallback: Optional[float]=None) -> float:
        try:
            date_str, idx = self._date_index_with_offset(offset_steps)
            return float(self.data_interface.get_pv_generation(date_str, idx))
        except Exception:
            if fallback is not None:
                return float(fallback)
            return float(getattr(self, 'state', {}).get('pv_generation', 0.0))

    def _normalized_price_value(self, price_raw: float) -> float:
        p_min, p_max = self._tou_price_min_max()
        return float(np.clip((float(price_raw) - p_min) / max(1e-08, p_max - p_min), 0.0, 1.0))

    def _normalized_carbon_value(self, carbon_kg_per_kwh: float) -> float:
        denom = CONFIG['protocol']['information_protocol']['carbon_normalization_kg_per_kwh']
        return float(np.clip(float(carbon_kg_per_kwh) / denom, 0.0, 1.0))

    def _ev_energy_delta_for_power(self, power_kw: float, dt_h: float) -> float:
        power_kw = float(power_kw)
        if power_kw > 0:
            return power_kw * self.charge_efficiency * dt_h
        if power_kw < 0:
            return power_kw / self.discharge_efficiency * dt_h
        return 0.0

    def _price_rank_in_wash_window(self, date_str: str, price: float) -> float:
        try:
            earliest = self._wash_abs_earliest_start_step()
            latest = self._wash_abs_latest_start_step()
            slots = [float(self.data_interface.get_electricity_price(date_str, s)) for s in range(earliest, latest + 1)]
            if not slots:
                return 0.0
            cheaper_or_equal = sum((1 for p in slots if p <= float(price)))
            return float(np.clip(cheaper_or_equal / len(slots), 0.0, 1.0))
        except Exception:
            return 0.0

    def _rank_in_values(self, value: float, values: List[float]) -> float:
        vals = [float(v) for v in values if np.isfinite(float(v))]
        if not vals:
            return 0.5
        cheaper_or_equal = sum((1 for v in vals if v <= float(value)))
        return float(np.clip(cheaper_or_equal / len(vals), 0.0, 1.0))

    def _forecast_context(self, state: Dict) -> Dict[str, float]:
        vec = self._forecast_vectors(self.prior_forecast_horizon_steps)
        prices_norm = [self._normalized_price_value(p) for p in vec['price']]
        carbons_norm = [self._normalized_carbon_value(c) for c in vec['carbon']]
        future_prices = prices_norm[1:] or prices_norm[:1]
        future_carbons = carbons_norm[1:] or carbons_norm[:1]
        base_load_now = float(state.get('home_load', 0.0)) + float(state.get('Air_conditioner_power', 0.0)) + float(state.get('Air_conditioner_power2', 0.0)) + float(state.get('wash_machine_state', 0.0)) * float(self.wash_machine_power) + float(state.get('ewh_power', 0.0)) - float(state.get('pv_generation', 0.0))
        future_net = np.asarray(vec['load'][1:] or vec['load'][:1]) - np.asarray(vec['pv'][1:] or vec['pv'][:1])
        calibrated_peak_target = float(getattr(self, 'current_episode_peak_target_kw', 0.0) or 0.0)
        peak_threshold_candidates = [float(self.prior_peak_soft_threshold_kw)]
        if calibrated_peak_target > 0.0:
            peak_threshold_candidates.append(0.92 * calibrated_peak_target)
        peak_threshold = max(1e-08, min((v for v in peak_threshold_candidates if v > 0.0)))
        future_peak_import = float(max(0.0, np.max(future_net))) if future_net.size else 0.0
        future_peak_score = float(np.clip((future_peak_import - peak_threshold) / peak_threshold, 0.0, 1.0))
        return {'price_now': float(prices_norm[0]), 'price_future_mean': float(np.mean(future_prices)), 'price_future_min': float(np.min(future_prices)), 'price_future_max': float(np.max(future_prices)), 'carbon_now': float(carbons_norm[0]), 'carbon_future_mean': float(np.mean(future_carbons)), 'carbon_future_min': float(np.min(future_carbons)), 'carbon_future_max': float(np.max(future_carbons)), 'pv_surplus_now': float(max(0.0, float(state.get('pv_generation', 0.0)) - float(state.get('home_load', 0.0)))), 'future_net_load_mean': float(np.mean(future_net)), 'future_peak_import': future_peak_import, 'future_peak_score': future_peak_score, 'base_import_now': float(max(0.0, base_load_now)), 'peak_score': float(np.clip((max(0.0, base_load_now) - peak_threshold) / peak_threshold, 0.0, 1.0))}

    def _clip_prior_list(self, values: List[float], branch_name: Optional[str]=None) -> List[float]:
        branch_mult = 1.0
        if branch_name:
            branch_mult = float(self.prior_branch_multipliers.get(branch_name, 1.0))
        arr = np.asarray(values, dtype=np.float64) * float(self.prior_strength) * branch_mult
        arr = np.clip(arr, -abs(self.prior_max_abs_bias), abs(self.prior_max_abs_bias))
        return [float(x) for x in arr]

    def _record_action_prior_stats(self, priors: Dict[str, List[float]]) -> None:
        branch_mean = {}
        all_abs = []
        for name, values in priors.items():
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                continue
            branch_mean[name] = float(np.mean(np.abs(arr)))
            all_abs.extend(np.abs(arr).tolist())
        self.current_action_prior_branch_mean = branch_mean
        self.current_action_prior_abs_mean = float(np.mean(all_abs)) if all_abs else 0.0
        self.current_action_prior_max_abs = float(np.max(all_abs)) if all_abs else 0.0

    def _battery_action_prior(self, state: Dict, ctx: Dict[str, float]) -> List[float]:
        actions = list(self.action_space['battery_power'])
        max_power = max(1e-08, max((abs(float(v)) for v in actions)))
        dt_h = self._dt_hours()
        soc = float(state.get('ess_state', 0.0)) / max(1e-08, float(self.ess_capacity))
        low_risk = float(np.clip((self.mask_free_low - soc) / max(1e-08, self.mask_free_low), 0.0, 1.0))
        high_risk = float(np.clip((soc - self.mask_free_high) / max(1e-08, 1.0 - self.mask_free_high), 0.0, 1.0))
        reserve_soc = float(np.clip(self.prior_ess_reserve_soc, self.mask_warn_low, self.mask_free_high))
        reserve_deficit = float(np.clip((reserve_soc - soc) / max(1e-08, reserve_soc), 0.0, 1.0))
        discharge_floor = float(np.clip(self.prior_ess_peak_discharge_soc_floor, self.mask_warn_low, self.mask_free_high))
        discharge_soc_gate = float(np.clip((soc - discharge_floor) / max(1e-08, self.mask_free_high - discharge_floor), 0.0, 1.0))
        cheap_or_clean_now = max(max(0.0, ctx['price_future_max'] - ctx['price_now']), max(0.0, ctx['carbon_future_max'] - ctx['carbon_now']), np.clip(ctx['pv_surplus_now'] / max_power, 0.0, 1.0))
        charge_value = self.prior_ess_price_weight * max(0.0, ctx['price_future_max'] - ctx['price_now']) + self.prior_ess_carbon_weight * max(0.0, ctx['carbon_future_max'] - ctx['carbon_now']) + self.prior_ess_pv_weight * np.clip(ctx['pv_surplus_now'] / max_power, 0.0, 1.0) + self.prior_ess_reserve_weight * ctx['future_peak_score'] * reserve_deficit * (0.5 + 0.5 * cheap_or_clean_now) - self.prior_ess_edge_soc_weight * high_risk
        discharge_value = self.prior_ess_price_weight * max(0.0, ctx['price_now'] - ctx['price_future_min']) + self.prior_ess_carbon_weight * max(0.0, ctx['carbon_now'] - ctx['carbon_future_min']) + self.prior_ess_peak_weight * ctx['peak_score'] * (0.35 + 0.65 * discharge_soc_gate) - self.prior_ess_edge_soc_weight * low_risk
        prev_sign = self._power_sign(self.prev_battery_power)
        priors = []
        for power in actions:
            power = float(power)
            sign = self._power_sign(power)
            pnorm = abs(power) / max_power
            reversal_cost = self.prior_ess_reversal_weight if prev_sign != 0 and sign == -prev_sign else 0.0
            throughput_cost = 0.25 * pnorm
            edge_cost = (high_risk if power > 0 else low_risk if power < 0 else 0.0) * pnorm
            degradation_cost = self.prior_ess_degradation_weight * throughput_cost + self.prior_ess_edge_soc_weight * edge_cost + reversal_cost
            if power > 0:
                value = charge_value * pnorm - degradation_cost
                if discharge_value > charge_value + self.prior_ess_idle_margin:
                    value += self.prior_dominated_bias * pnorm
            elif power < 0:
                value = discharge_value * pnorm - degradation_cost
                if charge_value > discharge_value + self.prior_ess_idle_margin:
                    value += self.prior_dominated_bias * pnorm
            else:
                best_value = max(charge_value, discharge_value)
                value = self.prior_idle_bias if best_value < self.prior_ess_idle_margin else 0.0
            priors.append(float(value))
        return priors

    def _find_departure_step_within_episode(self, start_step: Optional[int]=None) -> Optional[int]:
        start = int(self.current_time_index if start_step is None else start_step)
        departure_step = self.data_interface.get_ev_departure_step(self.current_time)
        if departure_step is None or int(departure_step) < max(0, start):
            return None
        return int(departure_step)

    def _is_ev_arrival_step(self, step: Optional[int]=None) -> bool:
        idx = int(self.current_time_index if step is None else step)
        if idx <= 0:
            return False
        return not self.data_interface.is_ev_at_home(self.current_time, idx - 1) and self.data_interface.is_ev_at_home(self.current_time, idx)

    def _trip_kwh_for_date(self, date_str: str) -> float:
        if self.data_interface.get_ev_departure_step(date_str) is None:
            return 0.0
        trip_km = self.data_interface.get_daily_ev_trip_km(date_str, self.ev_daily_km_mean, self.ev_daily_km_sigma)
        energy_per_km = float(self.ev_trip_config.get('energy_consumption_per_km', 0.2))
        return float(np.clip(trip_km * energy_per_km, 0.0, max(0.0, self.ev_capacity - self.ev_min_charge)))

    def _register_daily_ev_plan(self) -> None:
        self.declared_ev_plan = {'date': self.current_time, 'departure_step': self.data_interface.get_ev_departure_step(self.current_time), 'arrival_step': self.data_interface.get_ev_arrival_step(self.current_time), 'trip_kwh': float(self._episode_trip_kwh), 'information_source': 'user_declared_daily_plan_zero_error'}

    def _ev_operational_reserve_kwh(self) -> float:
        departure_step = self.data_interface.get_ev_departure_step(self.current_time)
        if departure_step is None:
            return float(self.ev_no_trip_reserve_kwh)
        return float(self.ev_min_charge)

    def _ev_deadline_target_kwh(self, hard: bool=False, trip_kwh: Optional[float]=None) -> float:
        margin = self.ev_deadline_hard_target_margin_kwh if hard else self.ev_deadline_target_margin_kwh
        uncertainty = 0.0 if hard else self.ev_deadline_uncertainty_margin_kwh
        if trip_kwh is None:
            trip_kwh = float(self.declared_ev_plan['trip_kwh'])
        target = self.ev_min_charge + max(0.0, float(trip_kwh)) + uncertainty + max(0.0, margin)
        return float(np.clip(target, 0.0, self.ev_capacity))

    def _terminal_ev_reference_kwh(self) -> float:
        mobility_adjusted_inventory = max(0.0, float(self._episode_initial_ev_kwh) - float(self._episode_trip_kwh))
        return float(np.clip(max(mobility_adjusted_inventory, self._ev_operational_reserve_kwh()), 0.0, self.ev_capacity))

    def _ev_charging_deadline_context(self) -> Dict[str, float]:
        at_home = bool(self.is_ev_at_home())
        if not at_home:
            return {'active': 0.0, 'deadline_step': -1.0, 'trip_kwh': 0.0, 'actual_departure': 0.0}
        departure_step = self._find_departure_step_within_episode()
        if departure_step is not None:
            return {'active': 1.0, 'deadline_step': float(departure_step), 'trip_kwh': float(self.declared_ev_plan['trip_kwh']), 'actual_departure': 1.0}
        return {'active': 0.0, 'deadline_step': -1.0, 'trip_kwh': 0.0, 'actual_departure': 0.0}

    def _ev_deadline_metrics(self, state: Dict, ctx: Optional[Dict[str, float]]=None) -> Dict[str, float]:
        dt_h = self._dt_hours()
        actions = list(self.action_space['ev_power'])
        max_charge_kw = max(0.0, max((float(v) for v in actions)))
        deadline = self._ev_charging_deadline_context()
        departure_step = int(deadline['deadline_step']) if deadline['active'] > 0.5 else None
        at_home = bool(deadline['active'] > 0.5)
        steps_remaining = max(0, int(departure_step) - int(self.current_time_index) + 1) if at_home and departure_step is not None else 0
        ev_kwh = float(state.get('ev_battery_state', 0.0))
        target_kwh = self._ev_deadline_target_kwh(hard=False, trip_kwh=float(deadline['trip_kwh']))
        hard_target_kwh = self._ev_deadline_target_kwh(hard=True, trip_kwh=float(deadline['trip_kwh']))
        has_daily_deadline = at_home and departure_step is not None
        gap_kwh = max(0.0, target_kwh - ev_kwh) if has_daily_deadline else 0.0
        hard_gap_kwh = max(0.0, hard_target_kwh - ev_kwh) if has_daily_deadline else 0.0
        max_reachable_kwh = steps_remaining * max_charge_kw * self.charge_efficiency * dt_h
        required_kw = gap_kwh / max(1e-08, steps_remaining * dt_h * self.charge_efficiency) if steps_remaining > 0 and gap_kwh > 0.0 else 0.0
        raw_risk = gap_kwh / max(1e-08, max_reachable_kwh) if gap_kwh > 0.0 else 0.0
        risk = float(np.clip(raw_risk, 0.0, 1.0))
        risk_gate = float(np.clip((risk - self.ev_deadline_risk_start) / max(1e-08, self.ev_deadline_risk_full - self.ev_deadline_risk_start), 0.0, 1.0))
        if ctx is None:
            ctx = self._forecast_context(state)
        max_power = max(1e-08, max((abs(float(v)) for v in actions)))
        pv_relief = float(np.clip(ctx.get('pv_surplus_now', 0.0) / max_power, 0.0, 1.0))
        business_penalty = self.ev_deadline_price_weight * float(ctx.get('price_now', 0.0)) + self.ev_deadline_carbon_weight * float(ctx.get('carbon_now', 0.0)) + self.ev_deadline_peak_weight * float(ctx.get('peak_score', 0.0)) - self.ev_deadline_pv_relief_weight * pv_relief
        business_penalty = float(np.clip(business_penalty, 0.0, 1.5))
        business_multiplier = float(np.clip(1.0 - self.ev_deadline_business_penalty_weight * business_penalty, 0.25, 1.2))
        if risk_gate >= 0.8:
            business_multiplier = max(business_multiplier, self.ev_deadline_urgent_business_floor, 0.95)
        return {'target_kwh': target_kwh, 'hard_target_kwh': hard_target_kwh, 'gap_kwh': gap_kwh, 'hard_gap_kwh': hard_gap_kwh, 'risk': risk, 'risk_gate': risk_gate, 'steps_remaining': float(steps_remaining), 'required_kw': float(required_kw), 'max_reachable_kwh': float(max_reachable_kwh), 'business_penalty': business_penalty, 'business_multiplier': business_multiplier, 'departure_step': float(departure_step) if departure_step is not None else -1.0, 'actual_departure': float(deadline['actual_departure']), 'deadline_trip_kwh': float(deadline['trip_kwh'])}

    def _ev_deadline_observation_dict(self, state: Optional[Dict]=None) -> Dict[str, float]:
        if state is None:
            state = getattr(self, 'state', {})
        metrics = self._ev_deadline_metrics(state)
        max_ev_power = self._max_abs_action_power('ev_power')
        return {'ev_steps_to_departure_norm': float(np.clip(metrics['steps_remaining'] / max(1.0, float(self.steps_per_day)), 0.0, 1.0)), 'ev_deadline_gap_norm': float(np.clip(metrics['gap_kwh'] / max(1e-08, float(self.ev_capacity)), 0.0, 1.0)), 'ev_required_charge_norm': float(np.clip(metrics['required_kw'] / max_ev_power, 0.0, 1.0))}

    def _boundary_gap_metrics(self, state: Optional[Dict]=None) -> Dict[str, float]:
        state = state or getattr(self, 'state', {})
        ess_target = float(getattr(self, '_episode_initial_ess_kwh', 0.0))
        ev_target = max(float(self._ev_operational_reserve_kwh()), float(getattr(self, '_episode_initial_ev_kwh', 0.0)) - float(getattr(self, '_block_trip_energy_consumed_kwh', 0.0)))
        ewh_target = float(getattr(self, '_episode_initial_ewh_temp_c', self.boundary_ewh_reference_temp_c))
        zone1_target = float(getattr(self, '_episode_initial_indoor_temp_c', self.boundary_hvac_reference_c[0]))
        zone2_target = float(getattr(self, '_episode_initial_indoor_temp2_c', self.boundary_hvac_reference_c[1]))
        ess_gap_kwh = max(0.0, ess_target - float(state.get('ess_state', ess_target)))
        ev_gap_kwh = max(0.0, ev_target - float(state.get('ev_battery_state', ev_target)))
        ewh_gap_kwh_th = max(0.0, float(self.ewh_params['mass']) * 4180.0 * (ewh_target - float(state.get('ewh_temp', ewh_target))) / 3600000.0)
        hvac_zone1_gap_kwh_th = max(0.0, self.zone_thermal_capacity_kwh_per_c[0] * (zone1_target - float(state.get('indoor_temp', zone1_target))))
        hvac_zone2_gap_kwh_th = max(0.0, self.zone_thermal_capacity_kwh_per_c[1] * (zone2_target - float(state.get('indoor_temp2', zone2_target))))
        hvac_gap_kwh_th = hvac_zone1_gap_kwh_th + hvac_zone2_gap_kwh_th
        ewh_efficiency = max(1e-08, float(self.ewh_params.get('heater_efficiency', 0.98)))
        recovery_cop = max(1.0, self._heating_cop(float(getattr(self, 'outdoor_temp', 7.0))))
        grid_equivalent_kwh = ess_gap_kwh / max(1e-08, self.charge_efficiency) + ev_gap_kwh / max(1e-08, self.charge_efficiency) + ewh_gap_kwh_th / ewh_efficiency + hvac_gap_kwh_th / recovery_cop
        return {'ess_target_kwh': float(ess_target), 'ev_target_kwh': float(ev_target), 'ess_gap_kwh': float(ess_gap_kwh), 'ev_gap_kwh': float(ev_gap_kwh), 'ewh_gap_kwh_th': float(ewh_gap_kwh_th), 'hvac_zone1_gap_kwh_th': float(hvac_zone1_gap_kwh_th), 'hvac_zone2_gap_kwh_th': float(hvac_zone2_gap_kwh_th), 'hvac_gap_kwh_th': float(hvac_gap_kwh_th), 'grid_equivalent_kwh': float(grid_equivalent_kwh)}

    def _boundary_observation_dict(self, state: Optional[Dict]=None) -> Dict[str, float]:
        state = state or getattr(self, 'state', {})
        total_steps = max(1, int(getattr(self, 'current_episode_length_steps', self.episode_length_steps)))
        block_step = int(getattr(self, '_current_block_step', 0))
        steps_remaining = max(1, total_steps - block_step)
        last_step = max(1, total_steps - 1)
        gap_metrics = self._boundary_gap_metrics(state)
        ess_gap_kwh = float(gap_metrics['ess_gap_kwh'])
        ev_gap_kwh = float(gap_metrics['ev_gap_kwh'])
        ewh_gap_kwh_th = float(gap_metrics['ewh_gap_kwh_th'])
        hvac_gap_kwh_th = float(gap_metrics['hvac_gap_kwh_th'])
        dt_h = self._dt_hours()
        ess_recoverable = float(CONFIG['environment']['battery']['ess_max_charge_power']) * self.charge_efficiency * dt_h * steps_remaining
        ev_recoverable = float(CONFIG['environment']['ev']['ev_max_charge_power']) * self.charge_efficiency * dt_h * steps_remaining
        ewh_recoverable = float(self.ewh_params.get('heater_max_power_kw', 1.8)) * float(self.ewh_params.get('heater_efficiency', 0.98)) * dt_h * steps_remaining
        hvac_recoverable = self.hvac_shared_max_power_kw * max(1.0, self._heating_cop(float(getattr(self, 'outdoor_temp', 7.0)))) * dt_h * steps_remaining
        recovery_risk = max(ess_gap_kwh / max(1e-08, ess_recoverable), ev_gap_kwh / max(1e-08, ev_recoverable), ewh_gap_kwh_th / max(1e-08, ewh_recoverable), hvac_gap_kwh_th / max(1e-08, hvac_recoverable))
        reserve = float(self._ev_operational_reserve_kwh())
        return {'ev_operational_reserve_norm': float(np.clip(reserve / self.ev_capacity, 0.0, 1.0)), 'block_day_index_norm': float(np.clip(self._current_block_day_index / max(1, self._current_block_days - 1), 0.0, 1.0)), 'block_length_days_norm': float(np.clip(self._current_block_days / max(1, self.block_max_days), 0.0, 1.0)), 'block_step_norm': float(np.clip(block_step / last_step, 0.0, 1.0)), 'steps_to_block_end_norm': float(np.clip((total_steps - 1 - block_step) / last_step, 0.0, 1.0)), 'is_day_boundary': float(int(self.current_time_index) == self.steps_per_day - 1), 'ess_boundary_gap_norm': float(np.clip(ess_gap_kwh / max(1e-08, self.ess_capacity), 0.0, 1.0)), 'ev_boundary_gap_norm': float(np.clip(ev_gap_kwh / max(1e-08, self.ev_capacity), 0.0, 1.0)), 'ewh_boundary_gap_norm': float(np.clip(ewh_gap_kwh_th / max(1e-08, self.ewh_params['mass'] * 4180.0 * 35.0 / 3600000.0), 0.0, 1.0)), 'hvac_boundary_gap_norm': float(np.clip(hvac_gap_kwh_th / max(1e-08, sum(self.zone_thermal_capacity_kwh_per_c) * 8.0), 0.0, 1.0)), 'boundary_recovery_risk': float(np.clip(recovery_risk, 0.0, 1.0))}

    def _boundary_recovery_penalty(self, state: Dict) -> float:
        if not self.boundary_recovery_enabled or self.w_boundary_recovery <= 0.0:
            return 0.0
        obs = self._boundary_observation_dict(state)
        risk = float(obs['boundary_recovery_risk'])
        gate = float(np.clip((risk - self.boundary_risk_start) / max(1e-08, 1.0 - self.boundary_risk_start), 0.0, 1.0))
        physical_gap = self._boundary_gap_metrics(state)
        grid_equivalent_kwh = float(physical_gap['grid_equivalent_kwh'])
        if self.is_terminal_state() and grid_equivalent_kwh > 0.0:
            gate = 1.0
            return float(max(0.0, self.boundary_terminal_multiplier * grid_equivalent_kwh))
        total_steps = max(1, int(getattr(self, 'current_episode_length_steps', self.episode_length_steps)))
        block_step = int(getattr(self, '_current_block_step', 0))
        future_action_steps = max(1, total_steps - 1 - block_step)
        required_recovery_kwh_per_step = grid_equivalent_kwh / future_action_steps
        return float(max(0.0, gate * required_recovery_kwh_per_step))

    def _ev_service_action_context(self, state: Dict) -> Dict[str, float]:
        ev_kwh = float(state['ev_battery_state'])
        dt_h = self._dt_hours()
        max_charge_power = max(0.0, max(map(float, self.action_space['ev_power'])))
        max_discharge_power = max(0.0, -min(map(float, self.action_space['ev_power'])))
        max_charge_now = min(max_charge_power, max(0.0, (self.ev_capacity - ev_kwh) / (dt_h * self.charge_efficiency)))
        max_discharge_now = min(max_discharge_power, max(0.0, ev_kwh * self.discharge_efficiency / dt_h))
        metrics = self._ev_deadline_metrics(state)
        steps_remaining = int(metrics['steps_remaining'])
        reserve = self._ev_operational_reserve_kwh()
        required_next = reserve
        if steps_remaining > 0:
            future_charge = max(0, steps_remaining - 1) * max_charge_power * self.charge_efficiency * dt_h
            required_next = max(required_next, float(metrics['hard_target_kwh']) - future_charge)
        required_delta = required_next - ev_kwh
        minimum_power = required_delta / (dt_h * self.charge_efficiency) if required_delta >= 0.0 else required_delta * self.discharge_efficiency / dt_h
        minimum_power = max(-max_discharge_now, minimum_power)
        return {'active_task': float(steps_remaining > 0), 'steps_remaining': float(steps_remaining), 'target_kwh': float(metrics['target_kwh']) if steps_remaining > 0 else 0.0, 'hard_target_kwh': float(metrics['hard_target_kwh']) if steps_remaining > 0 else 0.0, 'reserve_kwh': float(reserve), 'required_next_kwh': float(required_next), 'min_power_kw': float(minimum_power), 'max_charge_kw': float(max_charge_now), 'recovery_unreachable': float(minimum_power > max_charge_now + 1e-08), 'preexisting_reserve_gap_kwh': float(max(0.0, reserve - ev_kwh))}

    def _ev_power_keeps_deadline_reachable(self, state: Dict, power_kw: float) -> bool:
        if not self.ev_shield_enabled or not self.ev_deadline_enabled:
            return True
        if not self.data_interface.is_ev_at_home(self.current_time, self.current_time_index):
            return abs(float(power_kw)) <= 1e-08
        context = self._ev_service_action_context(state)
        return float(context['min_power_kw']) - 1e-08 <= float(power_kw) <= float(context['max_charge_kw']) + 1e-08

    def _apply_ev_viability_shield(self, state: Dict, requested_power_kw: float) -> float:
        requested = float(requested_power_kw)
        self.current_ev_shield_intervened = 0
        self.current_ev_shield_action_delta_kw = 0.0
        self.current_ev_forced_charge_kwh = 0.0
        self.current_ev_service_infeasible = 0
        self.current_ev_service_context = {}
        if not self.ev_shield_enabled or not self.ev_deadline_enabled:
            return requested
        if not self.data_interface.is_ev_at_home(self.current_time, self.current_time_index):
            return 0.0
        context = self._ev_service_action_context(state)
        self.current_ev_service_context = dict(context)
        self.current_ev_service_infeasible = int(context['recovery_unreachable'])
        dt_h = self._dt_hours()
        executed = float(context['max_charge_kw']) if self.current_ev_service_infeasible else float(np.clip(requested, context['min_power_kw'], context['max_charge_kw']))
        self.current_ev_shield_intervened = int(abs(executed - requested) > 1e-08)
        self.current_ev_shield_action_delta_kw = float(executed - requested)
        self.current_ev_forced_charge_kwh = max(0.0, max(0.0, self._ev_energy_delta_for_power(executed, dt_h)) - max(0.0, self._ev_energy_delta_for_power(requested, dt_h)))
        return float(executed)

    def _terminal_energy_penalty(self, next_state: Dict) -> Tuple[float, float, float]:
        if not self.terminal_energy_enabled or not self.is_terminal_state():
            return (0.0, 0.0, 0.0)
        ess_end = float(next_state.get('ess_state', 0.0))
        ess_deficit = max(0.0, float(self._episode_initial_ess_kwh) - ess_end)
        ev_reference = self._terminal_ev_reference_kwh()
        ev_end = float(next_state.get('ev_battery_state', 0.0))
        ev_deficit = max(0.0, ev_reference - ev_end)
        grid_equivalent_kwh = ess_deficit / max(1e-08, self.charge_efficiency) + ev_deficit / max(1e-08, self.charge_efficiency)
        ewh_deficit_kwh_th = 0.0
        if self.terminal_ewh_enabled:
            ewh_end_temp = float(next_state.get('ewh_temp', self._episode_initial_ewh_temp_c))
            ewh_reference_temp = max(float(self._episode_initial_ewh_temp_c), float(self.terminal_ewh_reference_temp_c))
            ewh_deficit_kwh_th = max(0.0, float(self.ewh_params['mass']) * 4180.0 * (ewh_reference_temp - ewh_end_temp) / 3600000.0)
            heater_efficiency = max(1e-08, float(self.ewh_params.get('heater_efficiency', 0.95)))
            grid_equivalent_kwh += ewh_deficit_kwh_th / heater_efficiency
        hvac_deficit_kwh_th = 0.0
        if self.terminal_hvac_enabled:
            end_temps = (float(next_state.get('indoor_temp', self._episode_initial_indoor_temp_c)), float(next_state.get('indoor_temp2', self._episode_initial_indoor_temp2_c)))
            ref_temps = (float(self._episode_initial_indoor_temp_c), float(self._episode_initial_indoor_temp2_c))
            replacement_cop = max(1.0, self._heating_cop(float(self.outdoor_temp)))
            for end_temp, ref_temp, capacity in zip(end_temps, ref_temps, self.zone_thermal_capacity_kwh_per_c):
                zone_deficit = max(0.0, float(capacity) * (ref_temp - end_temp))
                hvac_deficit_kwh_th += zone_deficit
                grid_equivalent_kwh += zone_deficit / replacement_cop
        price = max(0.0, float(self._episode_mean_price))
        carbon_intensity = max(0.0, float(np.mean([self._carbon_intensity_kg_per_kwh_at_offset(-lag) for lag in range(self.current_time_index + 1)])))
        settlement_cost = grid_equivalent_kwh * price
        settlement_carbon = grid_equivalent_kwh * carbon_intensity
        settlement_cost_norm = self.reward_scaler.scale('cost', settlement_cost)
        settlement_carbon_norm = self.reward_scaler.scale('carbon', settlement_carbon)
        penalty = max(0.0, self.w_cost * settlement_cost_norm + self.w_carbon * settlement_carbon_norm)
        self.current_terminal_ewh_deficit_kwh_th = float(ewh_deficit_kwh_th)
        self.current_terminal_hvac_deficit_kwh_th = float(hvac_deficit_kwh_th)
        self.current_terminal_grid_equivalent_kwh = float(grid_equivalent_kwh)
        self.current_terminal_settlement_cost_eur = float(settlement_cost)
        self.current_terminal_settlement_carbon_kg = float(settlement_carbon)
        return (float(penalty), float(ess_deficit), float(ev_deficit))

    def _ev_action_prior(self, state: Dict, ctx: Dict[str, float]) -> List[float]:
        actions = list(self.action_space['ev_power'])
        max_power = max(1e-08, max((abs(float(v)) for v in actions)))
        dt_h = self._dt_hours()
        ev_kwh = float(state.get('ev_battery_state', 0.0))
        deadline = self._ev_deadline_metrics(state, ctx)
        gap_kwh = float(deadline['gap_kwh'])
        risk = float(deadline['risk'])
        risk_gate = float(deadline['risk_gate'])
        required_kw = float(deadline['required_kw'])
        business_multiplier = float(deadline['business_multiplier'])
        reserve_kwh = min(float(self.ev_capacity), float(self._ev_operational_reserve_kwh() + self.prior_ev_reserve_margin_kwh))
        cheap_charge_value = max(0.0, ctx['price_future_max'] - ctx['price_now'])
        v2h_value = max(0.0, ctx['price_now'] - ctx['price_future_min']) + ctx['peak_score']
        prev_sign = self._power_sign(self.prev_ev_power)
        priors = []
        for power in actions:
            power = float(power)
            sign = self._power_sign(power)
            pnorm = abs(power) / max_power
            next_kwh = ev_kwh + self._ev_energy_delta_for_power(power, dt_h)
            reversal_cost = 0.5 * self.prior_ev_degradation_weight if prev_sign != 0 and sign == -prev_sign else 0.0
            degradation_cost = 0.2 * self.prior_ev_degradation_weight * pnorm + reversal_cost
            if not self.is_ev_at_home():
                priors.append(0.0 if power == 0 else self.prior_dominated_bias)
                continue
            if gap_kwh > 0.0 and risk_gate > 0.0:
                if power > 0:
                    progress = min(1.0, max(0.0, self._ev_energy_delta_for_power(power, dt_h)) / max(1e-08, gap_kwh))
                    rate_score = min(1.0, power / max(1e-08, required_kw)) if required_kw > 1e-08 else 1.0
                    value = self.prior_ev_deadline_weight * (0.45 + risk_gate) * business_multiplier * (0.55 * progress + 0.45 * rate_score)
                    if required_kw > 1e-08 and power < required_kw:
                        rate_gap = np.clip(1.0 - power / required_kw, 0.0, 1.0)
                        value -= self.prior_ev_deadline_weight * risk_gate * (1.0 + risk_gate) * rate_gap
                elif power < 0:
                    value = -self.prior_ev_mobility_risk_weight * (1.0 + risk + risk_gate) * pnorm
                else:
                    value = -self.prior_ev_deadline_weight * risk_gate * (1.0 + 0.75 * risk_gate)
            elif power > 0:
                pv_relief = np.clip(ctx.get('pv_surplus_now', 0.0) / max_power, 0.0, 1.0)
                non_urgent_charge_value = self.prior_ev_cheap_charge_weight * cheap_charge_value + self.ev_deadline_pv_relief_weight * pv_relief - self.prior_ev_deadline_business_weight * deadline['business_penalty']
                value = non_urgent_charge_value * pnorm
            elif power < 0:
                if next_kwh >= reserve_kwh:
                    value = self.prior_ev_v2h_weight * v2h_value * pnorm
                else:
                    deficit = (reserve_kwh - next_kwh) / max(1e-08, reserve_kwh)
                    value = -self.prior_ev_mobility_risk_weight * np.clip(deficit, 0.0, 1.0)
            else:
                value = self.prior_idle_bias if max(cheap_charge_value, v2h_value) < 0.15 else 0.0
            priors.append(float(value - degradation_cost))
        return priors

    def _wash_action_prior(self, state: Dict) -> List[float]:
        actions = list(self.action_space['wash_machine_schedule'])
        ti = int(state.get('time_index', 0))
        earliest = self._wash_abs_earliest_start_step()
        latest = self._wash_abs_latest_start_step()
        span = max(1, latest - earliest)
        preferred_mid = 0.5 * (earliest + latest)
        if self.has_run_in_current_period or self.remaining_runtime > 0:
            return [self.prior_idle_bias if int(v) == 0 else self.prior_dominated_bias for v in actions]
        window_prices = [float(self.data_interface.get_electricity_price(self.current_time, s)) for s in range(earliest, latest + 1)]
        current_carbon = self._carbon_intensity_kg_per_kwh_at_offset(0)
        window_carbons = [current_carbon] * len(window_prices)
        urgency = float(np.clip((ti - earliest) / max(1, latest - earliest), 0.0, 1.0))
        priors = []
        for action_value in actions:
            v = int(action_value)
            if v == 0:
                value = self.prior_idle_bias - self.prior_wash_deadline_weight * urgency ** 2
                priors.append(float(value))
                continue
            target_step = earliest + v - 1
            if target_step < ti or target_step < earliest or target_step > latest:
                priors.append(self.prior_dominated_bias)
                continue
            target_price = float(self.data_interface.get_electricity_price(self.current_time, target_step))
            target_carbon = current_carbon
            price_rank = self._rank_in_values(target_price, window_prices)
            carbon_rank = self._rank_in_values(target_carbon, window_carbons)
            deviation = abs(float(target_step) - preferred_mid) / float(span)
            value = self.prior_wash_price_weight * (1.0 - price_rank) + self.prior_wash_carbon_weight * (1.0 - carbon_rank) - self.prior_wash_deviation_weight * deviation
            priors.append(float(value))
        return priors

    def _ac_action_prior(self, state: Dict, branch_name: str, indoor_key: str, user_target: float) -> List[float]:
        actions = list(self.action_space[branch_name])
        indoor = float(state.get(indoor_key, user_target))
        price_norm = self._normalized_price(state)
        zone_idx = 0 if branch_name == 'Air_conditioner_set_temp' else 1
        occupancy = float(state.get(f'zone{zone_idx + 1}_occupied', 0.0))
        lower, upper = self.zone_occupied_bounds_c[zone_idx]
        target = 0.5 * (lower + upper)
        temp_span = max(1.0, float(max(actions) - min(actions)))
        priors = []
        for set_temp in actions:
            set_temp = float(set_temp)
            if occupancy > 0.0:
                comfort_loss = max(0.0, lower - set_temp, set_temp - upper) / temp_span
                if indoor < lower and set_temp <= indoor:
                    comfort_loss += (lower - set_temp) / temp_span
                if indoor > upper and set_temp >= indoor:
                    comfort_loss += (set_temp - upper) / temp_span
                target_loss = 0.2 * abs(set_temp - target) / temp_span
            else:
                comfort_loss = 0.0
                target_loss = 0.0
            energy_saving = 1.0 - min(1.0, abs(set_temp - indoor) / temp_span)
            value = -self.prior_thermal_comfort_weight * occupancy * (comfort_loss + target_loss) + self.prior_thermal_econ_weight * (0.5 + 0.5 * price_norm) * energy_saving
            priors.append(float(value))
        return priors

    def _ewh_action_prior(self, state: Dict) -> List[float]:
        actions = list(self.action_space['ewh_set_temp'])
        upcoming_draw = float(np.clip(state.get('hot_water_draw_forecast_norm', 0.0), 0.0, 1.0))
        service_target = 45.0 if upcoming_draw > 0.03 else 42.0
        current_temp = float(state.get('ewh_temp', service_target))
        price_norm = self._normalized_price(state)
        temp_span = max(1.0, float(max(actions) - min(actions)))
        priors = []
        for set_temp in actions:
            set_temp = float(set_temp)
            if set_temp <= 0.0:
                service_loss = upcoming_draw * max(0.0, service_target - current_temp) / 10.0
                energy_saving = 1.0
                preheat = 0.0
            else:
                service_loss = upcoming_draw * abs(set_temp - service_target) / temp_span
                if current_temp < 42.0 and set_temp < 42.0:
                    service_loss += (42.0 - set_temp) / temp_span
                energy_saving = (max(actions) - set_temp) / temp_span
                preheat = (1.0 - price_norm) * upcoming_draw * max(0.0, set_temp - service_target) / temp_span
            value = -self.prior_thermal_comfort_weight * service_loss + self.prior_thermal_econ_weight * price_norm * energy_saving + self.prior_ewh_preheat_weight * preheat
            priors.append(float(value))
        return priors

    def _apply_structured_action_prior(self, state: Dict, masks: Dict[str, List[bool]], penalties: Dict[str, List[float]]) -> None:
        if not self.action_prior_enabled:
            self._record_action_prior_stats({})
            return
        ctx = self._forecast_context(state)
        priors = {'battery_power': self._battery_action_prior(state, ctx), 'ev_power': self._ev_action_prior(state, ctx), 'wash_machine_schedule': self._wash_action_prior(state), 'Air_conditioner_set_temp': self._ac_action_prior(state, 'Air_conditioner_set_temp', 'indoor_temp', self.user_temp_preference), 'Air_conditioner_set_temp2': self._ac_action_prior(state, 'Air_conditioner_set_temp2', 'indoor_temp2', self.user_temp_preference2), 'ewh_set_temp': self._ewh_action_prior(state)}
        clipped_priors = {}
        for name, values in priors.items():
            clipped = self._clip_prior_list(values, branch_name=name)
            clipped_priors[name] = clipped
            if name not in penalties:
                penalties[name] = [0.0] * len(clipped)
            for idx, val in enumerate(clipped):
                if idx < len(penalties[name]):
                    penalties[name][idx] += float(val)
        self._record_action_prior_stats(clipped_priors)

    def get_action_mask(self, state, apply_soc_mask: bool=True, mask_mode: Optional[str]=None):
        mode = str(mask_mode or getattr(self, 'mask_mode', 'full') or 'full').lower()
        if not apply_soc_mask:
            mode = 'none'
        if mode in {'none', 'off', 'unmasked'}:
            masks = {name: [True] * len(values) for name, values in self.action_space.items()}
            penalties = {name: [0.0] * len(values) for name, values in self.action_space.items()}
            self._record_action_prior_stats({})
            return (masks, penalties)
        if mode in {'physical', 'physics', 'pure_physical'}:
            return self._get_physical_action_mask(state)
        if mode in {'hcr', 'hcr_viability', 'physics_temporal'}:
            return self._get_hcr_action_mask(state)
        if mode in {'minimal', 'light', 'shield', 'safety_shield'}:
            return self._get_minimal_action_mask(state)
        masks_n_penalties = self._build_soc_masks(state, apply_soc_mask=apply_soc_mask)
        battery_mask = masks_n_penalties['battery']
        ev_mask = masks_n_penalties['ev']
        masks = {'battery_power': battery_mask, 'ev_power': ev_mask}
        penalties = {'battery_power': masks_n_penalties.get('battery_penalty', [0.0] * len(battery_mask)), 'ev_power': masks_n_penalties.get('ev_penalty', [0.0] * len(ev_mask))}
        ti = int(state.get('time_index', 0))
        e = self._wash_abs_earliest_start_step()
        l = self._wash_abs_latest_start_step()
        masks['wash_machine_schedule'] = []
        for val in self.action_space['wash_machine_schedule']:
            v = int(val)
            if v == 0:
                masks['wash_machine_schedule'].append(True)
            else:
                tgt = e + v - 1
                masks['wash_machine_schedule'].append(bool(e <= tgt <= l and tgt >= ti))
        ac_mask1 = [True] * len(self.action_space['Air_conditioner_set_temp'])
        ac_mask2 = [True] * len(self.action_space['Air_conditioner_set_temp2'])
        indoor_temp1 = state.get('indoor_temp', 25)
        indoor_temp2 = state.get('indoor_temp2', 25)
        safety_low, safety_high = self.occupied_safety_bounds_c
        if indoor_temp1 >= safety_high:
            for i, val in enumerate(self.action_space['Air_conditioner_set_temp']):
                if val > 24:
                    ac_mask1[i] = False
        elif indoor_temp1 <= safety_low:
            for i, val in enumerate(self.action_space['Air_conditioner_set_temp']):
                if val < 20:
                    ac_mask1[i] = False
        if indoor_temp2 >= safety_high:
            for i, val in enumerate(self.action_space['Air_conditioner_set_temp2']):
                if val > 24:
                    ac_mask2[i] = False
        elif indoor_temp2 <= safety_low:
            for i, val in enumerate(self.action_space['Air_conditioner_set_temp2']):
                if val < 20:
                    ac_mask2[i] = False
        masks['Air_conditioner_set_temp'] = ac_mask1
        masks['Air_conditioner_set_temp2'] = ac_mask2
        masks['ewh_set_temp'] = [True] * len(self.action_space['ewh_set_temp'])
        if apply_soc_mask:
            self._apply_structured_action_prior(state, masks, penalties)
        else:
            self._record_action_prior_stats({})
        self._ensure_valid_branch('battery_power', masks['battery_power'])
        self._ensure_valid_branch('ev_power', masks['ev_power'], prefer_charge=bool(apply_soc_mask and self.is_ev_at_home()))
        return (masks, penalties)

    def _get_hcr_action_mask(self, state):
        masks, penalties = self._get_minimal_action_mask(state)
        ev_mask = list(masks['ev_power'])
        physically_valid_ev = list(ev_mask)
        for idx, power_kw in enumerate(self.action_space['ev_power']):
            if ev_mask[idx] and (not self._ev_power_keeps_deadline_reachable(state, float(power_kw))):
                ev_mask[idx] = False
        if not any(ev_mask):
            candidates = [idx for idx, valid in enumerate(physically_valid_ev) if valid]
            if candidates:
                best = max(candidates, key=lambda idx: float(self.action_space['ev_power'][idx]))
                ev_mask[best] = True
            else:
                ev_mask[self._zero_action_index('ev_power')] = True
        masks['ev_power'] = ev_mask
        water_mask = [bool(valid) and self._ewh_setpoint_keeps_service_reachable(state, float(setpoint)) for valid, setpoint in zip(masks['ewh_set_temp'], self.action_space['ewh_set_temp'])]
        if not any(water_mask):
            hottest = max(range(len(water_mask)), key=lambda index: float(self.action_space['ewh_set_temp'][index]))
            water_mask[hottest] = True
        masks['ewh_set_temp'] = water_mask
        wash_actions = list(self.action_space['wash_machine_schedule'])
        wash_mask = list(masks['wash_machine_schedule'])
        already_committed = bool(self.has_run_in_current_period or self.remaining_runtime > 0 or self.wash_machine_scheduled_start_step is not None)
        if already_committed:
            wash_mask = [int(value) == 0 for value in wash_actions]
        else:
            current_step = int(state.get('time_index', self.current_time_index))
            latest = self._wash_abs_latest_start_step()
            if current_step >= latest:
                for idx, value in enumerate(wash_actions):
                    target = self._decode_wash_action_to_start_step(int(value))
                    wash_mask[idx] = target is not None and int(target) == int(latest)
        if not any(wash_mask):
            zero_idx = next((idx for idx, value in enumerate(wash_actions) if int(value) == 0), 0)
            wash_mask[zero_idx] = True
        masks['wash_machine_schedule'] = wash_mask
        penalties['wash_machine_schedule'] = [0.0] * len(wash_mask)
        return (masks, penalties)

    def _get_physical_action_mask(self, state):
        delta_t = self._dt_hours()
        battery_mask = [True] * len(self.action_space['battery_power'])
        ev_mask = [True] * len(self.action_space['ev_power'])
        max_discharge = state['ess_state'] / delta_t * self.discharge_efficiency
        max_charge = (self.ess_capacity - state['ess_state']) / delta_t / self.charge_efficiency
        for idx, av in enumerate(self.action_space['battery_power']):
            if av < 0 and abs(av) > max_discharge:
                battery_mask[idx] = False
            if av > 0 and av > max_charge:
                battery_mask[idx] = False
        max_ev_d = max(0.0, float(state['ev_battery_state'])) / delta_t * self.discharge_efficiency
        max_ev_c = (self.ev_capacity - state['ev_battery_state']) / delta_t / self.charge_efficiency
        for idx, av in enumerate(self.action_space['ev_power']):
            if av < 0 and abs(av) > max_ev_d:
                ev_mask[idx] = False
            if av > 0 and av > max_ev_c:
                ev_mask[idx] = False
            if not self.is_ev_at_home() and av != 0:
                ev_mask[idx] = False
        masks = {'battery_power': battery_mask, 'ev_power': ev_mask}
        penalties = {'battery_power': [0.0] * len(battery_mask), 'ev_power': [0.0] * len(ev_mask)}
        ti = int(state.get('time_index', 0))
        e = self._wash_abs_earliest_start_step()
        l = self._wash_abs_latest_start_step()
        masks['wash_machine_schedule'] = []
        for val in self.action_space['wash_machine_schedule']:
            v = int(val)
            if v == 0:
                masks['wash_machine_schedule'].append(True)
            else:
                tgt = e + v - 1
                masks['wash_machine_schedule'].append(bool(e <= tgt <= l and tgt >= ti))
        masks['Air_conditioner_set_temp'] = [True] * len(self.action_space['Air_conditioner_set_temp'])
        masks['Air_conditioner_set_temp2'] = [True] * len(self.action_space['Air_conditioner_set_temp2'])
        masks['ewh_set_temp'] = [True] * len(self.action_space['ewh_set_temp'])
        penalties['wash_machine_schedule'] = [0.0] * len(masks['wash_machine_schedule'])
        penalties['Air_conditioner_set_temp'] = [0.0] * len(masks['Air_conditioner_set_temp'])
        penalties['Air_conditioner_set_temp2'] = [0.0] * len(masks['Air_conditioner_set_temp2'])
        penalties['ewh_set_temp'] = [0.0] * len(masks['ewh_set_temp'])
        self._record_action_prior_stats({})
        self._ensure_valid_branch('battery_power', masks['battery_power'])
        self._ensure_valid_branch('ev_power', masks['ev_power'], prefer_charge=bool(self.is_ev_at_home()))
        return (masks, penalties)

    def _get_minimal_action_mask(self, state):
        return self._get_physical_action_mask(state)

    def _build_soc_masks(self, state, apply_soc_mask: bool=True):
        delta_t = self._dt_hours()
        n_batt = len(self.action_space['battery_power'])
        n_ev = len(self.action_space['ev_power'])
        result = {'battery': [True] * n_batt, 'ev': [True] * n_ev}
        if not apply_soc_mask:
            result['battery_penalty'] = [0.0] * n_batt
            result['ev_penalty'] = [0.0] * n_ev
            return result
        ess_soc = float(state['ess_state']) / max(1e-08, float(self.ess_capacity))
        ev_soc = float(state['ev_battery_state']) / max(1e-08, float(self.ev_capacity))
        max_discharge = state['ess_state'] / delta_t * self.discharge_efficiency
        max_charge = (self.ess_capacity - state['ess_state']) / delta_t / self.charge_efficiency
        for idx, av in enumerate(self.action_space['battery_power']):
            if av < 0 and abs(av) > max_discharge:
                result['battery'][idx] = False
            if av > 0 and av > max_charge:
                result['battery'][idx] = False
        max_ev_d = max(0.0, float(state['ev_battery_state']) - self._ev_operational_reserve_kwh()) / delta_t * self.discharge_efficiency
        max_ev_c = (self.ev_capacity - state['ev_battery_state']) / delta_t / self.charge_efficiency
        for idx, av in enumerate(self.action_space['ev_power']):
            if av < 0 and abs(av) > max_ev_d:
                result['ev'][idx] = False
            if av > 0 and av > max_ev_c:
                result['ev'][idx] = False
        if not self.is_ev_at_home():
            for idx, av in enumerate(self.action_space['ev_power']):
                if av != 0:
                    result['ev'][idx] = False
        battery_penalty = [0.0] * n_batt
        ev_penalty = [0.0] * n_ev

        def _apply_graduated(soc, actions, result_mask, penalty_list, previous_power, rated_power, steps_since_switch, healthy_low, healthy_high):
            prev_sign = self._power_sign(previous_power)
            for idx, av in enumerate(actions):
                if soc < self.mask_warn_low:
                    if av < 0:
                        result_mask[idx] = False
                    elif av > 0:
                        penalty_list[idx] += self.mask_soc_recovery_bonus
                elif soc < healthy_low:
                    ratio = (healthy_low - soc) / max(1e-08, healthy_low - self.mask_warn_low)
                    if av < 0:
                        penalty_list[idx] += self.mask_remind_pen * ratio
                    elif av > 0:
                        penalty_list[idx] += self.mask_soc_recovery_bonus * ratio
                elif soc <= healthy_high:
                    pass
                elif soc <= self.mask_warn_high:
                    ratio = (soc - healthy_high) / max(1e-08, self.mask_warn_high - healthy_high)
                    if av > 0:
                        penalty_list[idx] += self.mask_remind_pen * ratio
                    elif av < 0:
                        penalty_list[idx] += self.mask_soc_recovery_bonus * ratio
                elif av > 0:
                    result_mask[idx] = False
                elif av < 0:
                    penalty_list[idx] += self.mask_soc_recovery_bonus
                reversal_power_ratio = max(abs(float(previous_power)), abs(float(av))) / max(1e-08, float(rated_power))
                if result_mask[idx] and prev_sign != 0 and (self._power_sign(av) == -prev_sign) and (reversal_power_ratio >= 0.5) and (int(steps_since_switch) <= 2):
                    penalty_list[idx] += self.mask_reversal_penalty
        _apply_graduated(ess_soc, self.action_space['battery_power'], result['battery'], battery_penalty, self.prev_battery_power, self._max_abs_action_power('battery_power'), self._ess_steps_since_switch, self.ess_healthy_soc_low, self.ess_healthy_soc_high)
        _apply_graduated(ev_soc, self.action_space['ev_power'], result['ev'], ev_penalty, self.prev_ev_power, self._max_abs_action_power('ev_power'), self._ev_steps_since_switch, self.ev_healthy_soc_low, self.ev_healthy_soc_high)
        self._ensure_valid_branch('battery_power', result['battery'])
        self._ensure_valid_branch('ev_power', result['ev'], prefer_charge=self.is_ev_at_home())
        result['battery_penalty'] = battery_penalty
        result['ev_penalty'] = ev_penalty
        return result

    def _advance_to_next_block_day(self) -> None:
        if self._current_block_day_index + 1 >= self._current_block_days:
            raise RuntimeError('Cannot advance beyond the active episode block')
        self._current_block_day_index += 1
        self.current_time = self._current_block_dates[self._current_block_day_index]
        self._current_day_date_str = self.current_time
        self.current_time_index = 0
        self._ewh_draw_profile_cache.clear()
        self._ewh_draw_profile_l, self._ewh_draw_flow_lpm = self._generate_ewh_draw_profile(self.current_time)
        self._configure_episode_peak_target()
        self.current_episode_peak_import_kw = 0.0
        self.current_episode_peak_potential = 0.0
        departure_step = self._find_departure_step_within_episode(start_step=0)
        self._episode_trip_kwh = self._trip_kwh_for_date(self.current_time) if departure_step is not None else 0.0
        self._register_daily_ev_plan()
        self.current_ev_reserve_floor_kwh = self._ev_operational_reserve_kwh()
        daily_prices = [float(self.data_interface.get_electricity_price(self.current_time, idx)) for idx in range(self.steps_per_day)]
        self._episode_mean_price = float(np.mean(daily_prices)) if daily_prices else 0.0
        self._episode_mean_carbon_intensity = 0.0
        self._ess_cumulative_fec_today = 0.0
        self._ev_cumulative_fec_today = 0.0
        self._ess_cycle_count_today = 0
        self.current_period_start = None
        self.has_run_in_current_period = False
        self.time_deviation = 0
        self.wash_machine_state = 0
        self.last_action_time = None
        self.remaining_runtime = 0
        self.wash_machine_scheduled_start_step = None
        self._wash_price_step_for_penalty = None

    def reset(self):
        self.total_cost = 0
        if self.episode_length_steps is not None and self.episode_schedule is not None and (len(self.episode_schedule) > 0):
            schedule_item = self.episode_schedule[self.episode_schedule_cursor % len(self.episode_schedule)]
            house_id, date_str, block_days = self._parse_episode_schedule_item(schedule_item)
            self.episode_schedule_cursor += 1
            legacy_pair = isinstance(schedule_item, (tuple, list)) and len(schedule_item) == 2
            if self.cross_day_enabled and (not legacy_pair) and (not self.block_min_days <= block_days <= self.block_max_days):
                raise ValueError(f'v9 episode block must contain {self.block_min_days}--{self.block_max_days} days, got {block_days} for {(house_id, date_str)}')
            self.data_interface = self.data_class.from_storenet_ireland_2020(house_id=house_id, base_dir=self.storenet_base_dir, price_profile=self.price_profile, steps_per_day=self.steps_per_day, tariff_params=self.tariff_config, ev_schedule_params=self.ev_trip_config)
            if self._seed_value is not None:
                self.data_interface.seed(self._seed_value)
            self.current_time = self._normalize_date_str(date_str)
            self._current_episode_house_id = house_id
            self._current_episode_date_str = self.current_time
            self._current_day_date_str = self.current_time
            self._current_block_days = int(block_days)
        else:
            self.current_time = self._first_cons_date_str(self.data_interface)
            self._current_episode_house_id = None
            self._current_episode_date_str = self.current_time
            self._current_day_date_str = self.current_time
            self._current_block_days = 1
        start_date = datetime.strptime(self.current_time, '%Y-%m-%d')
        self._current_block_dates = [(start_date + timedelta(days=offset)).strftime('%Y-%m-%d') for offset in range(self._current_block_days)]
        available_dates = {self._normalize_date_str(value) for value in self.data_interface.cons_data.index}
        missing_dates = [value for value in self._current_block_dates if value not in available_dates]
        if missing_dates:
            raise ValueError(f'Episode block contains dates unavailable for {self._current_episode_house_id}: {missing_dates}')
        self._current_block_day_index = 0
        self._current_block_step = 0
        self.current_episode_length_steps = self._current_block_days * self.steps_per_day
        self._block_trip_energy_consumed_kwh = 0.0
        self._ewh_draw_profile_cache.clear()
        self._ewh_draw_profile_l, self._ewh_draw_flow_lpm = self._generate_ewh_draw_profile(self.current_time)
        self._configure_episode_peak_target()
        self.current_episode_peak_import_kw = 0.0
        self.current_episode_peak_potential = 0.0
        self.current_time_index = 0
        self.current_ev_power = 0
        self.current_battery_power = 0
        self.prev_battery_power = 0.0
        self.prev_ev_power = 0.0
        self._prev_ev_power_sign = 0
        self._ess_steps_since_switch = self.steps_per_day
        self._ev_steps_since_switch = self.steps_per_day
        self._ess_recent_switch_flags = deque(maxlen=self.battery_micro_cycle_window_steps)
        self._ev_recent_switch_flags = deque(maxlen=self.battery_micro_cycle_window_steps)
        self.current_step_ess_ramp_norm = 0.0
        self.current_step_ev_ramp_norm = 0.0
        self.current_step_battery_switching_cost = 0.0
        self.current_step_ess_switching_cost = 0.0
        self.current_step_ev_switching_cost = 0.0
        self.current_step_high_power_reversal_count = 0.0
        self.current_step_micro_cycle_count = 0.0
        self.current_invalid_action_penalty = 0.0
        self.current_invalid_action_dist = 0.0
        self.current_invalid_action_dist_kw = 0.0
        self.current_ess_repair_distance_norm = 0.0
        self.current_ev_repair_distance_norm = 0.0
        self.current_smoothness_penalty = 0.0
        self.current_step_raw_violations = 0
        self.current_step_chattering_events = 0
        self.current_step_ess_chattering_events = 0
        self.current_step_ev_chattering_events = 0
        self.current_step_grid_import_kw = 0.0
        self.current_step_grid_export_kw = 0.0
        self.current_step_net_load_kw = 0.0
        self.current_step_grid_import_kwh = 0.0
        self.current_step_grid_export_kwh = 0.0
        self.current_step_grid_import_cost = 0.0
        self.current_step_grid_export_price = 0.0
        self.current_step_grid_export_revenue = 0.0
        self.current_step_net_energy_cost = 0.0
        self.current_step_ev_departure_shortfall = 0.0
        self.current_step_ev_departure_event = 0
        self.current_step_ev_departure_success = 0
        self.current_step_ev_trip_consumption_kwh = 0.0
        self.current_step_user_flow = 0.0
        self.current_step_hot_water_draw_l = 0.0
        self.current_step_hot_water_delivery_temp_c = float(self.ewh_params['temp_init'])
        self.current_ev_deadline_risk = 0.0
        self.current_ev_deadline_required_kw = 0.0
        self.current_ev_deadline_gap_kwh = 0.0
        self.current_ev_deadline_reward = 0.0
        self.current_ev_deadline_business_penalty = 0.0
        self.current_ev_shield_intervened = 0
        self.current_ev_shield_action_delta_kw = 0.0
        self.current_ev_forced_charge_kwh = 0.0
        self.current_terminal_energy_penalty = 0.0
        self.current_terminal_ess_deficit_kwh = 0.0
        self.current_terminal_ev_deficit_kwh = 0.0
        self.current_terminal_ewh_deficit_kwh_th = 0.0
        self.current_terminal_hvac_deficit_kwh_th = 0.0
        self.current_terminal_grid_equivalent_kwh = 0.0
        self.current_terminal_settlement_cost_eur = 0.0
        self.current_terminal_settlement_carbon_kg = 0.0
        self.current_initial_ev_feasibility_adjustment_kwh = 0.0
        self.current_ev_reserve_floor_kwh = self._ev_operational_reserve_kwh()
        self.current_ev_reserve_gap_kwh = 0.0
        self._last_action_info = {}
        self.current_step_wash_started = 0
        self.current_step_wash_completed = 0
        self.current_step_wash_start_price = 0.0
        self.current_step_wash_on_window = 0
        self.current_step_wash_price_rank = 0.0
        self.current_step_hvac_safety_interventions = 0
        self._reset_step_energy_flow_metrics()
        self._ess_cumulative_fec_today = 0.0
        self._ev_cumulative_fec_today = 0.0
        self._ess_cycle_count_today = 0
        self._ess_cumulative_fec_block = 0.0
        self._ev_cumulative_fec_block = 0.0
        self._ess_cycle_count_block = 0
        self._prev_battery_power_sign = 0
        departure_step = self._find_departure_step_within_episode(start_step=0)
        self._episode_trip_kwh = self._trip_kwh_for_date(self.current_time) if departure_step is not None else 0.0
        self._register_daily_ev_plan()
        self.current_ev_reserve_floor_kwh = self._ev_operational_reserve_kwh()
        daily_prices = [float(self.data_interface.get_electricity_price(self.current_time, idx)) for idx in range(self.steps_per_day)]
        self._episode_mean_price = float(np.mean(daily_prices)) if daily_prices else 0.0
        self._episode_mean_carbon_intensity = 0.0
        min_feasible_initial_kwh = self._ev_operational_reserve_kwh()
        if departure_step is not None:
            available_steps = int(departure_step) + 1
            max_charge_kw = max(0.0, max((float(v) for v in self.action_space['ev_power'])))
            max_charge_kwh = available_steps * max_charge_kw * self.charge_efficiency * self._dt_hours()
            min_feasible_initial_kwh = max(min_feasible_initial_kwh, self._ev_deadline_target_kwh(hard=True) - max_charge_kwh)
        if self.randomize_initial_soc:
            ess_rng = self.data_interface.scenario_rng(self.current_time, 0, salt=501)
            ev_rng = self.data_interface.scenario_rng(self.current_time, 0, salt=502)
            ess_soc0 = float(ess_rng.uniform(self.ess_initial_soc_low, self.ess_initial_soc_high))
            ev_low_kwh = max(float(np.clip(self.ev_initial_soc_low, 0.0, 1.0) * self.ev_capacity), min_feasible_initial_kwh)
            ev_high_kwh = float(np.clip(self.ev_initial_soc_high, 0.0, 1.0) * self.ev_capacity)
            if ev_low_kwh > ev_high_kwh + 1e-08:
                raise ValueError(f'EV 初始 SOC 配置区间无法满足当日离家可达性：required={ev_low_kwh:.3f} kWh, configured_high={ev_high_kwh:.3f} kWh')
            ev_init_kwh = ev_low_kwh if np.isclose(ev_low_kwh, ev_high_kwh) else float(ev_rng.uniform(ev_low_kwh, ev_high_kwh))
        else:
            ess_soc0 = self.ess_initial_soc
            ev_init_kwh = float(np.clip(self.ev_initial_soc, 0.0, 1.0) * self.ev_capacity)
            if ev_init_kwh < min_feasible_initial_kwh - 1e-08:
                raise ValueError(f'固定 EV 初始 SOC 无法满足当日离家可达性：initial={ev_init_kwh:.3f} kWh, required={min_feasible_initial_kwh:.3f} kWh')
        ess_init_kwh = float(np.clip(ess_soc0, 0.0, 1.0) * self.ess_capacity)
        ev_init_kwh = float(np.clip(ev_init_kwh, 0.0, self.ev_capacity))
        self._episode_initial_ess_kwh = ess_init_kwh
        self._episode_initial_ev_kwh = ev_init_kwh
        self._episode_initial_ewh_temp_c = float(self.ewh_params['temp_init'])
        self._episode_initial_indoor_temp_c = float(CONFIG['environment']['comfort']['indoor_temp_init'])
        self._episode_initial_indoor_temp2_c = float(CONFIG['environment']['comfort']['indoor_temp2_init'])
        self.indoor_temp = self._episode_initial_indoor_temp_c
        self.indoor_temp2 = self._episode_initial_indoor_temp2_c
        initial_outdoor_temp = self.data_interface.get_outdoor_temp(self.current_time, self.current_time_index)
        self.state = {'home_load': self.data_interface.get_home_load(self.current_time, self.current_time_index), 'pv_generation': self.data_interface.get_pv_generation(self.current_time, self.current_time_index), 'ess_state': ess_init_kwh, 'ev_battery_state': ev_init_kwh, 'time_index': self.current_time_index, 'electricity_price': self.data_interface.get_electricity_price(self.current_time, self.current_time_index), 'grid_export_price': self.data_interface.get_grid_export_price(self.current_time, self.current_time_index), 'temperature': initial_outdoor_temp, 'indoor_temp': self.indoor_temp, 'indoor_temp2': self.indoor_temp2, 'wash_machine_state': 0, 'Air_conditioner_power': 0, 'Air_conditioner_power2': 0, 'ewh_temp': self.ewh_params['temp_init'], 'ewh_power': 0, **self._get_future_price_dict(), **self._forecast_observation_dict(), **self._calendar_observation_dict(), **self._scenario_observation_dict(), 'ess_cumulative_fec_today': 0.0, 'ev_cumulative_fec_today': 0.0, 'ess_cycle_count_today': 0.0, **self._battery_motion_observation_dict()}
        self.state.update(self._ev_deadline_observation_dict(self.state))
        self.state.update(self._boundary_observation_dict(self.state))
        self.ev_battery_record = []
        self.ess_state_record = []
        self.home_load_record = []
        self.pv_generation_record = []
        self.electricity_price_record = []
        self.ev_at_home_record = []
        self.wash_machine_record = []
        self.air_conditioner_power_record = []
        self.air_conditioner_power_record2 = []
        self.ess_actions = []
        self.wash_machine_actions = []
        self.air_conditioner_actions = []
        self.ess_charge_pv = []
        self.ess_charge_grid = []
        self.ess_discharge_ev = []
        self.ess_discharge_house = []
        self.ess_discharge_grid = []
        self.records = {'timestamps': [], 'ev_soc': [], 'ess_soc': [], 'grid_power': [], 'energy_cost': [], 'daily_costs': [], 'indoor_temp': [], 'indoor_temp2': [], 'outdoor_temp': [], 'current_daily_cost': 0, 'total_load': [], 'user_dissatisfaction': [], 'requested_battery_power': [], 'executed_battery_power': [], 'requested_ev_power': [], 'executed_ev_power': [], 'ev_shield_intervened': [], 'solar_irradiance_norm': [], 'zone1_occupied': [], 'zone2_occupied': [], 'hot_water_draw_l': [], 'hvac_cop': []}
        self.reward_components = {'total': [], 'energy_cost': [], 'violation_penalty': [], 'aging_penalty': [], 'battery_health_penalty': [], 'smoothness_penalty': [], 'invalid_action_penalty': [], 'user_penalty': [], 'temp_reward': [], 'ac_comfort_reward': [], 'hot_water_service_reward': [], 'comfort_bonus': [], 'comfort_penalty': [], 'carbon_penalty': [], 'pv_self_use_reward': [], 'peak_shave_reward': [], 'ev_deadline_reward': [], 'ev_shield_reward': [], 'terminal_energy_reward': [], 'boundary_recovery_reward': [], 'core_reward': [], 'raw_violation_count': [], 'chattering_count': []}
        self.current_period_start = None
        self.has_run_in_current_period = False
        self.time_deviation = 0
        self.wash_machine_state = 0
        self.last_action_time = None
        self.remaining_runtime = 0
        self.wash_machine_scheduled_start_step = None
        self._wash_price_step_for_penalty = None
        _cmf = CONFIG['environment']['comfort']
        self.indoor_temp = self._episode_initial_indoor_temp_c
        self.indoor_temp2 = self._episode_initial_indoor_temp2_c
        self.outdoor_temp = self.data_interface.get_outdoor_temp(self.current_time, self.current_time_index)
        self.state['indoor_temp'] = self.indoor_temp
        self.state['indoor_temp2'] = self.indoor_temp2
        self.state.update(self._calendar_observation_dict())
        self.state.update(self._scenario_observation_dict())
        self.state.update(self._time_index_observation_dict())
        self.state.update(self._wash_schedule_observation_dict())
        self.state.update(self._carbon_observation_dict())
        self.state.update(self._peak_observation_dict())
        self.state.update(self._ev_deadline_observation_dict(self.state))
        self.state.update(self._boundary_observation_dict(self.state))
        self._ess_stored_from_pv_kwh = float(self.state['ess_state']) * 0.5
        self._ess_stored_from_grid_kwh = float(self.state['ess_state']) * 0.5
        self.state['ess_charge_pv_fraction'] = self._ess_charge_pv_fraction()
        self.ewh_temp_record = []
        self.ewh_power_record = []
        self.user_flow_record = []
        self.hot_water_draw_volume_record = []
        self._validate_observation_schema(self.state, context='reset')
        return dict(self.state)

    def step(self, state, action):
        state = dict(state)
        requested_action = dict(action)
        executed_action = dict(requested_action)
        dt_h = self._dt_hours()
        requested_battery = float(requested_action['battery_power'])
        requested_ev = float(requested_action['ev_power'])
        ess_kwh = float(state['ess_state'])
        ev_kwh = float(state['ev_battery_state'])
        ev_energy_before_action = ev_kwh
        ev_at_home_now = bool(self.data_interface.is_ev_at_home(self.current_time, self.current_time_index))
        max_ess_discharge = ess_kwh / dt_h * self.discharge_efficiency
        max_ess_charge = (self.ess_capacity - ess_kwh) / dt_h / self.charge_efficiency
        max_ev_discharge = max(0.0, ev_energy_before_action) / dt_h * self.discharge_efficiency
        max_ev_charge = (self.ev_capacity - ev_energy_before_action) / dt_h / self.charge_efficiency
        raw_violations = 0
        if requested_battery < -max_ess_discharge - 0.0001 or requested_battery > max_ess_charge + 0.0001:
            raw_violations += 1
        if not ev_at_home_now and abs(requested_ev) > 0.0001:
            raw_violations += 1
        elif requested_ev < -max_ev_discharge - 0.0001 or requested_ev > max_ev_charge + 0.0001:
            raw_violations += 1
        self.current_step_raw_violations = raw_violations
        physical_battery = float(np.clip(requested_battery, -max_ess_discharge, max_ess_charge))
        if ev_at_home_now:
            physical_ev = float(np.clip(requested_ev, -max_ev_discharge, max_ev_charge))
        else:
            physical_ev = 0.0
        shield_state = dict(state)
        shield_state['ev_battery_state'] = float(ev_energy_before_action)
        executed_action['battery_power'] = physical_battery
        executed_action['ev_power'] = self._apply_ev_viability_shield(shield_state, physical_ev)
        ess_repair_kw = abs(requested_battery - physical_battery)
        ev_repair_kw = abs(requested_ev - physical_ev)
        self.current_invalid_action_dist_kw = float(ess_repair_kw + ev_repair_kw)
        self.current_ess_repair_distance_norm = float(ess_repair_kw / self._max_abs_action_power('battery_power'))
        self.current_ev_repair_distance_norm = float(ev_repair_kw / self._max_abs_action_power('ev_power'))
        if self.raw_action_distance_normalization == 'per_device_max_abs_action':
            repair_distance = self.current_ess_repair_distance_norm + self.current_ev_repair_distance_norm
        else:
            raise ValueError(f'Unsupported raw_action_distance_normalization: {self.raw_action_distance_normalization!r}')
        self.current_invalid_action_dist = float(np.clip(repair_distance, 0.0, self.raw_action_distance_cap))
        self.current_invalid_action_penalty = 0.0
        self.current_smoothness_penalty = 0.0
        self.current_step_ess_ramp_norm = 0.0
        self.current_step_ev_ramp_norm = 0.0
        self.current_step_battery_switching_cost = 0.0
        self.current_step_ess_switching_cost = 0.0
        self.current_step_ev_switching_cost = 0.0
        self.current_step_high_power_reversal_count = 0.0
        self.current_step_micro_cycle_count = 0.0
        self.current_step_chattering_events = 0
        self.current_step_ess_chattering_events = 0
        self.current_step_ev_chattering_events = 0
        self.current_step_ev_departure_shortfall = 0.0
        self.current_step_ev_departure_event = 0
        self.current_step_ev_departure_success = 0
        self.current_step_ev_trip_consumption_kwh = 0.0
        self.current_step_user_flow = 0.0
        self.current_ev_deadline_risk = 0.0
        self.current_ev_deadline_required_kw = 0.0
        self.current_ev_deadline_gap_kwh = 0.0
        self.current_ev_deadline_reward = 0.0
        self.current_ev_deadline_business_penalty = 0.0
        self.current_terminal_energy_penalty = 0.0
        self.current_terminal_ess_deficit_kwh = 0.0
        self.current_terminal_ev_deficit_kwh = 0.0
        self.current_terminal_ewh_deficit_kwh_th = 0.0
        self.current_terminal_hvac_deficit_kwh_th = 0.0
        self.current_terminal_grid_equivalent_kwh = 0.0
        self.current_terminal_settlement_cost_eur = 0.0
        self.current_terminal_settlement_carbon_kg = 0.0
        self.current_step_wash_started = 0
        self.current_step_wash_completed = 0
        self.current_step_wash_start_price = 0.0
        self.current_step_wash_on_window = 0
        self.current_step_wash_price_rank = 0.0
        self.current_step_hvac_safety_interventions = 0
        self._reset_step_energy_flow_metrics()
        battery_power = float(executed_action['battery_power'])
        ev_power = float(executed_action['ev_power'])
        if self._power_sign(battery_power) == -self._power_sign(self.prev_battery_power) and self._power_sign(battery_power) != 0:
            self.current_step_chattering_events += 1
            self.current_step_ess_chattering_events += 1
        if self._power_sign(ev_power) == -self._power_sign(self.prev_ev_power) and self._power_sign(ev_power) != 0:
            self.current_step_chattering_events += 1
            self.current_step_ev_chattering_events += 1
        self._update_battery_motion_metrics(executed_action)
        self.prev_battery_power = battery_power
        self.prev_ev_power = ev_power
        self.current_battery_power = battery_power
        self.current_ev_power = ev_power
        ess_fec_increment = abs(battery_power) * dt_h / max(1e-08, 2.0 * self.ess_capacity)
        ev_fec_increment = abs(ev_power) * dt_h / max(1e-08, 2.0 * self.ev_capacity)
        self._ess_cumulative_fec_today += ess_fec_increment
        self._ev_cumulative_fec_today += ev_fec_increment
        self._ess_cumulative_fec_block += ess_fec_increment
        self._ev_cumulative_fec_block += ev_fec_increment
        battery_sign = self._power_sign(battery_power)
        if battery_sign != 0 and battery_sign != self._prev_battery_power_sign:
            self._ess_cycle_count_today += 1
            self._ess_cycle_count_block += 1
        if battery_sign != 0:
            self._prev_battery_power_sign = battery_sign
        current_dt = datetime.strptime(self.current_time, '%Y-%m-%d') + timedelta(minutes=60.0 * dt_h * int(self.current_time_index))
        self.outdoor_temp = self.data_interface.get_outdoor_temp(self.current_time, self.current_time_index)
        ac1_power, self.indoor_temp, ac2_power, self.indoor_temp2 = self.update_hvac_system(executed_action['Air_conditioner_set_temp'], executed_action['Air_conditioner_set_temp2'], self.indoor_temp, self.indoor_temp2, state)
        wash_state = self.update_wash_machine2(executed_action['wash_machine_schedule'])
        if self._wash_price_step_for_penalty is not None:
            self.current_step_wash_started = 1
            self.current_step_wash_start_price = float(self.data_interface.get_electricity_price(self.current_time, self._wash_price_step_for_penalty))
            earliest = self._wash_abs_earliest_start_step()
            latest = self._wash_abs_latest_start_step()
            self.current_step_wash_on_window = int(earliest <= int(self._wash_price_step_for_penalty) <= latest)
            self.current_step_wash_price_rank = self._price_rank_in_wash_window(self.current_time, self.current_step_wash_start_price)
        executed_action['ewh_set_temp'] = self._apply_ewh_viability_shield(state, requested_action['ewh_set_temp'])
        ewh_power = self._fuzzy_heating_control(executed_action['ewh_set_temp'], state['ewh_temp'])
        ewh_temp, user_flow, draw_volume_l, delivery_temp_c = self.update_water_heater(ewh_power, state['ewh_temp'])
        self.current_step_user_flow = float(user_flow)
        self.current_step_hot_water_draw_l = float(draw_volume_l)
        self.current_step_hot_water_delivery_temp_c = float(delivery_temp_c)
        post_state = {'home_load': float(state['home_load']), 'pv_generation': float(state['pv_generation']), 'ess_state': self.update_ess(battery_power, state['pv_generation'], base_energy=ess_kwh), 'ev_battery_state': self.update_ev_battery(ev_power, base_energy=ev_energy_before_action), 'time_index': int(self.current_time_index), 'electricity_price': float(state['electricity_price']), 'grid_export_price': float(state['grid_export_price']), 'temperature': float(self.outdoor_temp), 'indoor_temp': float(self.indoor_temp), 'indoor_temp2': float(self.indoor_temp2), 'wash_machine_state': int(wash_state), 'Air_conditioner_power': float(ac1_power), 'Air_conditioner_power2': float(ac2_power), 'ewh_temp': float(ewh_temp), 'ewh_power': float(ewh_power), 'ess_cumulative_fec_today': float(self._ess_cumulative_fec_today), 'ev_cumulative_fec_today': float(self._ev_cumulative_fec_today), 'ess_cycle_count_today': float(self._ess_cycle_count_today), **self._battery_motion_observation_dict(), **self._scenario_observation_dict()}
        departure_step = self._find_departure_step_within_episode()
        if departure_step is not None and int(departure_step) == int(self.current_time_index):
            self.current_step_ev_departure_event = 1
            hard_target = self._ev_deadline_target_kwh(hard=True)
            self.current_step_ev_departure_shortfall = max(0.0, hard_target - float(post_state['ev_battery_state']))
            self.current_step_ev_departure_success = int(self.current_step_ev_departure_shortfall <= 1e-06)
        self._last_action_info = {'requested_action': dict(requested_action), 'physical_action': {**requested_action, 'battery_power': physical_battery, 'ev_power': physical_ev}, 'executed_action': dict(executed_action), 'physical_repair_distance_kw': float(self.current_invalid_action_dist_kw), 'physical_repair_distance_norm': float(self.current_invalid_action_dist), 'ess_repair_distance_norm': float(self.current_ess_repair_distance_norm), 'ev_repair_distance_norm': float(self.current_ev_repair_distance_norm), 'ev_shield_intervened': int(self.current_ev_shield_intervened), 'ev_shield_action_delta_kw': float(self.current_ev_shield_action_delta_kw), 'ev_forced_charge_kwh': float(self.current_ev_forced_charge_kwh), 'ev_service_context': dict(self.current_ev_service_context), 'ev_service_infeasible': int(self.current_ev_service_infeasible), 'ewh_service_context': dict(self.current_ewh_service_context), 'ewh_shield_intervened': int(self.current_ewh_shield_intervened), 'ewh_shield_action_delta_c': float(self.current_ewh_shield_action_delta_c), 'ewh_service_infeasible': int(self.current_ewh_service_infeasible)}
        self.state = post_state
        reward = self.calculate_reward(state, executed_action, post_state)
        post_state.update(self._get_future_price_dict())
        post_state.update(self._forecast_observation_dict())
        post_state.update(self._calendar_observation_dict())
        post_state.update(self._time_index_observation_dict())
        post_state.update(self._wash_schedule_observation_dict())
        post_state.update(self._carbon_observation_dict())
        post_state['ess_charge_pv_fraction'] = self._ess_charge_pv_fraction()
        post_state.update(self._peak_observation_dict(self.current_step_grid_import_kw))
        post_state.update(self._ev_deadline_observation_dict(post_state))
        post_state.update(self._boundary_observation_dict(post_state))
        done = self.is_terminal_state()
        self.ev_battery_record.append(float(post_state['ev_battery_state']))
        self.ess_state_record.append(float(post_state['ess_state']))
        self.home_load_record.append(float(post_state['home_load']))
        self.pv_generation_record.append(float(post_state['pv_generation']))
        self.electricity_price_record.append(float(post_state['electricity_price']))
        self.ev_at_home_record.append(ev_at_home_now)
        self.wash_machine_record.append(int(post_state['wash_machine_state']))
        self.air_conditioner_power_record.append(float(post_state['Air_conditioner_power']))
        self.air_conditioner_power_record2.append(float(post_state['Air_conditioner_power2']))
        self.ess_actions.append(battery_power)
        self.records['indoor_temp'].append(float(self.indoor_temp))
        self.records['indoor_temp2'].append(float(self.indoor_temp2))
        self.records['outdoor_temp'].append(float(self.outdoor_temp))
        self.records['timestamps'].append(current_dt)
        self.records['ev_soc'].append(float(post_state['ev_battery_state']))
        self.records['ess_soc'].append(float(post_state['ess_state']))
        self.records['total_load'].append(float(self.total_load_compute()))
        self.records['requested_battery_power'].append(requested_battery)
        self.records['executed_battery_power'].append(battery_power)
        self.records['requested_ev_power'].append(requested_ev)
        self.records['executed_ev_power'].append(ev_power)
        self.records['ev_shield_intervened'].append(int(self.current_ev_shield_intervened))
        self.records['solar_irradiance_norm'].append(float(post_state['solar_irradiance_norm']))
        self.records['zone1_occupied'].append(float(post_state['zone1_occupied']))
        self.records['zone2_occupied'].append(float(post_state['zone2_occupied']))
        self.records['hot_water_draw_l'].append(float(draw_volume_l))
        self.records['hvac_cop'].append(float(self.current_hvac_cop))
        self.ewh_temp_record.append(float(ewh_temp))
        self.ewh_power_record.append(float(ewh_power))
        self.user_flow_record.append(float(user_flow))
        self.hot_water_draw_volume_record.append(float(draw_volume_l))
        self.ess_charge_pv.append(float(self.current_step_ess_charge_from_pv_kwh))
        self.ess_charge_grid.append(float(self.current_step_ess_charge_non_pv_kwh))
        self.ess_discharge_ev.append(0.0)
        self.ess_discharge_house.append(float(self.current_step_ess_discharge_to_load_kwh))
        self.ess_discharge_grid.append(float(self.current_step_ess_discharge_to_grid_kwh))
        if done:
            self.state = post_state
            self._validate_observation_schema(post_state, context='terminal_step')
            return (dict(post_state), float(reward), True)
        self._current_block_step += 1
        crossed_day_boundary = self.current_time_index >= self.steps_per_day - 1
        if crossed_day_boundary:
            self._advance_to_next_block_day()
            wash_state = 0
        else:
            self.current_time_index += 1
        next_ev_kwh = float(post_state['ev_battery_state'])
        if self._is_ev_arrival_step():
            trip_consumed = min(next_ev_kwh, float(self._episode_trip_kwh))
            next_ev_kwh = max(0.0, next_ev_kwh - trip_consumed)
            self._block_trip_energy_consumed_kwh += trip_consumed
            self.current_step_ev_trip_consumption_kwh = float(trip_consumed)
        next_state = {'home_load': float(self.data_interface.get_home_load(self.current_time, self.current_time_index)), 'pv_generation': float(self.data_interface.get_pv_generation(self.current_time, self.current_time_index)), 'ess_state': float(post_state['ess_state']), 'ev_battery_state': next_ev_kwh, 'time_index': int(self.current_time_index), 'electricity_price': float(self.data_interface.get_electricity_price(self.current_time, self.current_time_index)), 'grid_export_price': float(self.data_interface.get_grid_export_price(self.current_time, self.current_time_index)), 'temperature': float(self.data_interface.get_outdoor_temp(self.current_time, self.current_time_index)), 'indoor_temp': float(self.indoor_temp), 'indoor_temp2': float(self.indoor_temp2), 'wash_machine_state': int(wash_state), 'Air_conditioner_power': float(ac1_power), 'Air_conditioner_power2': float(ac2_power), 'ewh_temp': float(ewh_temp), 'ewh_power': float(ewh_power), **self._get_future_price_dict(), **self._forecast_observation_dict(), **self._calendar_observation_dict(), **self._scenario_observation_dict(), **self._time_index_observation_dict(), **self._wash_schedule_observation_dict(), **self._carbon_observation_dict(), 'ess_charge_pv_fraction': self._ess_charge_pv_fraction(), **self._peak_observation_dict(0.0 if crossed_day_boundary else self.current_step_grid_import_kw), 'ess_cumulative_fec_today': float(self._ess_cumulative_fec_today), 'ev_cumulative_fec_today': float(self._ev_cumulative_fec_today), 'ess_cycle_count_today': float(self._ess_cycle_count_today), **self._battery_motion_observation_dict()}
        next_state.update(self._ev_deadline_observation_dict(next_state))
        next_state.update(self._boundary_observation_dict(next_state))
        self.state = next_state
        self._validate_observation_schema(next_state, context='step')
        return (dict(next_state), float(reward), False)

    def update_ev_battery(self, ev_charge_discharge, base_energy: Optional[float]=None):
        dt_h = self._dt_hours()
        energy = float(self.state['ev_battery_state'] if base_energy is None else base_energy)
        new_soc = energy + self._ev_energy_delta_for_power(ev_charge_discharge, dt_h)
        return np.clip(new_soc, 0, self.ev_capacity)

    def total_load_compute(self):
        ev_power = self.current_ev_power
        ess_power = self.current_battery_power
        pv_load = self.state['pv_generation']
        home_load = self.state['home_load']
        air_cond_power = self.state['Air_conditioner_power']
        air_cond_power2 = self.state['Air_conditioner_power2']
        wash_machine_power = self.state['wash_machine_state'] * self.wash_machine_power
        ewh_power = self.state['ewh_power']
        total_load = home_load + air_cond_power + air_cond_power2 + wash_machine_power + ewh_power + ev_power + ess_power - pv_load
        return total_load

    def update_ess(self, ess_charge_discharge, pv_generation, base_energy: Optional[float]=None):
        dt_h = self._dt_hours()
        energy = float(self.state['ess_state'] if base_energy is None else base_energy)
        if ess_charge_discharge > 0:
            new_ess = energy + ess_charge_discharge * self.charge_efficiency * dt_h
        else:
            new_ess = energy + ess_charge_discharge / self.discharge_efficiency * dt_h
        return np.clip(new_ess, 0, self.ess_capacity)

    def _get_period_start(self, dt):
        if dt.hour >= 0:
            return datetime(dt.year, dt.month, dt.day, 0, 0)
        else:
            prev_day = dt - timedelta(days=1)
            return datetime(prev_day.year, prev_day.month, prev_day.day, 0, 0)

    def _get_period_end(self, dt):
        return datetime(dt.year, dt.month, dt.day, 23, 0)

    def update_time_deviation(self, scheduled_time):
        pref_start = self.current_period_start.replace(hour=self.wash_machine_preferred_time[0], minute=0)
        pref_end = self.current_period_start.replace(hour=self.wash_machine_preferred_time[1], minute=0) - timedelta(hours=self.wash_machine_duration)
        if scheduled_time < pref_start:
            self.time_deviation = (pref_start - scheduled_time).total_seconds() / 3600
        elif scheduled_time > pref_end:
            self.time_deviation = (scheduled_time - pref_end).total_seconds() / 3600
        else:
            self.time_deviation = 0

    def _wash_abs_earliest_start_step(self) -> int:
        return int(self.wash_machine_preferred_time[0] * self.steps_per_day / 24)

    def _wash_duration_steps(self) -> int:
        return max(1, int(round(float(self.wash_machine_duration) * self.steps_per_day / 24)))

    def _wash_abs_latest_start_step(self) -> int:
        dur_steps = self._wash_duration_steps()
        latest = int(self.wash_machine_preferred_time[1] * self.steps_per_day / 24) - dur_steps
        e = self._wash_abs_earliest_start_step()
        return max(e, min(self.steps_per_day - 1, latest))

    def _decode_wash_action_to_start_step(self, action_val: int) -> Optional[int]:
        if int(action_val) <= 0:
            return None
        return self._wash_abs_earliest_start_step() + int(action_val) - 1

    def _clamp_wash_start_step(self, target: int, cur_idx: int) -> int:
        e = self._wash_abs_earliest_start_step()
        l = self._wash_abs_latest_start_step()
        t = int(np.clip(int(target), e, l))
        return int(min(l, max(t, cur_idx)))

    def _dt_at_step_index(self, step_index: int) -> datetime:
        return datetime.strptime(self.current_time, '%Y-%m-%d') + timedelta(minutes=60.0 * self._dt_hours() * int(step_index))

    def update_wash_machine2(self, schedule_action):
        self._wash_price_step_for_penalty = None
        current_dt = datetime.strptime(self.current_time, '%Y-%m-%d') + timedelta(minutes=60.0 * self._dt_hours() * int(self.current_time_index))
        schedule_action = int(schedule_action)

        def _mark_wash_start_price_step() -> None:
            self._wash_price_step_for_penalty = int(self.current_time_index)
        if self.remaining_runtime > 0:
            self.wash_machine_state = 1
            self.remaining_runtime -= 1
            if self.remaining_runtime == 0:
                self.current_step_wash_completed = 1
            return self.wash_machine_state
        current_period_start = self._get_period_start(current_dt)
        if current_period_start != self.current_period_start:
            self.current_period_start = current_period_start
            self.has_run_in_current_period = False
            self.wash_machine_state = 0
            self.wash_machine_scheduled_start_step = None
            self.time_deviation = 0
        if self.has_run_in_current_period:
            self.wash_machine_state = 0
            self.wash_machine_scheduled_start_step = None
            self.time_deviation = 0
            return self.wash_machine_state
        earliest = self._wash_abs_earliest_start_step()
        latest = self._wash_abs_latest_start_step()
        _spd = int(self.steps_per_day)
        _idx = int(self.current_time_index)
        _ep = getattr(self, 'episode_length_steps', None)
        _last_step_of_episode = _ep is not None and int(_ep) > 0 and (int(_ep) <= _spd) and (_idx >= int(_ep) - 1)
        _need_daily = not self.has_run_in_current_period and self.remaining_runtime <= 0
        _force = _need_daily and (_idx >= latest or _idx >= _spd - 1 or _last_step_of_episode)

        def _start_immediate_wash() -> int:
            self.wash_machine_scheduled_start_step = None
            self.wash_machine_state = 1
            self.has_run_in_current_period = True
            self.remaining_runtime = self._wash_duration_steps() - 1
            if self.remaining_runtime == 0:
                self.current_step_wash_completed = 1
            self.update_time_deviation(current_dt)
            _mark_wash_start_price_step()
            return self.wash_machine_state

        def _schedule_at_step(target_step: int) -> int:
            ts = self._clamp_wash_start_step(target_step, _idx)
            sdt = self._dt_at_step_index(ts)
            self.update_time_deviation(sdt)
            if ts == _idx:
                return _start_immediate_wash()
            self.wash_machine_scheduled_start_step = int(ts)
            self.wash_machine_state = 0
            return self.wash_machine_state
        if _force:
            if _idx < earliest:
                return _schedule_at_step(earliest)
            return _start_immediate_wash()
        if self.wash_machine_scheduled_start_step is not None:
            if self.current_time_index >= int(self.wash_machine_scheduled_start_step):
                self.update_time_deviation(current_dt)
                self.wash_machine_state = 1
                self.has_run_in_current_period = True
                self.remaining_runtime = self._wash_duration_steps() - 1
                if self.remaining_runtime == 0:
                    self.current_step_wash_completed = 1
                self.wash_machine_scheduled_start_step = None
                _mark_wash_start_price_step()
                return self.wash_machine_state
            if schedule_action == 0:
                self.wash_machine_state = 0
                return self.wash_machine_state
            tgt = self._decode_wash_action_to_start_step(schedule_action)
            if tgt is None:
                self.wash_machine_state = 0
                return self.wash_machine_state
            return _schedule_at_step(tgt)
        if schedule_action == 0:
            self.wash_machine_state = 0
            self.time_deviation = 0
            return self.wash_machine_state
        tgt = self._decode_wash_action_to_start_step(schedule_action)
        if tgt is None:
            self.wash_machine_state = 0
            self.time_deviation = 0
            return self.wash_machine_state
        return _schedule_at_step(tgt)

    def _heating_cop(self, outdoor_temp_c: float) -> float:
        cop = self.heating_cop_at_7c + self.heating_cop_slope_per_c * (float(outdoor_temp_c) - 7.0)
        return float(np.clip(cop, self.heating_cop_min, self.heating_cop_max))

    def _hvac_power_request(self, set_temp_c: float, indoor_temp_c: float) -> Tuple[float, float]:
        error_c = float(set_temp_c) - float(indoor_temp_c)
        active_error_c = max(0.0, abs(error_c) - self.hvac_deadband_c)
        fraction = float(np.clip(active_error_c / self.hvac_full_power_error_c, 0.0, 1.0))
        direction = 1.0 if error_c > self.hvac_deadband_c else -1.0 if error_c < -self.hvac_deadband_c else 0.0
        return (self.hvac_zone_max_power_kw * fraction, direction)

    def update_hvac_system(self, set_temp1: float, set_temp2: float, indoor_temp1: float, indoor_temp2: float, state: Optional[Dict]=None) -> Tuple[float, float, float, float]:
        state = state or getattr(self, 'state', {})
        safety_low, safety_high = self.occupied_safety_bounds_c
        safe_set_temp1 = float(set_temp1)
        safe_set_temp2 = float(set_temp2)
        if float(indoor_temp1) <= safety_low:
            safe_set_temp1 = max(safe_set_temp1, self.zone_occupied_bounds_c[0][0])
        elif float(indoor_temp1) >= safety_high:
            safe_set_temp1 = min(safe_set_temp1, self.zone_occupied_bounds_c[0][1])
        if float(indoor_temp2) <= safety_low:
            safe_set_temp2 = max(safe_set_temp2, self.zone_occupied_bounds_c[1][0])
        elif float(indoor_temp2) >= safety_high:
            safe_set_temp2 = min(safe_set_temp2, self.zone_occupied_bounds_c[1][1])
        self.current_step_hvac_safety_interventions = int(abs(safe_set_temp1 - float(set_temp1)) > 1e-08) + int(abs(safe_set_temp2 - float(set_temp2)) > 1e-08)
        dt_h = self._dt_hours()
        request1, direction1 = self._hvac_power_request(safe_set_temp1, indoor_temp1)
        request2, direction2 = self._hvac_power_request(safe_set_temp2, indoor_temp2)
        request_total = request1 + request2
        scale = min(1.0, self.hvac_shared_max_power_kw / max(1e-08, request_total))
        power1 = request1 * scale
        power2 = request2 * scale
        cop_heat = self._heating_cop(self.outdoor_temp)
        thermal_hvac1 = power1 * (cop_heat if direction1 > 0.0 else self.cooling_eer) * direction1
        thermal_hvac2 = power2 * (cop_heat if direction2 > 0.0 else self.cooling_eer) * direction2
        self.current_hvac_cop = float((power1 * (cop_heat if direction1 > 0.0 else self.cooling_eer) + power2 * (cop_heat if direction2 > 0.0 else self.cooling_eer)) / max(1e-08, power1 + power2)) if power1 + power2 > 1e-08 else cop_heat
        occ1, occ2 = self.data_interface.get_zone_occupancy(self.current_time, self.current_time_index)
        home_load_kw = max(0.0, float(state.get('home_load', 0.0)))
        appliance_gain_kw = min(self.internal_load_heat_cap_kw, self.internal_load_heat_fraction * home_load_kw)
        base_and_appliance = self.internal_gain_base_kw + appliance_gain_kw
        internal_gain1 = 0.65 * base_and_appliance + self.internal_gain_per_occupant_kw * occ1
        internal_gain2 = 0.35 * base_and_appliance + self.internal_gain_per_occupant_kw * occ2
        solar_norm = self.data_interface.get_solar_irradiance_norm(self.current_time, self.current_time_index)
        q1_kw = self.zone_envelope_conductance_kw_per_c[0] * (self.outdoor_temp - float(indoor_temp1)) + self.interzone_conductance_kw_per_c * (float(indoor_temp2) - float(indoor_temp1)) + internal_gain1 + self.zone_solar_gain_peak_kw[0] * solar_norm + thermal_hvac1
        q2_kw = self.zone_envelope_conductance_kw_per_c[1] * (self.outdoor_temp - float(indoor_temp2)) + self.interzone_conductance_kw_per_c * (float(indoor_temp1) - float(indoor_temp2)) + internal_gain2 + self.zone_solar_gain_peak_kw[1] * solar_norm + thermal_hvac2
        rng1 = self.data_interface.scenario_rng(self.current_time, self.current_time_index, salt=411)
        rng2 = self.data_interface.scenario_rng(self.current_time, self.current_time_index, salt=412)
        next_temp1 = float(indoor_temp1) + dt_h * q1_kw / self.zone_thermal_capacity_kwh_per_c[0] + float(rng1.normal(0.0, self.ac_user_behavior_std))
        next_temp2 = float(indoor_temp2) + dt_h * q2_kw / self.zone_thermal_capacity_kwh_per_c[1] + float(rng2.normal(0.0, self.ac_user_behavior_std))
        return (float(np.clip(power1, 0.0, self.hvac_zone_max_power_kw)), float(np.clip(next_temp1, 5.0, 40.0)), float(np.clip(power2, 0.0, self.hvac_zone_max_power_kw)), float(np.clip(next_temp2, 5.0, 40.0)))

    def update_air_conditioner(self, set_temp, indoor_temp, zone_id: int=1):
        if int(zone_id) == 1:
            p1, t1, _p2, _t2 = self.update_hvac_system(set_temp, self.indoor_temp2, indoor_temp, self.indoor_temp2, self.state)
            return (p1, t1)
        _p1, _t1, p2, t2 = self.update_hvac_system(self.indoor_temp, set_temp, self.indoor_temp, indoor_temp, self.state)
        return (p2, t2)

    @staticmethod
    def _mixed_tank_draw(tank_temp_c, draw_volume_l, tank_volume_l, cold_temp_c, target_temp_c) -> Dict[str, float]:
        volume = float(draw_volume_l)
        tank_volume = float(tank_volume_l)
        initial = float(tank_temp_c)
        cold, target = (float(cold_temp_c), float(target_temp_c))
        if tank_volume <= 0.0 or volume < 0.0 or target <= cold or (initial < cold - 1e-08):
            raise ValueError('Invalid completely mixed tank draw inputs')
        excess = max(0.0, initial - cold)
        target_excess = target - cold
        at_target_l = min(volume, max(0.0, tank_volume * (excess - target_excess) / target_excess))
        excess_after_mixing = excess - at_target_l * target_excess / tank_volume
        withdrawal_mixed_l = tank_volume * np.log(excess / excess_after_mixing) if at_target_l > 0.0 else 0.0
        remaining_l = volume - at_target_l
        final_excess = excess_after_mixing * np.exp(-remaining_l / tank_volume)
        thermal_kwh = tank_volume * (excess - final_excess) * 4.18 / 3600.0
        mean_temp = cold + thermal_kwh / (volume * 4.18 / 3600.0) if volume > 0.0 else min(initial, target)
        return {'post_draw_temp_c': float(cold + final_excess), 'mean_delivery_temp_c': float(mean_temp), 'delivered_at_target_l': float(at_target_l), 'tank_withdrawal_l': float(withdrawal_mixed_l + remaining_l), 'delivered_energy_kwh_th': float(thermal_kwh)}

    @staticmethod
    def _ewh_thermal_transition(current_temp, draw_volume_l, power_kw, params, dt_h):
        result = HomeEnergyManagementEnv._mixed_tank_draw(current_temp, draw_volume_l, params['volume'], params.get('temp_cold', 10.0), params.get('delivery_temp_c', 42.0))
        thermal_capacity = float(params['mass']) * 4.18 / 3600.0
        mixed_temp = result['post_draw_temp_c']
        loss = max(0.0, float(params.get('standing_loss_ua_w_per_k', 1.07))) / 1000.0 * (mixed_temp - float(params.get('ambient_temp', 20.0))) * dt_h
        power = float(np.clip(power_kw, 0.0, params.get('heater_max_power_kw', 1.8)))
        heat = power * float(params.get('heater_efficiency', 0.98)) * dt_h
        unconstrained = mixed_temp + (heat - loss) / thermal_capacity
        final = float(np.clip(unconstrained, params.get('temp_min', 10.0), params.get('temp_max', 65.0)))
        result.update({'final_temp_c': final, 'heater_input_kwh_th': heat, 'standing_loss_kwh_th': float(loss), 'temperature_limit_spill_kwh_th': float((unconstrained - final) * thermal_capacity)})
        return result

    @staticmethod
    def _ewh_heating_power(setpoint, temperature, max_power):
        return float(np.clip(float(max_power) * (float(setpoint) - float(temperature) - 0.5) / 4.5, 0.0, float(max_power)))

    @staticmethod
    def _ewh_predict_guard_temperature(temperature, setpoint, expected_draws_l, params, dt_h):
        value = float(temperature)
        max_setpoint = float(params.get('temp_max', 65.0))
        max_power = float(params.get('heater_max_power_kw', 1.8))
        for offset, draw in enumerate(expected_draws_l):
            action = float(setpoint) if offset == 0 else max_setpoint
            power = HomeEnergyManagementEnv._ewh_heating_power(action, value, max_power)
            value = HomeEnergyManagementEnv._ewh_thermal_transition(value, draw, power, params, dt_h)['final_temp_c']
        return float(value)

    def _ewh_service_guard_context(self, state: Dict) -> Dict:
        params = self.ewh_params
        temperature = float(state['ewh_temp'])
        key = (str(self.current_time), int(self.current_time_index), temperature, int(self.steps_per_day), tuple(map(float, self.action_space['ewh_set_temp'])), tuple(sorted(((str(name), repr(value)) for name, value in params.items()))))
        cached = getattr(self, '_ewh_service_context_cache', None)
        if cached is not None and cached[0] == key:
            return dict(cached[1])
        start = float(params.get('service_support_start_hour', 5.0))
        end = float(params.get('service_support_end_hour', 23.0))
        offset = next((i for i in range(1, self.steps_per_day + 1) if start <= (self.current_time_index + i) % self.steps_per_day * 24.0 / self.steps_per_day <= end), self.steps_per_day)
        burst = float(params.get('service_design_burst_l', 48.0))
        cold = float(params.get('temp_cold', 10.0))
        outlet = float(params.get('delivery_temp_c', 42.0))
        required = outlet + burst / float(params['volume']) * (outlet - cold)
        expected_draws = [float(self._hot_water_draw_at_offset(i)) for i in range(offset)]
        candidates = []
        for setpoint in self.action_space['ewh_set_temp']:
            predicted = self._ewh_predict_guard_temperature(temperature, setpoint, expected_draws, params, self._dt_hours())
            candidates.append({'setpoint_c': float(setpoint), 'predicted_ready_temp_c': predicted, 'power_kw': self._ewh_heating_power(setpoint, temperature, params.get('heater_max_power_kw', 1.8)), 'feasible': bool(predicted >= required - 1e-08)})
        feasible = [item for item in candidates if item['feasible']]
        context = {'enabled': bool(params.get('service_guard_enabled', True)), 'active': True, 'steps_to_service': int(offset), 'required_temp_c': float(required), 'expected_draw_l': float(sum(expected_draws)), 'expected_draws_l': expected_draws, 'design_burst_l': burst, 'required_power_kw': min((item['power_kw'] for item in feasible), default=float(params.get('heater_max_power_kw', 1.8))), 'feasible': bool(feasible), 'recovery_unreachable': not bool(feasible), 'candidate_predictions': candidates, 'tank_temp_before_c': temperature}
        self._ewh_service_context_cache = (key, dict(context))
        return context

    def _ewh_setpoint_keeps_service_reachable(self, state: Dict, setpoint: float) -> bool:
        context = self._ewh_service_guard_context(state)
        if not context['enabled']:
            return True
        predicted = self._ewh_predict_guard_temperature(state['ewh_temp'], setpoint, context['expected_draws_l'], self.ewh_params, self._dt_hours())
        return bool(predicted >= context['required_temp_c'] - 1e-08)

    def _apply_ewh_viability_shield(self, state: Dict, requested_setpoint: float) -> float:
        context = self._ewh_service_guard_context(state)
        requested = float(requested_setpoint)
        executed = requested
        if context['enabled'] and (not self._ewh_setpoint_keeps_service_reachable(state, requested)):
            feasible = [item['setpoint_c'] for item in context['candidate_predictions'] if item['feasible']]
            executed = min(feasible, key=lambda value: (abs(value - requested), value)) if feasible else max(map(float, self.action_space['ewh_set_temp']))
        self.current_ewh_service_context = dict(context)
        self.current_ewh_service_context['executed_setpoint_c'] = float(executed)
        self.current_ewh_shield_intervened = int(abs(executed - requested) > 1e-08)
        self.current_ewh_shield_action_delta_c = float(executed - requested)
        self.current_ewh_service_infeasible = int(context['enabled'] and context['recovery_unreachable'])
        return float(executed)

    @staticmethod
    def _hot_water_shortfall_penalty(normalized_gap, linear_weight=2.0, quadratic_weight=2.0):
        x = max(0.0, float(normalized_gap))
        if x <= 1.0:
            return float(linear_weight * x + quadratic_weight * x * x)
        return float(linear_weight + quadratic_weight + (linear_weight + 2.0 * quadratic_weight) * (x - 1.0))

    def update_water_heater(self, power, current_temp):
        params = self.ewh_params
        dt_h = self._dt_hours()
        idx = int(self.current_time_index) % self.steps_per_day
        draw_volume_l = float(self._ewh_draw_profile_l[idx])
        flow_rate_lpm = float(self._ewh_draw_flow_lpm[idx])
        result = self._ewh_thermal_transition(current_temp, draw_volume_l, power, params, dt_h)
        self.current_ewh_draw_physics = result
        return (result['final_temp_c'], flow_rate_lpm, draw_volume_l, result['mean_delivery_temp_c'])

    def _fuzzy_heating_control(self, set_tem, current_tem):
        return self._ewh_heating_power(set_tem, current_tem, self.ewh_params.get('heater_max_power_kw', 1.8))

    def _tou_price_min_max(self) -> Tuple[float, float]:
        di = self.data_interface
        cache = getattr(di, '_tou_price_cache', None)
        if cache is not None:
            arr = np.asarray(cache, dtype=np.float64).ravel()
            if arr.size > 0:
                return (float(np.min(arr)), float(np.max(arr)))
        profile = getattr(di, 'price_profile', 'legacy')
        if profile == 'lee2020':
            return (0.06, 0.14)
        if profile == 'ireland_tou_2020':
            return (float(self.tariff_config.get('offpeak_eur_per_kwh', 0.13)), float(self.tariff_config.get('peak_eur_per_kwh', 0.24)))
        return (0.2, 0.8)

    def _resolve_project_path(self, path: str) -> str:
        if not path:
            return ''
        if os.path.isabs(path):
            return os.path.normpath(path)
        project_root = getattr(self, '_project_root', os.path.dirname(os.path.abspath(__file__)))
        return os.path.normpath(os.path.join(project_root, path))

    @staticmethod
    def _baseline_episode_key(house_id: Optional[str], date_str: Optional[str]) -> str:
        house = str(house_id or '').strip()
        date = HomeEnergyManagementEnv._normalize_date_str(date_str) if date_str else ''
        return f'{house}|{date}' if house else date

    def _load_peak_calibration(self, calibration_file: str) -> None:
        if not calibration_file:
            return
        path = self._resolve_project_path(calibration_file)
        if not os.path.isfile(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if str(data.get('split', '')).lower() != 'train':
            raise ValueError('peak calibration 必须来自 train split。')
        protocol_id = str(data.get('protocol_id', ''))
        if protocol_id != self.protocol_id:
            raise ValueError(f"peak calibration 协议不兼容: file={protocol_id or '<missing>'}, runtime={self.protocol_id}")
        calibration_reward_version = str(data.get('reward_design_version', ''))
        runtime_reward_version = str(CONFIG.get('protocol', {}).get('reward_design_version', ''))
        if calibration_reward_version != runtime_reward_version:
            raise ValueError(f"peak calibration reward 版本不兼容: file={calibration_reward_version or '<missing>'}, runtime={runtime_reward_version}")
        protocol_cfg = CONFIG.get('protocol', {})
        expected_split_seed = int(protocol_cfg.get('split_seed', 42))
        file_split_seed = int(data.get('split_seed', -1))
        if file_split_seed != expected_split_seed:
            raise ValueError(f'peak calibration split_seed 不兼容: file={file_split_seed}, runtime={expected_split_seed}')
        for field in ('environment_model_version', 'price_profile', 'tariff_source', 'weather_policy', 'terminal_settlement', 'load_model', 'date_split_policy', 'initial_state_policy', 'ev_boundary_policy', 'occupancy_policy', 'hot_water_policy', 'pv_policy'):
            file_value = str(data.get(field, ''))
            runtime_value = str(protocol_cfg.get(field, ''))
            if file_value != runtime_value:
                raise ValueError(f"peak calibration {field} 不兼容: file={file_value or '<missing>'}, runtime={runtime_value}")
        if data.get('frozen_parameters') != protocol_cfg.get('frozen_parameters'):
            raise ValueError('peak calibration 的冻结环境参数与当前代码不一致。')
        threshold = float(data.get('peak_threshold_kw', 0.0) or 0.0)
        if threshold <= 0.0:
            raise ValueError('peak calibration 中 peak_threshold_kw 必须大于 0。')
        self.peak_target_ratio = float(data.get('target_ratio', self.peak_target_ratio) or self.peak_target_ratio)
        self.peak_calibration_baseline_mean_kw = float(data.get('baseline_peak_mean_kw', 0.0) or 0.0)
        self.peak_threshold_kw = threshold
        self.peak_threshold_source = str(data.get('source', 'train_calibration'))
        self.peak_calibration_metadata = dict(data)
        self.peak_shaving_enabled = True

    def _configure_episode_peak_target(self) -> None:
        self.current_episode_baseline_peak_kw = float(self.peak_calibration_baseline_mean_kw)
        self.current_episode_peak_target_kw = float(self.peak_threshold_kw)

    @staticmethod
    def _comfort_score_from_discomfort(discomfort: float) -> float:
        return float(1.0 / (1.0 + max(0.0, float(discomfort))))

    def _piecewise_thermal_discomfort(self, temperature_c: float, comfort_bounds_c: Tuple[float, float], acceptable_bounds_c: Tuple[float, float]) -> float:
        temp = float(temperature_c)
        comfort_low, comfort_high = map(float, comfort_bounds_c)
        acceptable_low, acceptable_high = map(float, acceptable_bounds_c)
        if not acceptable_low < comfort_low <= comfort_high < acceptable_high:
            raise ValueError('热舒适边界必须满足 acceptable_low < comfort_low <= comfort_high < acceptable_high')
        if comfort_low <= temp <= comfort_high:
            return 0.0
        if temp < comfort_low:
            span = max(1e-08, comfort_low - acceptable_low)
            if temp >= acceptable_low:
                return float(self.acceptable_low_penalty_scale * ((comfort_low - temp) / span) ** 2)
            outside = (acceptable_low - temp) / span
            return float(self.acceptable_low_penalty_scale * (1.0 + self.unacceptable_low_linear_slope * outside))
        span = max(1e-08, acceptable_high - comfort_high)
        if temp <= acceptable_high:
            return float(self.acceptable_high_penalty_scale * ((temp - comfort_high) / span) ** 2)
        outside = (temp - acceptable_high) / span
        return float(self.acceptable_high_penalty_scale * (1.0 + self.unacceptable_high_linear_slope * outside))

    def _thermal_comfort_metrics(self, state: Dict) -> Dict[str, float]:
        temps = (float(state.get('indoor_temp', self.indoor_temp)), float(state.get('indoor_temp2', self.indoor_temp2)))
        occupancy = (float(state.get('zone1_occupied', 0.0)), float(state.get('zone2_occupied', 0.0)))
        discomforts: List[float] = []
        scores: List[float] = []
        safety_discomfort = 0.0
        for temp, occupied, bounds, acceptable_bounds in zip(temps, occupancy, self.zone_occupied_bounds_c, self.zone_acceptable_bounds_c):
            discomfort = self._piecewise_thermal_discomfort(temperature_c=temp, comfort_bounds_c=bounds, acceptable_bounds_c=acceptable_bounds)
            discomforts.append(float(discomfort))
            scores.append(self._comfort_score_from_discomfort(discomfort))
            if occupied <= 1e-08:
                unoccupied_mid = 0.5 * sum(self.unoccupied_bounds_c)
                unoccupied_deadband = 0.5 * (self.unoccupied_bounds_c[1] - self.unoccupied_bounds_c[0])
                safety_discomfort += self._quadratic_deadband_discomfort(temp, unoccupied_mid, deadband=unoccupied_deadband, scale=0.25, cap=10.0)
        total_weight = float(sum(occupancy))
        if total_weight > 1e-08:
            combined_score = float(np.average(np.asarray(scores), weights=np.asarray(occupancy)))
        else:
            combined_score = 1.0
        return {'ac1_discomfort': discomforts[0], 'ac2_discomfort': discomforts[1], 'ac1_score': scores[0], 'ac2_score': scores[1], 'ac_score': float(np.clip(combined_score, 0.0, 1.0)), 'service_weight': float(np.clip(total_weight, 0.0, 1.0)), 'safety_discomfort': float(safety_discomfort)}

    def _peak_shaving_penalty(self, grid_import_kw: float) -> Tuple[float, float, float, float, float]:
        target_kw = float(getattr(self, 'current_episode_peak_target_kw', self.peak_threshold_kw))
        if not self.peak_shaving_enabled or target_kw <= 0.0:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        grid_import_kw = max(0.0, float(grid_import_kw))
        excess_kw = max(0.0, grid_import_kw - target_kw)
        auc_penalty = excess_kw * self._dt_hours() / max(1e-08, target_kw * 24.0)
        previous_potential = float(getattr(self, 'current_episode_peak_potential', 0.0))
        self.current_episode_peak_import_kw = max(float(getattr(self, 'current_episode_peak_import_kw', 0.0)), grid_import_kw)
        peak_excess_kw = max(0.0, self.current_episode_peak_import_kw - target_kw)
        peak_potential = peak_excess_kw / max(1e-08, target_kw)
        peak_delta_penalty = max(0.0, peak_potential - previous_potential)
        self.current_episode_peak_potential = max(previous_potential, peak_potential)
        peak_memory_penalty = 0.0
        combined_penalty = self.peak_max_penalty_weight * peak_delta_penalty + self.peak_auc_penalty_weight * auc_penalty
        return (float(combined_penalty), float(excess_kw), float(peak_delta_penalty), float(auc_penalty), float(peak_memory_penalty))

    def _base_import_without_storage_kw(self, state: Dict) -> float:
        fixed_load_kw = float(state.get('home_load', 0.0)) + float(state.get('Air_conditioner_power', 0.0)) + float(state.get('Air_conditioner_power2', 0.0)) + float(state.get('wash_machine_state', 0.0)) * float(self.wash_machine_power) + float(state.get('ewh_power', 0.0)) - float(state.get('pv_generation', 0.0))
        return float(max(0.0, fixed_load_kw))

    def _peak_relief_reward(self, state: Dict, grid_import_kw: float) -> Tuple[float, float, float]:
        target_kw = float(getattr(self, 'current_episode_peak_target_kw', self.peak_threshold_kw))
        base_import_kw = self._base_import_without_storage_kw(state)
        if not self.peak_shaving_enabled or target_kw <= 0.0 or self.peak_relief_reward_weight <= 0.0:
            return (0.0, base_import_kw, 0.0)
        pressure_start_kw = 0.88 * target_kw
        pressure = float(np.clip((base_import_kw - pressure_start_kw) / max(1e-08, target_kw - pressure_start_kw), 0.0, 1.0))
        relief_kw = min(self.peak_relief_cap_kw, max(0.0, base_import_kw - max(0.0, float(grid_import_kw))))
        relief_norm = pressure * relief_kw / max(1e-08, target_kw)
        return (float(self.peak_relief_reward_weight * relief_norm), float(base_import_kw), float(relief_kw))

    def _load_carbon_intensity_table(self, project_root: str) -> None:
        self._carbon_df = None
        if not self.use_carbon_in_reward:
            self.carbon_source = 'disabled'
            return
        if not self._carbon_intensity_csv_path:
            raise ValueError('Registered carbon CSV path is required')
        p = self._carbon_intensity_csv_path
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(project_root, p))
        if not os.path.isfile(p):
            raise FileNotFoundError(f'Registered carbon CSV missing: {p}')
        try:
            df = pd.read_csv(p)
            df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
            gcol = None
            for name in ('carbon_intensity_gco2_per_kwh', 'carbon_intensity_gco2_per_kwh_roi', 'carbon_intensity_gco2_per_kwh_nsw'):
                if name in df.columns:
                    gcol = name
                    break
            if gcol is None:
                raise ValueError('no carbon intensity column (gCO2/kWh) in CSV')
            df['carbon_intensity_kgco2_per_kwh'] = pd.to_numeric(df[gcol], errors='coerce') / 1000.0
            df = df.sort_values('timestamp_utc').reset_index(drop=True)
            self._carbon_df = df
            values = df['carbon_intensity_kgco2_per_kwh'].to_numpy()
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError('Invalid carbon intensity values')
            if df['timestamp_utc'].duplicated().any():
                raise ValueError('Duplicate UTC carbon timestamps')
            self._carbon_kg_per_kwh_max = CONFIG['protocol']['information_protocol']['carbon_normalization_kg_per_kwh']
            self.carbon_source = os.path.basename(p)
            print(f'[Carbon] Loaded: {p} ({len(df)} rows)')
        except Exception as e:
            raise ValueError(f'Cannot load registered carbon data: {p}') from e

    def _simulation_time_utc(self, offset_steps: int=0) -> pd.Timestamp:
        date_str, step_idx = self._date_index_with_offset(offset_steps)
        naive = datetime.strptime(date_str, '%Y-%m-%d') + timedelta(minutes=60.0 * self._dt_hours() * int(step_idx))
        ts = pd.Timestamp(naive)
        try:
            loc = ts.tz_localize('Europe/Dublin', ambiguous='infer', nonexistent='shift_forward')
        except Exception:
            loc = ts.tz_localize('Europe/Dublin', ambiguous=False, nonexistent='shift_forward')
        return loc.tz_convert('UTC')

    def _normalized_price(self, state) -> float:
        price_raw = float(state.get('electricity_price', 0.0))
        return self._normalized_price_value(price_raw)

    def _carbon_intensity_kg_per_kwh_at_now(self) -> float:
        return self._carbon_intensity_kg_per_kwh_at_offset(0)

    def _carbon_intensity_kg_per_kwh_at_offset(self, offset_steps: int=0) -> float:
        if self._carbon_df is None or len(self._carbon_df) == 0:
            if not self.use_carbon_in_reward:
                return 0.0
            raise ValueError('Registered carbon data unavailable')
        utc_ts = self._simulation_time_utc(offset_steps)
        cdf = self._carbon_df
        index = int(cdf['timestamp_utc'].searchsorted(utc_ts, side='right')) - 1
        if index < 0:
            raise ValueError(f'Carbon data do not cover {utc_ts}')
        age = utc_ts - cdf['timestamp_utc'].iloc[index]
        max_age = CONFIG['protocol']['information_protocol']['carbon_max_age_minutes']
        if age >= pd.Timedelta(minutes=max_age):
            raise ValueError(f'Stale carbon data at {utc_ts}: {age}')
        return float(cdf['carbon_intensity_kgco2_per_kwh'].iloc[index])

    def _carbon_observation_dict(self) -> Dict[str, float]:
        if not self.use_carbon_in_reward:
            return {'carbon_intensity_norm': 0.0}
        ci = float(self._carbon_intensity_kg_per_kwh_at_now())
        return {'carbon_intensity_norm': self._normalized_carbon_value(ci)}

    @staticmethod
    def _asymmetric_quadratic_deadband(value, target, deadband=2.0, scale_high=2.0, scale_low=1.0, cap=50.0):
        diff = float(value) - float(target)
        if diff > deadband:
            return min(scale_high * (diff - deadband) ** 2, float(cap))
        elif diff < -deadband:
            return min(scale_low * (-diff - deadband) ** 2, float(cap))
        return 0.0

    def _dod_stress_penalty(self, soc, action_power, capacity):
        stress = 0.0
        if soc > 0.9 and action_power > 0:
            stress = (soc - 0.9) * 10.0 * action_power
        elif soc < 0.1 and action_power < 0:
            stress = (0.1 - soc) * 10.0 * abs(action_power)
        return float(stress)

    def _compute_ev_deadline_reward(self, state: Dict, action: Dict) -> float:
        self.current_ev_deadline_risk = 0.0
        self.current_ev_deadline_required_kw = 0.0
        self.current_ev_deadline_gap_kwh = 0.0
        self.current_ev_deadline_reward = 0.0
        self.current_ev_deadline_business_penalty = 0.0
        if not self.ev_deadline_enabled or not self.data_interface.is_ev_at_home(self.current_time, self.current_time_index):
            return 0.0
        ctx = self._forecast_context(state)
        metrics = self._ev_deadline_metrics(state, ctx)
        if int(metrics['steps_remaining']) <= 0 or float(metrics['actual_departure']) <= 0.5:
            return 0.0
        target = max(1e-08, float(metrics['target_kwh']))
        prev_gap = float(metrics['gap_kwh'])
        next_ev_kwh = float(self.state.get('ev_battery_state', state.get('ev_battery_state', 0.0)))
        next_gap = max(0.0, float(metrics['target_kwh']) - next_ev_kwh)
        risk_gate = float(metrics['risk_gate'])
        business_multiplier = float(metrics['business_multiplier'])
        progress_kwh = prev_gap - next_gap
        progress_reward = self.ev_deadline_reward_weight * (progress_kwh / target) * risk_gate * business_multiplier
        progress_reward = float(np.clip(progress_reward, -abs(self.ev_deadline_reward_cap), abs(self.ev_deadline_reward_cap)))
        lag_penalty = -self.ev_deadline_lag_penalty_weight * risk_gate * (next_gap / target)
        departing_after_step = float(metrics.get('actual_departure', 0.0)) > 0.5 and int(metrics.get('departure_step', -1)) == int(self.current_time_index)
        terminal_penalty = 0.0
        if departing_after_step:
            min_shortfall = max(0.0, float(metrics['hard_target_kwh']) - next_ev_kwh)
            terminal_penalty = -self.ev_deadline_terminal_penalty_weight * (min_shortfall / max(1e-08, float(metrics['hard_target_kwh'])))
        reward = float(progress_reward + lag_penalty + terminal_penalty)
        self.current_ev_deadline_risk = float(metrics['risk'])
        self.current_ev_deadline_required_kw = float(metrics['required_kw'])
        self.current_ev_deadline_gap_kwh = float(next_gap)
        self.current_ev_deadline_reward = reward
        self.current_ev_deadline_business_penalty = float(metrics['business_penalty'])
        return reward

    @staticmethod
    def _build_reward_state(prev_state: Dict, next_state: Optional[Dict]) -> Dict:
        reward_state = dict(prev_state)
        if not next_state:
            return reward_state
        for key in ('ess_state', 'ev_battery_state', 'indoor_temp', 'indoor_temp2', 'wash_machine_state', 'Air_conditioner_power', 'Air_conditioner_power2', 'ewh_temp', 'ewh_power'):
            if key in next_state:
                reward_state[key] = next_state[key]
        return reward_state

    def _reset_step_energy_flow_metrics(self) -> None:
        self.current_step_ess_charge_input_kwh = 0.0
        self.current_step_ess_discharge_output_kwh = 0.0
        self.current_step_ess_charge_from_pv_kwh = 0.0
        self.current_step_ess_charge_non_pv_kwh = 0.0
        self.current_step_ess_discharge_to_load_kwh = 0.0
        self.current_step_ess_discharge_to_grid_kwh = 0.0
        self.current_step_pv_generation_kwh = 0.0
        self.current_step_pv_self_consumed_kwh = 0.0
        self.current_step_pv_export_kwh = 0.0
        self.current_step_base_load_kwh = 0.0
        self.current_step_hvac_energy_kwh = 0.0
        self.current_step_ewh_energy_kwh = 0.0
        self.current_step_wash_energy_kwh = 0.0
        self.current_step_ev_charge_input_kwh = 0.0
        self.current_step_ev_discharge_output_kwh = 0.0
        self.current_step_ev_discharge_to_load_kwh = 0.0
        self.current_step_ev_discharge_to_grid_kwh = 0.0
        self.current_step_energy_balance_residual_kwh = 0.0

    def _update_step_energy_flow_metrics(self, state: Dict, reward_state: Dict, action: Dict, dt_h: float) -> None:
        self._reset_step_energy_flow_metrics()
        base_kw = max(0.0, float(state.get('home_load', 0.0)))
        hvac_kw = max(0.0, float(reward_state.get('Air_conditioner_power', 0.0))) + max(0.0, float(reward_state.get('Air_conditioner_power2', 0.0)))
        ewh_kw = max(0.0, float(reward_state.get('ewh_power', 0.0)))
        wash_kw = max(0.0, float(reward_state.get('wash_machine_state', 0.0))) * float(self.wash_machine_power)
        pv_kw = max(0.0, float(state.get('pv_generation', 0.0)))
        ess_charge_kw = max(0.0, float(action.get('battery_power', 0.0)))
        ess_discharge_kw = max(0.0, -float(action.get('battery_power', 0.0)))
        ev_charge_kw = max(0.0, float(action.get('ev_power', 0.0)))
        ev_discharge_kw = max(0.0, -float(action.get('ev_power', 0.0)))
        non_ess_demand_kw = base_kw + hvac_kw + ewh_kw + wash_kw + ev_charge_kw
        local_demand_kw = non_ess_demand_kw + ess_charge_kw
        total_local_generation_kw = pv_kw + ev_discharge_kw + ess_discharge_kw
        grid_import_kw = max(0.0, local_demand_kw - total_local_generation_kw)
        grid_export_kw = max(0.0, total_local_generation_kw - local_demand_kw)
        if total_local_generation_kw > 1e-12:
            export_fraction = grid_export_kw / total_local_generation_kw
        else:
            export_fraction = 0.0
        pv_export_kw = pv_kw * export_fraction
        ev_export_kw = ev_discharge_kw * export_fraction
        ess_export_kw = ess_discharge_kw * export_fraction
        pv_self_used_kw = pv_kw - pv_export_kw
        ev_to_load_kw = ev_discharge_kw - ev_export_kw
        ess_to_load_kw = ess_discharge_kw - ess_export_kw
        pv_surplus_kw = max(0.0, pv_kw - non_ess_demand_kw)
        ess_from_pv_kw = min(ess_charge_kw, pv_surplus_kw)
        self.current_step_ess_charge_input_kwh = ess_charge_kw * dt_h
        self.current_step_ess_discharge_output_kwh = ess_discharge_kw * dt_h
        self.current_step_ess_charge_from_pv_kwh = ess_from_pv_kw * dt_h
        self.current_step_ess_charge_non_pv_kwh = max(0.0, ess_charge_kw - ess_from_pv_kw) * dt_h
        self.current_step_ess_discharge_to_load_kwh = ess_to_load_kw * dt_h
        self.current_step_ess_discharge_to_grid_kwh = ess_export_kw * dt_h
        self.current_step_pv_generation_kwh = pv_kw * dt_h
        self.current_step_pv_self_consumed_kwh = pv_self_used_kw * dt_h
        self.current_step_pv_export_kwh = pv_export_kw * dt_h
        self.current_step_base_load_kwh = base_kw * dt_h
        self.current_step_hvac_energy_kwh = hvac_kw * dt_h
        self.current_step_ewh_energy_kwh = ewh_kw * dt_h
        self.current_step_wash_energy_kwh = wash_kw * dt_h
        self.current_step_ev_charge_input_kwh = ev_charge_kw * dt_h
        self.current_step_ev_discharge_output_kwh = ev_discharge_kw * dt_h
        self.current_step_ev_discharge_to_load_kwh = ev_to_load_kw * dt_h
        self.current_step_ev_discharge_to_grid_kwh = ev_export_kw * dt_h
        self.current_step_energy_balance_residual_kwh = abs(grid_import_kw + total_local_generation_kw - local_demand_kw - grid_export_kw) * dt_h

    def calculate_reward(self, state, action, next_state: Optional[Dict]=None):
        reward_state = self._build_reward_state(state, next_state)
        step_h = 24.0 / float(self.steps_per_day)
        ev_charge = max(action['ev_power'], 0)
        ev_discharge = max(-action['ev_power'], 0)
        battery_charge = max(action['battery_power'], 0)
        battery_discharge = max(-action['battery_power'], 0)
        total_consumption = state['home_load'] + ev_charge + battery_charge + reward_state['Air_conditioner_power'] + reward_state['Air_conditioner_power2'] + reward_state['wash_machine_state'] * self.wash_machine_power + reward_state['ewh_power']
        total_generation = state['pv_generation'] + ev_discharge + battery_discharge
        net_demand = total_consumption - total_generation
        self._update_step_energy_flow_metrics(state, reward_state, action, step_h)
        purchase_kwh = max(net_demand, 0) * step_h
        sell_kwh = max(-net_demand, 0) * step_h
        export_price = float(state.get('grid_export_price', self.data_interface.get_grid_export_price(self.current_time, self.current_time_index)))
        energy_cost = purchase_kwh * state['electricity_price'] - sell_kwh * export_price
        grid_import_kwh = max(net_demand, 0.0) * step_h
        grid_export_kwh = max(-net_demand, 0.0) * step_h
        self.current_step_net_load_kw = float(net_demand)
        self.current_step_grid_import_kw = float(max(net_demand, 0.0))
        self.current_step_grid_export_kw = float(max(-net_demand, 0.0))
        self.current_step_grid_import_kwh = float(grid_import_kwh)
        self.current_step_grid_export_kwh = float(grid_export_kwh)
        self.current_step_grid_import_cost = float(purchase_kwh * state['electricity_price'])
        self.current_step_grid_export_price = float(export_price)
        self.current_step_grid_export_revenue = float(sell_kwh * export_price)
        self.current_step_net_energy_cost = float(energy_cost)
        self.total_cost += energy_cost
        self.records['energy_cost'].append(self.total_cost)
        self.current_step_cost = energy_cost
        self._update_ess_energy_source_bookkeeping(state, action, float(total_consumption), float(net_demand))
        ci_kg = self._carbon_intensity_kg_per_kwh_at_now()
        self.last_step_carbon_intensity_kg_per_kwh = float(ci_kg)
        emission_kg = ci_kg * grid_import_kwh
        self.current_step_emission_kg = float(emission_kg)
        carbon_raw = emission_kg if self.use_carbon_in_reward else 0.0
        dt_h = float(step_h)
        thermal_comfort = self._thermal_comfort_metrics(reward_state)
        ac1_dis = float(thermal_comfort['ac1_discomfort'])
        ac2_dis = float(thermal_comfort['ac2_discomfort'])
        ac1_comfort_score = float(thermal_comfort['ac1_score'])
        ac2_comfort_score = float(thermal_comfort['ac2_score'])
        ac_comfort_score = float(thermal_comfort['ac_score'])
        ac_comfort_service_weight = float(thermal_comfort['service_weight'])
        ac_unoccupied_safety_discomfort = float(thermal_comfort['safety_discomfort'])
        hour = self._hour_of_step(state['time_index'])
        hot_water_service_window = 6 <= hour <= 9 or 18 <= hour <= 22
        hot_water_draw_event = float(getattr(self, 'current_step_hot_water_draw_l', 0.0)) > 1e-09
        hot_water_service_temp = float(self.current_step_hot_water_delivery_temp_c)
        ewh_dis = self._asymmetric_quadratic_deadband(value=hot_water_service_temp, target=float(self.hot_water_service_temp_c), deadband=1.0, scale_high=1.0, scale_low=3.0, cap=50.0) if hot_water_draw_event else 0.0
        comfort_legacy_raw = max(0.0, 10.0 - ac1_dis) + max(0.0, 10.0 - ac2_dis) + max(0.0, 10.0 - ewh_dis)
        ewh_comfort_score = self._comfort_score_from_discomfort(ewh_dis)
        combined_comfort_score = float(np.clip(0.5 * ac_comfort_score + 0.5 * ewh_comfort_score, 0.0, 1.0))
        comfort_score = float(np.clip(ac_comfort_score, 0.0, 1.0))
        hot_water_shortfall_c = max(0.0, self.hot_water_service_temp_c - hot_water_service_temp) if hot_water_draw_event else 0.0
        hot_water_overheat_c = max(0.0, float(reward_state['ewh_temp']) - self.hot_water_overheat_temp_c)
        hot_water_shortfall_norm = hot_water_shortfall_c / self.hot_water_shortfall_scale_c
        hot_water_service_score = float(np.clip(1.0 - hot_water_shortfall_norm, 0.0, 1.0))
        hot_water_success = float(hot_water_draw_event and hot_water_shortfall_c <= 1e-06)
        daily_hot_water_reference_l = max(1e-08, float(self.ewh_params.get('daily_mixed_water_mean_l', 65.0)))
        hot_water_volume_weight = float(self.current_step_hot_water_draw_l) / daily_hot_water_reference_l if hot_water_draw_event else 0.0
        hot_water_service_weight = self.hot_water_draw_event_weight * hot_water_volume_weight
        ess_dod_stress = self._dod_stress_penalty(state['ess_state'] / max(1e-08, self.ess_capacity), action.get('battery_power', 0.0), self.ess_capacity)
        ev_dod_stress = self._dod_stress_penalty(state['ev_battery_state'] / max(1e-08, self.ev_capacity), action.get('ev_power', 0.0), self.ev_capacity)
        ess_fec_step = float(abs(action.get('battery_power', 0.0)) * dt_h / (2.0 * self.ess_capacity))
        ev_fec_step = float(abs(action.get('ev_power', 0.0)) * dt_h / (2.0 * self.ev_capacity))
        ess_degradation_cost_eur = ess_fec_step * self.ess_replacement_cost_eur / self.ess_cycle_life_fec
        ev_degradation_cost_eur = ev_fec_step * self.ev_replacement_cost_eur / self.ev_cycle_life_fec
        aging_raw = ess_degradation_cost_eur + ev_degradation_cost_eur
        next_ess_soc = float(reward_state.get('ess_state', state['ess_state'])) / max(1e-08, self.ess_capacity)
        next_ev_kwh = float(reward_state.get('ev_battery_state', state['ev_battery_state']))
        next_ev_soc = next_ev_kwh / max(1e-08, self.ev_capacity)
        ev_at_home_now = bool(self.data_interface.is_ev_at_home(self.current_time, self.current_time_index))
        ess_edge_health_raw = self.ess_low_soc_weight * max(0.0, self.ess_healthy_soc_low - next_ess_soc) ** 2 + self.ess_high_soc_weight * max(0.0, next_ess_soc - self.ess_healthy_soc_high) ** 2
        ess_terminal_health_raw = 0.0
        ev_edge_health_raw = 0.0
        if ev_at_home_now:
            ev_edge_health_raw = self.ev_low_soc_weight * max(0.0, self.ev_healthy_soc_low - next_ev_soc) ** 2 + self.ev_high_soc_weight * max(0.0, next_ev_soc - self.ev_healthy_soc_high) ** 2
        battery_throughput_health_raw = 0.0
        ess_health_raw = ess_edge_health_raw
        battery_health_raw = ess_edge_health_raw + ev_edge_health_raw + battery_throughput_health_raw
        violation = 0.0
        wash_penalty_raw = self.calculate_user_dissatisfaction2(state, action)
        pv_excess = max(0.0, float(state['pv_generation']) - float(state['home_load']))
        pv_to_battery = min(battery_charge, pv_excess)
        pv_self_use_kwh = float(pv_to_battery) * dt_h
        price = self._normalized_price(state)
        peak_shave_kwh = 0.0
        if price > 0.7:
            peak_shave_kwh = float(battery_discharge) * dt_h * (price - 0.7)
        peak_penalty_norm, peak_excess_kw, peak_max_delta_penalty, peak_auc_penalty, peak_memory_penalty = self._peak_shaving_penalty(self.current_step_grid_import_kw)
        peak_relief_reward_norm, base_import_no_storage_kw, peak_relief_kw = self._peak_relief_reward(reward_state, self.current_step_grid_import_kw)
        cost_norm = self.reward_scaler.scale('cost', float(energy_cost))
        comfort_norm = self.reward_scaler.scale('comfort', float(comfort_score))
        carbon_norm = self.reward_scaler.scale('carbon', float(carbon_raw))
        aging_norm = self.reward_scaler.scale('aging', float(aging_raw))
        battery_health_norm = self.reward_scaler.scale('battery_health', float(battery_health_raw))
        pv_self_norm = self.reward_scaler.scale('pv_self_use', float(pv_self_use_kwh))
        peak_norm = self.reward_scaler.scale('peak_shave', float(peak_shave_kwh))
        comfort_target = float(np.clip(self.ac_comfort_target_score, 1e-06, 1.0))
        comfort_saturation = float(np.clip(max(self.ac_comfort_saturation_score, comfort_target + 1e-06), comfort_target + 1e-06, 1.0))
        comfort_deficit = ac_comfort_service_weight * max(0.0, comfort_target - comfort_score)
        comfort_base_bonus_raw = ac_comfort_service_weight * min(comfort_score, comfort_target) / comfort_target
        comfort_mid_bonus_raw = 0.0
        comfort_saturation_bonus_raw = 0.0
        if ac_comfort_service_weight > 0.0 and comfort_score > comfort_target:
            mid_span = max(1e-08, comfort_saturation - comfort_target)
            comfort_mid_bonus_raw = self.comfort_mid_reward_weight * min(comfort_score - comfort_target, mid_span) / mid_span
            if comfort_score > comfort_saturation:
                comfort_saturation_bonus_raw = self.comfort_saturation_reward_weight * (comfort_score - comfort_saturation) / max(1e-08, 1.0 - comfort_saturation)
        comfort_bonus_raw = comfort_base_bonus_raw + comfort_mid_bonus_raw + comfort_saturation_bonus_raw
        comfort_penalty_uncapped_raw = self.comfort_deficit_linear_weight * comfort_deficit + self.comfort_deficit_quadratic_weight * comfort_deficit ** 2 + 0.05 * ac_unoccupied_safety_discomfort
        comfort_penalty_raw = min(self.ac_comfort_penalty_cap, comfort_penalty_uncapped_raw)
        comfort_barrier_raw = 0.0
        comfort_reward = self.w_comfort * (comfort_bonus_raw - comfort_penalty_raw)
        hot_water_bonus_raw = self.hot_water_service_bonus_weight * hot_water_service_score * hot_water_service_weight if hot_water_draw_event else 0.0
        hot_water_penalty_uncapped_raw = hot_water_service_weight * self._hot_water_shortfall_penalty(hot_water_shortfall_norm, self.hot_water_deficit_linear_weight, self.hot_water_deficit_quadratic_weight)
        hot_water_penalty_raw = hot_water_penalty_uncapped_raw
        hot_water_overheat_penalty_raw = self.hot_water_overheat_weight * (hot_water_overheat_c / self.hot_water_shortfall_scale_c)
        hot_water_service_reward = self.w_hot_water_service * (hot_water_bonus_raw - hot_water_penalty_raw - hot_water_overheat_penalty_raw)
        peak_gate_span = max(1e-08, float(self.peak_relief_full_comfort_score - self.peak_relief_min_comfort_score))
        peak_comfort_gate = float(np.clip((comfort_score - self.peak_relief_min_comfort_score) / peak_gate_span, 0.0, 1.0))
        if ev_at_home_now:
            operational_reserve_kwh = self._ev_operational_reserve_kwh()
            peak_ev_mobility_gate = float(np.clip((next_ev_kwh - operational_reserve_kwh) / self.peak_relief_ev_reserve_margin_kwh, 0.0, 1.0))
        else:
            peak_ev_mobility_gate = 1.0
        peak_reward_gate = min(peak_comfort_gate, peak_ev_mobility_gate)
        peak_relief_reward_gated_norm = peak_relief_reward_norm * peak_reward_gate
        peak_shave_reward_term = self.w_peak_shave * (peak_relief_reward_gated_norm - peak_penalty_norm)
        battery_health_penalty = -self.w_battery_health * battery_health_norm
        energy_cost_reward = -self.w_cost * cost_norm
        carbon_reward = -self.w_carbon * carbon_norm
        aging_reward = -self.w_aging * aging_norm
        self.current_ev_reserve_floor_kwh = self._ev_operational_reserve_kwh()
        self.current_ev_reserve_gap_kwh = max(0.0, self.current_ev_reserve_floor_kwh - next_ev_kwh)
        core_reward = energy_cost_reward + comfort_reward + hot_water_service_reward + carbon_reward + aging_reward + battery_health_penalty + peak_shave_reward_term
        self.current_core_reward = float(core_reward)
        ev_deadline_reward = self._compute_ev_deadline_reward(state, action)
        ev_shield_reward = -self.ev_shield_intervention_weight * (abs(float(self.current_ev_shield_action_delta_kw)) / self._max_abs_action_power('ev_power'))
        self.current_terminal_energy_penalty, self.current_terminal_ess_deficit_kwh, self.current_terminal_ev_deficit_kwh = self._terminal_energy_penalty(reward_state)
        terminal_energy_reward = -float(self.current_terminal_energy_penalty)
        boundary_recovery_penalty_raw = self._boundary_recovery_penalty(reward_state)
        boundary_gap_metrics = self._boundary_gap_metrics(reward_state)
        boundary_recovery_reward = -self.w_boundary_recovery * boundary_recovery_penalty_raw
        violation_reward = -self.w_violation * violation
        smoothness_reward = -self.w_smoothness * self.current_smoothness_penalty
        raw_action_event_penalty = self.raw_action_event_weight * float(self.current_step_raw_violations)
        raw_action_distance_penalty = self.raw_action_distance_weight * float(self.current_invalid_action_dist)
        raw_action_repair_penalty = raw_action_event_penalty + raw_action_distance_penalty
        self.current_invalid_action_penalty = float(raw_action_repair_penalty)
        invalid_action_reward = -raw_action_repair_penalty
        user_reward = -self.w_wash * wash_penalty_raw
        reward = core_reward + ev_deadline_reward + ev_shield_reward + terminal_energy_reward + boundary_recovery_reward + violation_reward + smoothness_reward + invalid_action_reward + user_reward
        self._last_reward_components = {'comfort_raw': float(comfort_score), 'cost_raw': float(cost_norm), 'carbon_raw': float(carbon_norm), 'aging_raw': float(aging_norm), 'aging_cost_eur': float(aging_raw), 'ess_degradation_cost_eur': float(ess_degradation_cost_eur), 'ev_degradation_cost_eur': float(ev_degradation_cost_eur), 'ess_dod_stress_raw': float(ess_dod_stress), 'ev_dod_stress_raw': float(ev_dod_stress), 'battery_health_raw': float(battery_health_norm), 'battery_health_unnorm': float(battery_health_raw), 'ess_health_raw': float(ess_health_raw), 'ev_health_raw': float(ev_edge_health_raw), 'ess_edge_health_raw': float(ess_edge_health_raw), 'ess_terminal_health_raw': float(ess_terminal_health_raw), 'battery_throughput_health_raw': float(battery_throughput_health_raw), 'ess_soc_health_raw': float(ess_edge_health_raw), 'ev_soc_health_raw': float(ev_edge_health_raw), 'battery_throughput_raw': float(battery_throughput_health_raw), 'ess_terminal_reserve_raw': float(ess_terminal_health_raw), 'violation_raw': float(violation), 'smoothness_raw': float(self.current_smoothness_penalty), 'battery_switching_raw': float(self.current_step_battery_switching_cost), 'ess_switching_raw': float(self.current_step_ess_switching_cost), 'ev_switching_raw': float(self.current_step_ev_switching_cost), 'ess_ramp_norm': float(self.current_step_ess_ramp_norm), 'ev_ramp_norm': float(self.current_step_ev_ramp_norm), 'ramp_auc_raw': float(self.current_step_ess_ramp_norm + self.current_step_ev_ramp_norm), 'high_power_reversal_count': float(self.current_step_high_power_reversal_count), 'micro_cycle_count': float(self.current_step_micro_cycle_count), 'invalid_raw': float(raw_action_repair_penalty), 'invalid_distance_raw': float(self.current_invalid_action_dist), 'physical_repair_distance_kw': float(self.current_invalid_action_dist_kw), 'physical_repair_distance_norm': float(self.current_invalid_action_dist), 'ess_repair_distance_norm': float(self.current_ess_repair_distance_norm), 'ev_repair_distance_norm': float(self.current_ev_repair_distance_norm), 'wash_raw': float(wash_penalty_raw), 'pv_self_raw': float(pv_self_norm), 'peak_shave_raw': float(peak_norm), 'peak_penalty_raw': float(peak_penalty_norm), 'peak_relief_raw': float(peak_relief_reward_norm), 'peak_relief_gated_raw': float(peak_relief_reward_gated_norm), 'peak_reward_gate': float(peak_reward_gate), 'peak_comfort_gate': float(peak_comfort_gate), 'peak_ev_mobility_gate': float(peak_ev_mobility_gate), 'peak_relief_kw': float(peak_relief_kw), 'base_import_no_storage_kw': float(base_import_no_storage_kw), 'peak_max_delta_raw': float(peak_max_delta_penalty), 'peak_auc_raw': float(peak_auc_penalty), 'peak_memory_raw': float(peak_memory_penalty), 'peak_excess_kw': float(peak_excess_kw), 'episode_peak_grid_import_kw': float(getattr(self, 'current_episode_peak_import_kw', 0.0)), 'episode_peak_potential_raw': float(getattr(self, 'current_episode_peak_potential', 0.0)), 'peak_threshold_kw': float(self.peak_threshold_kw), 'peak_target_kw': float(getattr(self, 'current_episode_peak_target_kw', self.peak_threshold_kw)), 'peak_baseline_kw': float(getattr(self, 'current_episode_baseline_peak_kw', 0.0)), 'comfort_unnorm': float(comfort_legacy_raw), 'comfort_score': float(comfort_score), 'combined_comfort_score': float(combined_comfort_score), 'ac1_comfort_score': float(ac1_comfort_score), 'ac2_comfort_score': float(ac2_comfort_score), 'ac_comfort_score': float(ac_comfort_score), 'ac_comfort_service_weight': float(ac_comfort_service_weight), 'ac_unoccupied_safety_discomfort': float(ac_unoccupied_safety_discomfort), 'ewh_comfort_score': float(ewh_comfort_score), 'comfort_target_score': float(comfort_target), 'comfort_deficit': float(comfort_deficit), 'comfort_saturation_score': float(comfort_saturation), 'comfort_base_bonus_raw': float(comfort_base_bonus_raw), 'comfort_mid_bonus_raw': float(comfort_mid_bonus_raw), 'comfort_saturation_bonus_raw': float(comfort_saturation_bonus_raw), 'comfort_bonus_raw': float(comfort_bonus_raw), 'comfort_penalty_raw': float(comfort_penalty_raw), 'comfort_penalty_uncapped_raw': float(comfort_penalty_uncapped_raw), 'comfort_barrier_raw': float(comfort_barrier_raw), 'hot_water_service_score': float(hot_water_service_score), 'hot_water_volume_weight': float(hot_water_volume_weight), 'hot_water_service_success': float(hot_water_success), 'hot_water_shortfall_c': float(hot_water_shortfall_c), 'hot_water_overheat_c': float(hot_water_overheat_c), 'hot_water_service_window': float(hot_water_service_window), 'hot_water_draw_event': float(hot_water_draw_event), 'hot_water_required_temp_c': float(self.hot_water_service_temp_c), 'hot_water_delivery_temp_c': float(hot_water_service_temp), 'hot_water_draw_l': float(self.current_step_hot_water_draw_l), 'hot_water_delivered_at_target_l': float(getattr(self, 'current_ewh_draw_physics', {}).get('delivered_at_target_l', 0.0)), 'hot_water_tank_withdrawal_l': float(getattr(self, 'current_ewh_draw_physics', {}).get('tank_withdrawal_l', 0.0)), 'hot_water_delivered_energy_kwh_th': float(getattr(self, 'current_ewh_draw_physics', {}).get('delivered_energy_kwh_th', 0.0)), 'hot_water_post_draw_temp_c': float(getattr(self, 'current_ewh_draw_physics', {}).get('post_draw_temp_c', reward_state['ewh_temp'])), 'hot_water_heater_input_kwh_th': float(getattr(self, 'current_ewh_draw_physics', {}).get('heater_input_kwh_th', 0.0)), 'hot_water_standing_loss_kwh_th': float(getattr(self, 'current_ewh_draw_physics', {}).get('standing_loss_kwh_th', 0.0)), 'hot_water_temperature_limit_spill_kwh_th': float(getattr(self, 'current_ewh_draw_physics', {}).get('temperature_limit_spill_kwh_th', 0.0)), 'hot_water_outside_design_burst': float(self.current_step_hot_water_draw_l > float(self.ewh_params.get('service_design_burst_l', 48.0)) + 1e-08), 'ewh_shield_intervened': float(getattr(self, 'current_ewh_shield_intervened', 0)), 'ewh_shield_action_delta_c': float(getattr(self, 'current_ewh_shield_action_delta_c', 0.0)), 'ewh_service_infeasible': float(getattr(self, 'current_ewh_service_infeasible', 0)), 'ewh_service_required_temp_c': float(getattr(self, 'current_ewh_service_context', {}).get('required_temp_c', 0.0)), 'hot_water_penalty_raw': float(hot_water_penalty_raw), 'hot_water_penalty_uncapped_raw': float(hot_water_penalty_uncapped_raw), 'hot_water_overheat_penalty_raw': float(hot_water_overheat_penalty_raw), 'hot_water_bonus_raw': float(hot_water_bonus_raw), 'energy_cost_reward': float(energy_cost_reward), 'carbon_reward': float(carbon_reward), 'aging_reward': float(aging_reward), 'carbon_unnorm': float(carbon_raw), 'aging_fec_ess': float(ess_fec_step), 'aging_fec_ev': float(ev_fec_step), 'core_reward': float(core_reward), 'total_reward': float(reward), 'temp_reward': float(comfort_reward + hot_water_service_reward), 'ac_comfort_reward': float(comfort_reward), 'hot_water_service_reward': float(hot_water_service_reward), 'comfort_bonus': float(self.w_comfort * comfort_bonus_raw), 'comfort_penalty': float(-self.w_comfort * comfort_penalty_raw), 'peak_shave_reward': float(peak_shave_reward_term), 'battery_health_penalty': float(battery_health_penalty), 'ev_deadline_reward': float(ev_deadline_reward), 'ev_shield_reward': float(ev_shield_reward), 'ev_shield_intervened': float(self.current_ev_shield_intervened), 'ev_shield_action_delta_kw': float(self.current_ev_shield_action_delta_kw), 'ev_forced_charge_kwh': float(self.current_ev_forced_charge_kwh), 'terminal_energy_reward': float(terminal_energy_reward), 'boundary_recovery_reward': float(boundary_recovery_reward), 'boundary_recovery_penalty_raw': float(boundary_recovery_penalty_raw), 'boundary_ess_gap_kwh': float(boundary_gap_metrics['ess_gap_kwh']), 'boundary_ev_gap_kwh': float(boundary_gap_metrics['ev_gap_kwh']), 'boundary_ewh_gap_kwh_th': float(boundary_gap_metrics['ewh_gap_kwh_th']), 'boundary_hvac_gap_kwh_th': float(boundary_gap_metrics['hvac_gap_kwh_th']), 'boundary_hvac_zone1_gap_kwh_th': float(boundary_gap_metrics['hvac_zone1_gap_kwh_th']), 'boundary_hvac_zone2_gap_kwh_th': float(boundary_gap_metrics['hvac_zone2_gap_kwh_th']), 'boundary_hvac_zone1_gap_norm': float(np.clip(boundary_gap_metrics['hvac_zone1_gap_kwh_th'] / max(1e-08, self.zone_thermal_capacity_kwh_per_c[0] * 8.0), 0.0, 1.0)), 'boundary_hvac_zone2_gap_norm': float(np.clip(boundary_gap_metrics['hvac_zone2_gap_kwh_th'] / max(1e-08, self.zone_thermal_capacity_kwh_per_c[1] * 8.0), 0.0, 1.0)), 'boundary_grid_equivalent_kwh': float(boundary_gap_metrics['grid_equivalent_kwh']), **{key: float(value) for key, value in self._boundary_observation_dict(reward_state).items() if key in {'ess_boundary_gap_norm', 'ev_boundary_gap_norm', 'ewh_boundary_gap_norm', 'hvac_boundary_gap_norm', 'boundary_recovery_risk'}}, 'terminal_energy_penalty': float(self.current_terminal_energy_penalty), 'terminal_ess_deficit_kwh': float(self.current_terminal_ess_deficit_kwh), 'terminal_ev_deficit_kwh': float(self.current_terminal_ev_deficit_kwh), 'terminal_ewh_deficit_kwh_th': float(self.current_terminal_ewh_deficit_kwh_th), 'terminal_hvac_deficit_kwh_th': float(self.current_terminal_hvac_deficit_kwh_th), 'terminal_grid_equivalent_kwh': float(self.current_terminal_grid_equivalent_kwh), 'terminal_settlement_cost_eur': float(self.current_terminal_settlement_cost_eur), 'terminal_settlement_carbon_kg': float(self.current_terminal_settlement_carbon_kg), 'initial_ev_feasibility_adjustment_kwh': float(self.current_initial_ev_feasibility_adjustment_kwh), 'ev_operational_reserve_kwh': float(self.current_ev_reserve_floor_kwh), 'ev_operational_reserve_gap_kwh': float(self.current_ev_reserve_gap_kwh), 'ev_service_infeasible': float(getattr(self, 'current_ev_service_infeasible', 0)), 'ev_service_preexisting_reserve_gap_kwh': float(getattr(self, 'current_ev_service_context', {}).get('preexisting_reserve_gap_kwh', 0.0)), 'violation_reward': float(violation_reward), 'smoothness_reward': float(smoothness_reward), 'invalid_action_reward': float(invalid_action_reward), 'raw_action_event_penalty': float(raw_action_event_penalty), 'raw_action_distance_penalty': float(raw_action_distance_penalty), 'raw_action_repair_penalty': float(raw_action_repair_penalty), 'user_reward': float(user_reward), 'ev_deadline_risk': float(self.current_ev_deadline_risk), 'ev_deadline_required_kw': float(self.current_ev_deadline_required_kw), 'ev_deadline_gap_kwh': float(self.current_ev_deadline_gap_kwh), 'ev_deadline_business_penalty': float(self.current_ev_deadline_business_penalty), 'user_flow_lpm': float(getattr(self, 'current_step_user_flow', 0.0)), 'hot_water_draw_l': float(getattr(self, 'current_step_hot_water_draw_l', 0.0)), 'hot_water_delivery_temp_c': float(getattr(self, 'current_step_hot_water_delivery_temp_c', reward_state['ewh_temp'])), 'raw_violation_count': float(self.current_step_raw_violations), 'chattering_count': float(self.current_step_chattering_events), 'ess_chattering_count': float(self.current_step_ess_chattering_events), 'ev_chattering_count': float(self.current_step_ev_chattering_events), 'grid_import_kw': float(self.current_step_grid_import_kw), 'grid_export_kw': float(self.current_step_grid_export_kw), 'grid_import_kwh': float(self.current_step_grid_import_kwh), 'grid_export_kwh': float(self.current_step_grid_export_kwh), 'net_load_kw': float(self.current_step_net_load_kw), 'grid_import_cost': float(self.current_step_grid_import_cost), 'grid_import_cost_eur': float(self.current_step_grid_import_cost), 'grid_export_price': float(self.current_step_grid_export_price), 'grid_export_revenue': float(self.current_step_grid_export_revenue), 'grid_export_revenue_eur': float(self.current_step_grid_export_revenue), 'net_energy_cost_eur': float(self.current_step_net_energy_cost), 'ev_departure_shortfall': float(self.current_step_ev_departure_shortfall), 'ev_departure_event': float(self.current_step_ev_departure_event), 'ev_departure_success': float(self.current_step_ev_departure_success), 'wash_started': float(self.current_step_wash_started), 'wash_completed': float(self.current_step_wash_completed), 'wash_on_window': float(self.current_step_wash_on_window), 'wash_start_price': float(self.current_step_wash_start_price), 'wash_start_price_rank': float(self.current_step_wash_price_rank), 'action_prior_abs_mean': float(self.current_action_prior_abs_mean), 'action_prior_max_abs': float(self.current_action_prior_max_abs), 'hvac_safety_interventions': float(self.current_step_hvac_safety_interventions), 'ess_charge_input_kwh': float(self.current_step_ess_charge_input_kwh), 'ess_discharge_output_kwh': float(self.current_step_ess_discharge_output_kwh), 'ess_charge_from_pv_kwh': float(self.current_step_ess_charge_from_pv_kwh), 'ess_charge_non_pv_kwh': float(self.current_step_ess_charge_non_pv_kwh), 'ess_discharge_to_load_kwh': float(self.current_step_ess_discharge_to_load_kwh), 'ess_discharge_to_grid_kwh': float(self.current_step_ess_discharge_to_grid_kwh), 'pv_generation_kwh': float(self.current_step_pv_generation_kwh), 'pv_self_consumed_kwh': float(self.current_step_pv_self_consumed_kwh), 'pv_export_kwh': float(self.current_step_pv_export_kwh), 'base_load_kwh': float(self.current_step_base_load_kwh), 'hvac_energy_kwh': float(self.current_step_hvac_energy_kwh), 'ewh_energy_kwh': float(self.current_step_ewh_energy_kwh), 'wash_energy_kwh': float(self.current_step_wash_energy_kwh), 'ev_charge_input_kwh': float(self.current_step_ev_charge_input_kwh), 'ev_discharge_output_kwh': float(self.current_step_ev_discharge_output_kwh), 'ev_discharge_to_load_kwh': float(self.current_step_ev_discharge_to_load_kwh), 'ev_discharge_to_grid_kwh': float(self.current_step_ev_discharge_to_grid_kwh), 'energy_balance_residual_kwh': float(self.current_step_energy_balance_residual_kwh)}
        reward_breakdown = {'total': reward, 'energy_cost': energy_cost_reward, 'violation_penalty': violation_reward, 'aging_penalty': aging_reward, 'battery_health_penalty': float(battery_health_penalty), 'smoothness_penalty': smoothness_reward, 'invalid_action_penalty': invalid_action_reward, 'user_penalty': user_reward, 'temp_reward': float(comfort_reward + hot_water_service_reward), 'ac_comfort_reward': float(comfort_reward), 'hot_water_service_reward': float(hot_water_service_reward), 'comfort_bonus': float(self.w_comfort * comfort_bonus_raw), 'comfort_penalty': float(-self.w_comfort * comfort_penalty_raw), 'carbon_penalty': carbon_reward, 'pv_self_use_reward': 0.0, 'peak_shave_reward': float(peak_shave_reward_term), 'ev_deadline_reward': float(ev_deadline_reward), 'ev_shield_reward': float(ev_shield_reward), 'terminal_energy_reward': float(terminal_energy_reward), 'boundary_recovery_reward': float(boundary_recovery_reward), 'core_reward': core_reward, 'raw_violation_count': float(self.current_step_raw_violations), 'chattering_count': float(self.current_step_chattering_events)}
        for key in self.reward_components:
            self.reward_components[key].append(reward_breakdown[key])
        return reward

    def _ess_charge_pv_fraction(self) -> float:
        t = self._ess_stored_from_pv_kwh + self._ess_stored_from_grid_kwh
        if t <= 1e-09:
            return 0.5
        return float(np.clip(self._ess_stored_from_pv_kwh / t, 0.0, 1.0))

    def _sync_ess_stored_sources_to_soc(self) -> None:
        soc = float(self.state.get('ess_state', 0.0))
        t = self._ess_stored_from_pv_kwh + self._ess_stored_from_grid_kwh
        if soc <= 1e-09:
            self._ess_stored_from_pv_kwh = 0.0
            self._ess_stored_from_grid_kwh = 0.0
            return
        if t <= 1e-09:
            self._ess_stored_from_pv_kwh = soc * 0.5
            self._ess_stored_from_grid_kwh = soc * 0.5
            return
        if abs(t - soc) > 0.001:
            k = soc / t
            self._ess_stored_from_pv_kwh *= k
            self._ess_stored_from_grid_kwh *= k

    def _update_ess_energy_source_bookkeeping(self, state: Dict, action: Dict, total_consumption: float, _net_demand: float) -> None:
        prev_ess = float(state['ess_state'])
        new_ess = float(self.state['ess_state'])
        delta_soc = new_ess - prev_ess
        if delta_soc > 1e-09:
            bc = max(0.0, float(action.get('battery_power', 0.0)))
            if bc > 1e-12:
                load_fixed = float(total_consumption) - bc
                pv_kw = float(state.get('pv_generation', 0.0))
                pv_surplus_kw = max(0.0, pv_kw - max(0.0, load_fixed))
                from_pv_power = min(bc, pv_surplus_kw)
                frac_pv = float(from_pv_power / bc)
            else:
                frac_pv = 0.5
            self._ess_stored_from_pv_kwh += float(delta_soc) * frac_pv
            self._ess_stored_from_grid_kwh += float(delta_soc) * (1.0 - frac_pv)
        elif delta_soc < -1e-09:
            chem_out = -delta_soc
            tot = self._ess_stored_from_pv_kwh + self._ess_stored_from_grid_kwh
            if tot > 1e-12:
                fp = self._ess_stored_from_pv_kwh / tot
            else:
                fp = 0.5
            self._ess_stored_from_pv_kwh = max(0.0, self._ess_stored_from_pv_kwh - chem_out * fp)
            self._ess_stored_from_grid_kwh = max(0.0, self._ess_stored_from_grid_kwh - chem_out * (1.0 - fp))
        self._sync_ess_stored_sources_to_soc()

    def calculate_user_dissatisfaction0(self):
        return self._quadratic_deadband_discomfort(value=float(self.indoor_temp2), target=float(self.user_temp_preference2), deadband=2.0, scale=1.0, cap=50.0)

    def calculate_user_dissatisfaction1(self):
        return self._quadratic_deadband_discomfort(value=float(self.indoor_temp), target=float(self.user_temp_preference), deadband=2.0, scale=1.0, cap=50.0)

    def _wash_electricity_price_penalty(self, absolute_price: float) -> float:
        e = self._wash_abs_earliest_start_step()
        l = self._wash_abs_latest_start_step()
        date_str = self.current_time
        prices = [float(self.data_interface.get_electricity_price(date_str, s)) for s in range(e, l + 1)]
        p_lo, p_hi = (min(prices), max(prices))
        if p_hi - p_lo < 1e-10:
            return 0.0
        p = float(absolute_price)
        norm = (p - p_lo) / (p_hi - p_lo)
        norm = float(np.clip(norm, 0.0, 1.0))
        return (norm - 0.5) * 10.0

    def calculate_user_dissatisfaction2(self, state, action):
        time_penalty = float(self.wash_time_penalty_coef) * float(self.time_deviation)
        return float(time_penalty)

    @staticmethod
    def _quadratic_deadband_discomfort(value, target, deadband=2.0, scale=1.0, cap=500.0):
        exceed = max(0.0, abs(float(value) - float(target)) - float(deadband))
        return float(min(float(scale) * exceed ** 2, float(cap)))

    def is_terminal_state(self):
        if self.current_episode_length_steps is not None and self.current_episode_length_steps > 0:
            return self._current_block_step >= self.current_episode_length_steps - 1
        return self.current_time > self._last_cons_date_str(self.data_interface)

    def seed(self, seed=None):
        self._seed_value = int(seed) if seed is not None else None
        self.np_random = np.random.RandomState(seed)
        if seed is not None:
            if hasattr(self.data_interface, 'seed'):
                self.data_interface.seed(int(seed))

    def is_ev_at_home(self):
        return self.data_interface.is_ev_at_home(self.current_time, self.current_time_index)

    def _get_remaining_home_steps(self) -> int:
        if not self.is_ev_at_home():
            return 0
        departure_step = self._find_departure_step_within_episode()
        if departure_step is None:
            return 0
        return max(0, int(departure_step) - int(self.current_time_index) + 1)

    def update_ev_state_after_trip(self, current_ev_soc):
        return float(np.clip(float(current_ev_soc) - float(self._episode_trip_kwh), 0.0, self.ev_capacity))

    def _find_contiguous_segments(self, bool_list):
        segments = []
        start_idx = None
        for i, value in enumerate(bool_list):
            if value and start_idx is None:
                start_idx = i
            elif not value and start_idx is not None:
                segments.append((start_idx, i - 1))
                start_idx = None
        if start_idx is not None:
            segments.append((start_idx, len(bool_list) - 1))
        return segments

    def reward_shape(self, progress):
        raise RuntimeError('当前论文协议禁止训练脚本动态修改环境 reward；请使用统一 Reward。')

    def save_cost_data(self):
        results_dir = os.path.join(self._project_root, 'model', 'cost_results')
        os.makedirs(results_dir, exist_ok=True)
        csv_filename = os.path.join(results_dir, 'cost_data.csv')
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Energy Cost'])
            for i, (ts, cost) in enumerate(zip(self.records['timestamps'], self.records['energy_cost'])):
                writer.writerow([ts.strftime('%Y-%m-%d %H:%M:%S'), cost])
        print(f'Cost data saved to: {csv_filename}')

    def save_episode_costs(self):
        results_dir = os.path.join(self._project_root, 'model', 'episode_cost_results')
        os.makedirs(results_dir, exist_ok=True)
        filename = os.path.join(results_dir, 'episode_costs.csv')
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Total Cost'])
            for episode, cost in enumerate(self.episode_costs):
                writer.writerow([episode + 1, cost])
        print(f'Episode cost data saved to: {filename}')

    def save_simulation_data(self, filename=None):
        if filename is None:
            filename = 'simulation_data.csv'
        sim_data_dir = os.path.join(self._project_root, 'simulation_data')
        os.makedirs(sim_data_dir, exist_ok=True)
        filepath = os.path.join('simulation_data', filename)
        data_dict = {'timestamp': self.records['timestamps'], 'ev_soc': self.ev_battery_record, 'ess_soc': self.ess_state_record, 'home_load': self.home_load_record, 'pv_generation': self.pv_generation_record, 'electricity_price': self.electricity_price_record, 'ev_at_home': self.ev_at_home_record, 'wash_machine_state': self.wash_machine_record, 'air_conditioner_power': self.air_conditioner_power_record, 'ess_actions': self.ess_actions, 'wash_machine_actions': self.wash_machine_actions, 'air_conditioner_actions': self.air_conditioner_actions, 'ess_charge_pv': self.ess_charge_pv, 'ess_charge_grid': self.ess_charge_grid, 'ess_discharge_ev': self.ess_discharge_ev, 'ess_discharge_house': self.ess_discharge_house, 'ess_discharge_grid': self.ess_discharge_grid, 'indoor_temp': self.records['indoor_temp'], 'indoor_temp2': self.records['indoor_temp2'], 'outdoor_temp': self.records['outdoor_temp'], 'total_load': self.records['total_load'], 'energy_cost': self.records['energy_cost'], 'user_dissatisfaction': self.records['user_dissatisfaction'], 'ewh_temp': self.ewh_temp_record, 'ewh_power': self.ewh_power_record, 'user_flow': self.user_flow_record, 'user_flow_lpm': self.user_flow_record, 'hot_water_draw_l': self.hot_water_draw_volume_record, 'air_conditioner_power2': self.air_conditioner_power_record2, 'daily_costs': self.records['daily_costs']}
        list_values = [v for v in data_dict.values() if isinstance(v, list)]
        list_lengths = [len(v) for v in list_values]
        max_length = max(list_lengths) if list_lengths else 0
        for key, value in data_dict.items():
            if isinstance(value, list):
                if len(value) < max_length:
                    missing = max_length - len(value)
                    if len(value) == 0:
                        data_dict[key] = [None] * missing
                    else:
                        data_dict[key] = value + [value[-1]] * missing
                elif len(value) > max_length:
                    data_dict[key] = value[:max_length]
        df = pd.DataFrame(data_dict)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f'Simulation data saved to: {filepath}')
        reward_filename = filename.replace('.csv', '_rewards.csv')
        reward_filepath = os.path.join(sim_data_dir, reward_filename)
        reward_data = {}
        for key, values in self.reward_components.items():
            if isinstance(values, list):
                if len(values) < max_length:
                    missing = max_length - len(values)
                    if len(values) == 0:
                        reward_data[key] = [None] * missing
                    else:
                        reward_data[key] = values + [values[-1]] * missing
                else:
                    reward_data[key] = values[:max_length]
        reward_df = pd.DataFrame(reward_data)
        reward_df.to_csv(reward_filepath, index=False, encoding='utf-8-sig')
        print(f'Reward component data saved to: {reward_filepath}')
        return filepath

def make_storenet_train_env(seed: int=42, train_days: int=200, val_days: int=30, test_days: int=100, **env_kwargs) -> HomeEnergyManagementEnv:
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
    data_append = os.path.join(root, 'data')
    if data_append not in sys.path:
        sys.path.append(data_append)
    from storenet_splits import StoreNetSplitConfig, make_storenet_splits_2020, flatten_episode_schedule
    base_dir = os.path.join(root, 'data', 'storenet_ireland_2020')
    train_houses = [f'H{i}' for i in range(1, 16)]
    val_houses = [f'H{i}' for i in range(1, 16)]
    test_houses = [f'H{i}' for i in range(16, 21)]
    split_cfg = StoreNetSplitConfig(seed=seed, train_days=train_days, val_days=val_days, test_days=test_days)
    splits = make_storenet_splits_2020(base_dir=base_dir, houses_train=train_houses, houses_val=val_houses, houses_test=test_houses, config=split_cfg)
    schedule_train = flatten_episode_schedule(splits, house_ids=train_houses, split_name='train', seed=seed)
    return HomeEnergyManagementEnv(episode_length_steps=int(CONFIG['protocol']['episode_length_steps']), episode_schedule=schedule_train, storenet_base_dir=base_dir, price_profile='ireland_tou_2020', steps_per_day=48, **env_kwargs)
