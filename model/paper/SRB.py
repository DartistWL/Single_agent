"""SRB algorithm."""
from typing import Any, Dict
import numpy as np
from other.environment.physics import HomeEnergyManagementEnv
from other.environment.config import CONFIG
from model.common.service_rules import _remaining_block_action_steps, _service_ewh_setpoint

class PhysicalBaselineController:

    def __init__(self, env: HomeEnergyManagementEnv, ac_set_temp=22, ac2_set_temp=22, ewh_peak_target=55, ewh_offpeak_target=45, wash_policy: str='earliest'):
        self.env = env
        self.steps_per_day = env.steps_per_day
        self.dt_h = 24.0 / self.steps_per_day
        self.ev_min_charge = env.ev_min_charge
        self.ac_set_temp = ac_set_temp
        self.ac2_set_temp = ac2_set_temp
        self.ewh_peak_target = ewh_peak_target
        self.ewh_offpeak_target = ewh_offpeak_target
        self.wash_policy = str(wash_policy or 'earliest').lower()
        self.wash_used_today = False
        self.current_date = None

    def _is_peak_hour(self, time_index: int) -> bool:
        hour = time_index * self.dt_h % 24
        return 6 <= hour < 9 or 18 <= hour < 22

    def _nearest_action(self, options, target):
        arr = np.asarray(list(options), dtype=float)
        return float(arr[int(np.argmin(np.abs(arr - float(target))))])

    def _get_ev_action(self, state: Dict[str, Any]) -> float:
        if float(state['ev_at_home']) < 0.5:
            return 0.0
        metrics = self.env._ev_deadline_metrics(state)
        ev_kwh = float(state['ev_battery_state'])
        reserve_gap = max(0.0, self.env._ev_operational_reserve_kwh() - ev_kwh)
        max_charge_kw = max((float(v) for v in self.env.action_space['ev_power']))
        departure_gap = float(metrics['hard_gap_kwh']) if metrics['steps_remaining'] > 0.0 else 0.0
        gap = max(reserve_gap, departure_gap)
        if gap > 1e-08:
            return min(max_charge_kw, gap / (self.env.charge_efficiency * self.dt_h))
        return 0.0

    def _nearest_feasible_ev_power(self, state: Dict[str, Any], target: float) -> float:
        if float(state['ev_at_home']) < 0.5:
            return 0.0
        energy = float(state['ev_battery_state'])
        max_charge = max(0.0, (self.env.ev_capacity - energy) / max(1e-08, self.env.charge_efficiency * self.dt_h))
        physical = [float(v) for v in self.env.action_space['ev_power'] if 0.0 <= float(v) <= max_charge + 1e-08]
        if not physical:
            raise ValueError('No physically supported nonnegative EV baseline action')
        feasible = [value for value in physical if self.env._ev_power_keeps_deadline_reachable(state, value)]
        if not feasible:
            return max(physical)
        if float(target) > 1e-08:
            sufficient = [value for value in feasible if value + 1e-08 >= float(target)]
            if sufficient:
                return min(sufficient)
        return min(feasible, key=lambda value: (abs(value - float(target)), value))

    def _get_wash_action(self) -> int:
        current_date_str = self.env.current_time
        if current_date_str != self.current_date:
            self.current_date = current_date_str
            self.wash_used_today = False
        if self.wash_used_today:
            return 0
        earliest = self.env._wash_abs_earliest_start_step()
        latest = self.env._wash_abs_latest_start_step()
        current = int(self.env.current_time_index)
        if current > latest:
            self.wash_used_today = True
            return 0
        target_step = earliest
        if self.wash_policy == 'latest':
            target_step = latest
        elif self.wash_policy == 'cheapest':
            prices = [(float(self.env.data_interface.get_electricity_price(self.current_date, s)), s) for s in range(max(current, earliest), latest + 1)]
            target_step = min(prices)[1] if prices else earliest
        if current >= target_step:
            self.wash_used_today = True
            return int(target_step - earliest + 1)
        return 0

    def take_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ac_options = self.env.action_space['Air_conditioner_set_temp']
        ac2_options = self.env.action_space['Air_conditioner_set_temp2']
        ewh_target = _service_ewh_setpoint(self.env, state, self.ewh_offpeak_target)
        ev_action = self._nearest_feasible_ev_power(state, self._get_ev_action(state))
        return {'ev_power': ev_action, 'battery_power': 0.0, 'wash_machine_schedule': self._get_wash_action(), 'Air_conditioner_set_temp': int(self._nearest_action(ac_options, self.ac_set_temp)), 'Air_conditioner_set_temp2': int(self._nearest_action(ac2_options, self.ac2_set_temp)), 'ewh_set_temp': int(ewh_target)}
