"""TOU RBC algorithm."""
from typing import Any, Dict
import numpy as np
from other.environment.physics import HomeEnergyManagementEnv
from other.environment.config import CONFIG
from model.common.service_rules import _remaining_block_action_steps, _service_ewh_setpoint

class RuleBasedController:

    def __init__(self, env: HomeEnergyManagementEnv, cheap_price_threshold=0.13, expensive_price_threshold=0.24, ac_set_temp=22, ac2_set_temp=22, ewh_peak_target=55, ewh_offpeak_target=45):
        self.env = env
        self.steps_per_day = env.steps_per_day
        self.dt_h = 24.0 / self.steps_per_day
        self.ess_capacity = env.ess_capacity
        self.ev_capacity = env.ev_capacity
        self.ev_min_charge = env.ev_min_charge
        self.wash_pref_start, self.wash_pref_end = env.wash_machine_preferred_time
        self.wash_duration = env.wash_machine_duration
        self.cheap_price_threshold = cheap_price_threshold
        self.expensive_price_threshold = expensive_price_threshold
        self.ac_set_temp = ac_set_temp
        self.ac2_set_temp = ac2_set_temp
        self.ewh_peak_target = ewh_peak_target
        self.ewh_offpeak_target = ewh_offpeak_target
        self.wash_used_today = False
        self.current_date = None

    def _is_peak_hour(self, time_index: int) -> bool:
        hour = time_index * self.dt_h % 24
        return 6 <= hour < 9 or 18 <= hour < 22

    def _get_ewh_set_temp(self, state: Dict[str, Any]) -> int:
        upcoming_draw = float(state['hot_water_draw_forecast_norm'])
        current_temp = float(state['ewh_temp'])
        price = float(state['electricity_price'])
        if upcoming_draw > 0.03:
            if price <= self.cheap_price_threshold:
                target = self.ewh_peak_target
            elif current_temp < 42.0:
                target = 50
            else:
                target = self.ewh_offpeak_target
        elif current_temp > 43.0 and price > self.cheap_price_threshold:
            target = 0
        else:
            target = 45 if current_temp < 42.0 else 40
        return int(_service_ewh_setpoint(self.env, state, target))

    def _get_ev_action(self, state: Dict[str, Any]) -> float:
        if state['ev_at_home'] != 1.0:
            return 0.0
        ev_kwh = float(state['ev_battery_state'])
        price = float(state['electricity_price'])
        max_charge_kw = max((float(v) for v in self.env.action_space['ev_power']))
        metrics = self.env._ev_deadline_metrics(state)
        reserve_gap = max(0.0, self.env._ev_operational_reserve_kwh() - ev_kwh)
        if metrics['steps_remaining'] > 0.0:
            future_steps = max(0, int(metrics['steps_remaining']) - 1)
            future_charge_kwh = future_steps * max_charge_kw * self.env.charge_efficiency * self.dt_h
            minimum_now_kwh = max(0.0, float(metrics['hard_gap_kwh']) - future_charge_kwh)
            if minimum_now_kwh > 1e-08:
                return minimum_now_kwh / max(1e-08, self.env.charge_efficiency * self.dt_h)
            if price <= self.cheap_price_threshold and metrics['gap_kwh'] > 1e-08:
                return max_charge_kw
        elif reserve_gap > 1e-08:
            return reserve_gap / max(1e-08, self.env.charge_efficiency * self.dt_h)
        return 0.0

    def _get_ess_action(self, state: Dict[str, Any]) -> float:
        price = state['electricity_price']
        pv = state['pv_generation']
        home_load = state['home_load']
        ess_soc = state['ess_state']
        max_charge = max(self.env.action_space['battery_power'])
        max_discharge = min(self.env.action_space['battery_power'])
        target_gap_kwh = float(self.env._boundary_gap_metrics(state)['ess_gap_kwh'])
        steps_left = _remaining_block_action_steps(self.env) - 1
        future_charge_kwh = steps_left * max_charge * self.env.charge_efficiency * self.dt_h
        if target_gap_kwh > future_charge_kwh + 1e-08:
            return max_charge
        if ess_soc <= 0.1 * self.ess_capacity:
            if price <= self.expensive_price_threshold:
                return max_charge
            else:
                return 0.0
        if ess_soc >= 0.9 * self.ess_capacity:
            if price >= self.expensive_price_threshold and home_load > 0:
                return max_discharge
            else:
                return 0.0
        if price <= self.cheap_price_threshold:
            pv_excess = max(0.0, pv - home_load)
            if pv_excess > 0:
                return min(max_charge, pv_excess)
            else:
                return max_charge
        elif price >= self.expensive_price_threshold:
            if home_load > 0:
                return max_discharge
            else:
                return 0.0
        else:
            return 0.0

    def _nearest_feasible_power(self, branch: str, target: float, state: Dict[str, Any]) -> float:
        options = [float(v) for v in self.env.action_space[branch]]
        dt_h = self.dt_h
        if branch == 'battery_power':
            energy = float(state['ess_state'])
            capacity = float(self.env.ess_capacity)
            reserve = 0.0
        else:
            if float(state['ev_at_home']) < 0.5:
                return 0.0
            energy = float(state['ev_battery_state'])
            capacity = float(self.env.ev_capacity)
            reserve = float(self.env._ev_operational_reserve_kwh())
        max_discharge = max(0.0, energy - reserve) * self.env.discharge_efficiency / dt_h
        max_charge = max(0.0, capacity - energy) / max(1e-08, self.env.charge_efficiency * dt_h)
        physical = [v for v in options if -max_discharge - 1e-08 <= v <= max_charge + 1e-08]
        if not physical:
            raise ValueError(f'No physically supported rule action for {branch}')
        remaining = _remaining_block_action_steps(self.env)
        target_key = 'ess_target_kwh' if branch == 'battery_power' else 'ev_target_kwh'
        recovery = self.env._boundary_gap_metrics(state)
        reference = float(recovery[target_key])
        max_branch_charge = max(0.0, max(options))

        def keeps_recovery(value: float) -> bool:
            delta = value * self.env.charge_efficiency * dt_h if value >= 0.0 else value / self.env.discharge_efficiency * dt_h
            future = (remaining - 1) * max_branch_charge * self.env.charge_efficiency * dt_h
            return energy + delta + future >= reference - 1e-08
        if branch == 'ev_power':
            feasible = [v for v in physical if self.env._ev_power_keeps_deadline_reachable(state, v)]
            if float(self.env._ev_deadline_metrics(state)['steps_remaining']) <= 0.0:
                feasible = [v for v in feasible if keeps_recovery(v)]
        else:
            feasible = [v for v in physical if keeps_recovery(v)]
        if not feasible:
            return max(physical)
        if float(target) > 1e-08:
            sufficient = [value for value in feasible if value >= float(target) - 1e-08]
            if sufficient:
                return min(sufficient)
        return min(feasible, key=lambda value: (abs(value - float(target)), abs(value)))

    def _get_wash_action(self, state: Dict[str, Any]) -> int:
        current_date_str = self.env.current_time
        if current_date_str != self.current_date:
            self.current_date = current_date_str
            self.wash_used_today = False
        if self.wash_used_today:
            return 0
        earliest_step = self.env._wash_abs_earliest_start_step()
        latest_step = self.env._wash_abs_latest_start_step()
        current_step = self.env.current_time_index
        if current_step > latest_step:
            self.wash_used_today = True
            return 0
        best_step = earliest_step
        best_price = float('inf')
        for s in range(earliest_step, latest_step + 1):
            if s < current_step:
                continue
            price = self.env.data_interface.get_electricity_price(self.current_date, s)
            if price < best_price:
                best_price = price
                best_step = s
        if best_step == current_step:
            action_val = best_step - earliest_step + 1
            self.wash_used_today = True
            return action_val
        else:
            return 0

    def take_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ev_options = self.env.action_space['ev_power']
        ess_options = self.env.action_space['battery_power']
        ac_options = self.env.action_space['Air_conditioner_set_temp']
        ac2_options = self.env.action_space['Air_conditioner_set_temp2']
        ewh_options = self.env.action_space['ewh_set_temp']
        wash_options = self.env.action_space['wash_machine_schedule']
        ev_target = self._get_ev_action(state)
        ev_action = self._nearest_feasible_power('ev_power', ev_target, state)
        ess_target = self._get_ess_action(state)
        ess_action = self._nearest_feasible_power('battery_power', ess_target, state)
        ac_idx = ac_options.index(self.ac_set_temp) if self.ac_set_temp in ac_options else 0
        ac_action = ac_options[ac_idx]
        ac2_idx = ac2_options.index(self.ac2_set_temp) if self.ac2_set_temp in ac2_options else 0
        ac2_action = ac2_options[ac2_idx]
        ewh_target = self._get_ewh_set_temp(state)
        ewh_idx = ewh_options.index(ewh_target) if ewh_target in ewh_options else 0
        ewh_action = ewh_options[ewh_idx]
        wash_action = self._get_wash_action(state)
        return {'ev_power': ev_action, 'battery_power': ess_action, 'wash_machine_schedule': wash_action, 'Air_conditioner_set_temp': ac_action, 'Air_conditioner_set_temp2': ac2_action, 'ewh_set_temp': ewh_action}
