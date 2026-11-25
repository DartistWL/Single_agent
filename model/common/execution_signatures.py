"""execution signatures."""
from __future__ import annotations
import math
from collections.abc import Mapping
SEMANTICS_VERSION = 'hems-execution-v1'
DEVICE_ORDER = ('ev_power', 'wash_machine_schedule', 'Air_conditioner_set_temp', 'Air_conditioner_set_temp2', 'ewh_set_temp', 'battery_power')
SIGNATURE_FIELDS = ('selected', 'power_kw_over_6_6', 'power_is_exact', 'command_normalized', 'task_running', 'task_completed', 'appointment_pending', 'appointment_start_fraction')
SIGNATURE_SIZE = len(SIGNATURE_FIELDS)
POWER_SCALE_KW = 6.6

def _finite(value, name):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f'{name} must be finite')
    return number

def ewh_power(setpoint, current_temp, time_index):
    difference = _finite(setpoint, 'setpoint') - _finite(current_temp, 'ewh_temp')
    hour = _finite(time_index, 'time_index') // 2
    if 6 <= hour <= 9 or 18 <= hour <= 22:
        if difference < 1:
            return 0.0
        if difference < 4:
            return min(max(0.4 + 0.1 * max(0.0, difference - 1), 0.0), 1.2)
        if difference < 6:
            return 1.0
        return 1.2
    if difference < 5:
        return 0.0
    return 0.2 if difference < 8 else 0.5

def hvac_power(setpoint, indoor_temp):
    difference = _finite(setpoint, 'setpoint') - _finite(indoor_temp, 'indoor_temp')
    if difference > 0:
        for upper, power in ((0.5, 0.0), (1, 0.5), (2, 1.0), (3, 1.5), (4, 2.0)):
            if difference < upper:
                return power
        return 3.0
    for lower, power in ((-0.5, 0.0), (-1, 0.5), (-2, 1.0), (-3, 1.5), (-4, 2.0)):
        if difference >= lower:
            return power
    return 3.0

def _wash_status(state):
    remaining = state.get('wash_remaining_steps')
    if remaining is not None and _finite(remaining, 'wash_remaining_steps') > 0:
        return 'running'
    if remaining is not None and float(remaining) == 0 and (state.get('wash_completed_today') == 1) and (state.get('wash_period_reset_pending', 0) == 0):
        return 'completed'
    return 'unmerged'

def _wash_descriptor(state, command, status):
    power = _finite(state.get('wash_power_kw', 1.5), 'wash_power_kw')
    if status == 'running':
        return (power, 1.0, 1.0, 0.0, 0.0, 0.0)
    if status == 'completed':
        return (0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
    required = ('time_index', 'steps_per_day', 'wash_earliest_start_step', 'wash_latest_start_step', 'wash_scheduled_start_step', 'wash_force_start', 'wash_remaining_steps', 'wash_completed_today')
    if not all((key in state for key in required)):
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    slot, spd, earliest, latest, scheduled, force, _, _ = (_finite(state[key], key) for key in required)
    if spd <= 0:
        raise ValueError('steps_per_day must be positive')
    if force:
        target = earliest if slot < earliest else slot
    elif scheduled >= 0 and slot >= scheduled:
        target = slot
    elif command == 0:
        target = scheduled
    else:
        target = min(max(earliest + int(command) - 1, earliest), latest)
        target = min(max(target, slot), latest)
    active = float(target == slot)
    pending = float(target > slot)
    return (power * active, 1.0, active, 0.0, pending, target / spd if pending else 0.0)

def _signature(device, command, state, command_scale, status):
    token = command / command_scale
    power = exact = running = completed = pending = appointment = 0.0
    if device == 'ewh_set_temp' and 'ewh_temp' in state and ('time_index' in state):
        power = ewh_power(command, state['ewh_temp'], state['time_index'])
        exact, token = (1.0, 0.0)
    elif device == 'wash_machine_schedule':
        power, exact, running, completed, pending, appointment = _wash_descriptor(state, command, status)
        if status != 'unmerged':
            token = 0.0
    elif device in ('Air_conditioner_set_temp', 'Air_conditioner_set_temp2'):
        temperature_key = 'indoor_temp' if device == 'Air_conditioner_set_temp' else 'indoor_temp2'
        if temperature_key in state:
            power, exact = (hvac_power(command, state[temperature_key]), 1.0)
    elif device in ('battery_power', 'ev_power'):
        power = command
    return [1.0, power / POWER_SCALE_KW, exact, token, running, completed, pending, appointment]

def build_classes(public_state: Mapping, action_space: Mapping, masks: Mapping | None=None) -> dict:
    masks = {} if masks is None else masks
    result = {}
    for device, raw_commands in action_space.items():
        commands = [_finite(value, f'{device} command') for value in raw_commands]
        if not commands:
            raise ValueError(f'{device}: empty action space')
        valid = [bool(value) for value in masks.get(device, [True] * len(commands))]
        if len(valid) != len(commands):
            raise ValueError(f'{device}: mask/action length mismatch')
        if not any(valid):
            raise ValueError(f'{device}: no legal action')
        scale = max(1.0, max((abs(value) for value in commands)))
        status = _wash_status(public_state) if device == 'wash_machine_schedule' else 'unmerged'
        class_ids, members, by_key = ([-1] * len(commands), [], {})
        for index, (command, legal) in enumerate(zip(commands, valid)):
            if not legal:
                continue
            if device == 'ewh_set_temp' and 'ewh_temp' in public_state and ('time_index' in public_state):
                key = ('ewh_power', ewh_power(command, public_state['ewh_temp'], public_state['time_index']))
            elif device == 'wash_machine_schedule' and status != 'unmerged':
                key = ('ignored', status)
            else:
                key = ('singleton', index)
            if key not in by_key:
                by_key[key] = len(members)
                members.append([])
            class_id = by_key[key]
            class_ids[index] = class_id
            members[class_id].append(index)
        representatives, signatures = ([], [])
        for indices in members:
            if device == 'ewh_set_temp':
                representative = min(indices, key=lambda index: (commands[index], index))
            elif device == 'wash_machine_schedule':
                representative = next((index for index in indices if commands[index] == 0), indices[0])
            else:
                representative = indices[0]
            representatives.append(representative)
            signatures.append(_signature(device, commands[representative], public_state, scale, status))
        result[device] = dict(version=SEMANTICS_VERSION, class_ids=class_ids, members=members, representatives=representatives, signatures=signatures)
    return result
