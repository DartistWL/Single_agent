"""service rules."""
from typing import Any, Dict
import numpy as np
from other.environment.physics import HomeEnergyManagementEnv
from other.environment.config import CONFIG

def _remaining_block_action_steps(env: HomeEnergyManagementEnv) -> int:
    required = ('current_episode_length_steps', '_current_block_step')
    missing = [name for name in required if not hasattr(env, name)]
    if missing:
        raise ValueError('Rule controller requires block metadata: ' + ', '.join(missing))
    total = int(env.current_episode_length_steps)
    step = int(env._current_block_step)
    if total <= 0 or not 0 <= step < total:
        raise ValueError(f'Invalid rule-controller block position: step={step}, total={total}')
    return total - step

def _service_ewh_setpoint(env: HomeEnergyManagementEnv, state: Dict[str, Any], preferred: float) -> float:
    context = env._ewh_service_guard_context(state)
    options = [float(value) for value in env.action_space['ewh_set_temp']]
    if not options:
        raise ValueError('EWH action space is empty')
    if float(context['active']) > 0.5:
        safe = [value for value in options if env._ewh_setpoint_keeps_service_reachable(state, value)]
        if safe:
            return min(safe, key=lambda value: (abs(value - preferred), value))
        return max(options)
    return min(options, key=lambda value: (abs(value - preferred), value))
