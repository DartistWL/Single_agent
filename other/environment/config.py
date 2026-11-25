"""config."""
import json
from pathlib import Path
CONFIG = json.loads(Path(__file__).with_suffix('.json').read_text(encoding='utf8'))

def scheduled_training_stage(algorithm_config: dict, episode: int, total_episodes: int) -> dict:
    alg = algorithm_config or {}
    schedule = alg.get('training_schedule') or []
    if not schedule:
        return {}
    denom = max(1, int(total_episodes or 1))
    progress = min(1.0, max(0.0, float(episode + 1) / float(denom)))
    selected = schedule[-1]
    for stage in schedule:
        until = float(stage.get('until_fraction', stage.get('end_fraction', 1.0)))
        if progress <= until + 1e-12:
            selected = stage
            break
    return dict(selected)

def effective_algorithm_config_for_episode(algorithm_config: dict, episode: int, total_episodes: int) -> dict:
    effective = dict(algorithm_config or {})
    stage = scheduled_training_stage(algorithm_config, episode, total_episodes)
    for key, value in stage.items():
        if key not in {'name', 'until_fraction', 'end_fraction'}:
            effective[key] = value
    if stage:
        effective['active_stage_name'] = str(stage.get('name', 'stage'))
        effective['active_stage_until_fraction'] = float(stage.get('until_fraction', stage.get('end_fraction', 1.0)))
    return effective

def apply_episode_stage_to_agent(agent, effective_config: dict) -> None:
    cfg = effective_config or {}
    attr_map = {'use_dynamic_mask': 'use_dynamic_mask', 'use_action_penalty': 'use_action_penalty', 'mask_query_apply_soc': 'mask_query_apply_soc', 'mask_mode': 'mask_mode', 'constraint_mode': 'constraint_mode', 'constraint_weight': 'constraint_weight', 'ev_deadline_constraint_weight': 'ev_deadline_constraint_weight', 'peak_constraint_weight': 'peak_constraint_weight', 'switch_constraint_weight': 'switch_constraint_weight', 'risk_intervention_enabled': 'risk_intervention_enabled', 'intervention_peak_pressure_threshold': 'intervention_peak_pressure_threshold', 'intervention_lambda_threshold': 'intervention_lambda_threshold', 'intervention_switch_lambda_threshold': 'intervention_switch_lambda_threshold', 'intervention_peak_pressure_exit_threshold': 'intervention_peak_pressure_exit_threshold', 'intervention_lambda_exit_threshold': 'intervention_lambda_exit_threshold', 'intervention_switch_lambda_exit_threshold': 'intervention_switch_lambda_exit_threshold', 'intervention_min_dwell_steps': 'intervention_min_dwell_steps', 'mask_strength': 'mask_strength'}
    for cfg_key, attr_name in attr_map.items():
        if cfg_key not in cfg or not hasattr(agent, attr_name):
            continue
        value = cfg[cfg_key]
        if cfg_key in {'use_dynamic_mask', 'use_action_penalty', 'mask_query_apply_soc', 'risk_intervention_enabled'}:
            value = bool(value)
        elif cfg_key == 'mask_mode':
            value = str(value or 'none').lower()
        elif cfg_key == 'constraint_mode':
            value = str(value or 'none')
        elif cfg_key in {'constraint_weight', 'ev_deadline_constraint_weight', 'peak_constraint_weight', 'switch_constraint_weight', 'intervention_peak_pressure_threshold', 'intervention_lambda_threshold', 'intervention_switch_lambda_threshold', 'intervention_peak_pressure_exit_threshold', 'intervention_lambda_exit_threshold', 'intervention_switch_lambda_exit_threshold'}:
            value = float(value or 0.0)
        elif cfg_key == 'intervention_min_dwell_steps':
            value = max(0, int(value or 0))
        elif cfg_key == 'mask_strength':
            value = max(0.0, min(1.0, float(value or 0.0)))
        setattr(agent, attr_name, value)
