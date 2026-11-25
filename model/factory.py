"""Create algorithm instances."""
import importlib, json
from pathlib import Path
from .catalog import CATALOG, BASELINES, canonical_name, legacy_name
SETTINGS = json.loads((Path(__file__).parent / 'common/settings.json').read_text(encoding='utf8'))

def ppo_parameters(env, device, blocks, options):
    flags = dict(options)
    mode = flags.pop('selective_policy_mode')
    n = len(env.state_space)
    cfg = SETTINGS['training']
    length = cfg['state_stack_len']
    result = dict(env=env, state_dim=n * length, hidden_dim=cfg['hidden_dim'], action_space_config=env.action_space, device=device, base_state_dim=n, state_stack_len=length, device_order=SETTINGS['action_order'], selective_policy_mode=mode, refinement_start_episode=max(1, blocks * cfg['source_phase_one_blocks'] // cfg['source_total_blocks']))
    result.update({k: cfg[k] for k in ('gamma', 'lmbda', 'eps', 'epochs', 'ent_coef', 'max_grad_norm')})
    result.update(SETTINGS['policy'])
    result.update(flags)
    return result

def make_agent(name, env, device, blocks):
    name = canonical_name(name)
    if name in BASELINES:
        raise ValueError(name + ' is a controller')
    return importlib.import_module(CATALOG[name]['module']).build_agent(env, device, blocks)

def controller(name, env, protocol):
    name = canonical_name(name)
    module = importlib.import_module(CATALOG[name]['module'])
    if name == 'MPC':
        return module.MPC(env, protocol['mpc_horizon'], protocol['mpc_budget'])
    if name == 'CI_RBC':
        return module.CarbonRule(env, protocol['carbon_reference'])
    return (module.PhysicalBaselineController if name == 'SRB' else module.RuleBasedController)(env)

def begin_block(agent, index, blocks):
    if hasattr(agent, 'ref_config'):
        from other.environment.config import effective_algorithm_config_for_episode, apply_episode_stage_to_agent
        apply_episode_stage_to_agent(agent, effective_algorithm_config_for_episode(agent.ref_config, index, blocks))
        if hasattr(agent, 'reset_intervention_state'):
            agent.reset_intervention_state()
    else:
        from .common.PPO_policy import HouseholdPPO
        if isinstance(agent, HouseholdPPO) and index >= agent.refinement_start_episode:
            agent.enter_refinement()

def decision(agent, env, state, tensor, deterministic, *, skip_diagnostics=False):
    from .common.PPO_policy import HouseholdPPO
    if hasattr(agent, 'choose'):
        return agent.choose(env, state, tensor, deterministic)
    if hasattr(agent, 'ref_config'):
        selective = hasattr(agent, 'should_apply_minimal_intervention')
        active = agent.should_apply_minimal_intervention(state) if selective else agent.mask_query_apply_soc
        mask, penalty = env.get_action_mask(state, apply_soc_mask=active, mask_mode=agent.mask_mode if active else 'none')
        kwargs = dict(raw_state=state) if selective else {}
        actions, logp, _, value = agent.take_action(tensor, action_mask=mask, action_penalty=penalty, deterministic=deterministic, **kwargs)
        info = dict(action_mask=mask, action_penalty=penalty, intervention_active=bool(active))
        if selective:
            info.update(actions=dict(agent.last_policy_actions), executed_actions=dict(actions), projection_active=bool(agent.last_projection_count), projection_original_indices=dict(agent.last_projection_original_indices), projection_projected_indices=dict(agent.last_projection_projected_indices))
        return (actions, logp, value, info)
    mask, _ = env.get_action_mask(state, mask_mode='physical' if agent.use_dynamic_mask else 'none')
    if skip_diagnostics and isinstance(agent, HouseholdPPO):
        return (agent.take_action(tensor, mask, deterministic, raw_state=state, actions_only=True), None, None, {})
    actions, logp, _, value, info = agent.take_action(tensor, mask, deterministic=deterministic, return_info=True, raw_state=state)
    return (actions, logp, value, info)
