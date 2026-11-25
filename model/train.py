"""Train policies on registered household blocks."""
import csv
import importlib
import random
from collections import deque
import numpy as np
import torch
from .factory import begin_block, decision, make_agent
from .paper.Branching_Rainbow import FairOffPolicy
from .catalog import checkpoint_path, training_log_path, legacy_name
from . import checkpoint_pickle
from evaluation.metrics import meter
from other.environment.household import SEED, STEPS_PER_DAY, environment

class Inputs:

    def __init__(self, p, agent, device):
        self.keys = p['norm_keys']
        self.device = device
        self.mean = np.asarray(p['norm_mean'])
        self.std = np.maximum(p['norm_std'], 1e-06)
        self.length = agent.state_stack_len
        self.normalize = getattr(agent, 'use_state_normalization', True)
        self.reset()

    def reset(self):
        self.history = deque([np.zeros(len(self.keys)) for _ in range(self.length - 1)], maxlen=self.length)

    def push(self, state):
        value = np.asarray([state[k] for k in self.keys], dtype=float)
        if self.normalize:
            value = np.clip((value - self.mean) / self.std, -5.0, 5.0)
        self.history.append(value)
        return torch.as_tensor(np.concatenate(self.history)[None, :], dtype=torch.float32, device=self.device)

def attach(agent, env):
    for name in ('env_ref', 'household', 'env'):
        if hasattr(agent, name):
            setattr(agent, name, env)

def checkpoint(path, agent, p, index, curve, env):
    path.parent.mkdir(parents=True, exist_ok=True)
    attach(agent, None)
    try:
        value = dict(id=p['id'], index=index, agent=agent, curve=curve, torch_rng=torch.get_rng_state(), method=path.stem, cuda_rng=torch.cuda.get_rng_state() if agent.device.type == 'cuda' else None, numpy_rng=np.random.get_state(), python_rng=random.getstate())
        temp = path.with_suffix('.tmp')
        torch.save(value, temp)
        temp.replace(path)
    finally:
        attach(agent, env)

def restore(path, name, p, device, env):
    placement = 'cpu' if device.type == 'cpu' else {'cuda:0': 'cuda:0'}
    saved = torch.load(path, map_location=placement, pickle_module=checkpoint_pickle, weights_only=False)
    if saved['id'] != p['id'] or saved['method'] not in (name, legacy_name(name)):
        raise ValueError('Checkpoint belongs to another experiment or method')
    agent = saved['agent']
    if str(agent.device) != str(device):
        raise ValueError('Resume this run on the original device type')
    implementation = getattr(agent, 'agent', agent)
    module = importlib.import_module(type(implementation).__module__)
    if hasattr(module, 'device'):
        module.device = device
    attach(agent, env)
    return (saved, agent)

def training(root, name, p, device):
    seed = SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    blocks = p['training_blocks']
    calendar = p['schedule']['train']
    env = environment(calendar, p['threshold_kw'])
    path = checkpoint_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        saved, agent = restore(path, name, p, device, env)
        index, curve = (saved['index'], saved['curve'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
    else:
        agent = make_agent(name, env, device, blocks)
        index, curve = (0, [])
    if index != len(curve) or not 0 <= index <= blocks:
        raise ValueError('Checkpoint progress does not match the registered training budget')
    environment_steps = sum((row['days'] for row in curve)) * STEPS_PER_DAY
    env.episode_schedule_cursor = index
    inputs = Inputs(p, agent, device)
    for i in range(index, blocks):
        begin_block(agent, i, blocks)
        state = env.reset()
        _, day, days = calendar[i % len(calendar)]
        if env.current_time != day or env._current_block_days != days:
            raise ValueError('Training calendar drift')
        inputs.reset()
        stacked = inputs.push(state)
        batch, total = ([], 0.0)
        block_steps = days * STEPS_PER_DAY
        for step in range(block_steps):
            with torch.no_grad():
                action, logp, value, info = decision(agent, env, state, stacked, False)
            nxt, reward, done = env.step(state, action)
            meter(env, state, nxt, reward)
            target = reward
            next_tensor = inputs.push(nxt)
            if batch:
                batch[-1]['next_values'] = value
            if done != (step + 1 == block_steps):
                raise ValueError('Training transition and block boundary disagree')
            c = env._last_reward_components
            row = dict(state=state, next_state=nxt, actions=action, rewards=target, dones=done, values=value, next_values=0.0, log_probs=logp.detach(), stacked_state=stacked, stacked_next_state=next_tensor, peak_excess_kw=c['peak_excess_kw'], peak_target_kw=c['peak_target_kw'], switching_cost=c['battery_switching_raw'], ramp_auc=c['ramp_auc_raw'], high_power_reversal=c['high_power_reversal_count'], micro_cycle=c['micro_cycle_count'])
            row.update(info)
            row.update(invalid_raw=c['invalid_raw'], ess_violation=c['ess_repair_distance_norm'], ev_violation=c['ev_repair_distance_norm'])
            if isinstance(agent, FairOffPolicy):
                row['next_mask'] = env.get_action_mask(nxt, mask_mode='physical')[0] if not done else {n: [True] * len(a) for n, a in env.action_space.items()}
            batch.append(row)
            state, stacked = (nxt, next_tensor)
            total += reward
        if not done:
            raise ValueError('Training horizon mismatch')
        agent.update(batch, None, inputs.keys, i, blocks)
        environment_steps += block_steps
        record = dict(block=i + 1, days=days, return_per_day=total / days, environment_steps=environment_steps)
        curve.append(record)
        if (i + 1) % 8 == 0 or i + 1 == blocks:
            checkpoint(path, agent, p, i + 1, curve, env)
            print(f'[{name}] {i + 1}/{blocks} | steps={environment_steps} | return/day={total / days:.3f}', flush=True)
    if environment_steps != p['budget']['train']['steps']:
        raise ValueError('Completed training steps do not match the registered budget')
    logfile = training_log_path(root, name)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=curve[0].keys())
        writer.writeheader()
        writer.writerows(curve)

def inference(root, name, p, env, device):
    saved, agent = restore(checkpoint_path(root, name), name, p, device, env)
    if saved['index'] != p['training_blocks']:
        raise ValueError('Training incomplete')

    class Policy:

        def __init__(self):
            self.inputs = Inputs(p, agent, device)

        def reset(self):
            self.inputs.reset()
            if hasattr(agent, 'reset_intervention_state'):
                agent.reset_intervention_state()

        @torch.no_grad()
        def take_action(self, state):
            tensor = self.inputs.push(state)
            return decision(agent, env, state, tensor, True, skip_diagnostics=True)[0]
    return Policy()

def _train(root, method, p, device):
    path = checkpoint_path(root, method)
    if p['seed'] != SEED and (not path.exists()):
        import numpy as np
        import torch
        random.seed(p['seed'])
        np.random.seed(p['seed'])
        torch.manual_seed(p['seed'])
        env = environment(p['schedule']['train'], p['threshold_kw'])
        agent = make_agent(method, env, device, p['training_blocks'])
        checkpoint(path, agent, p, 0, [], env)
    training(root, method, p, device)
