"""Branching Rainbow algorithm."""
from __future__ import annotations
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class Replay:

    def __init__(self, capacity, state_dim, action_dim, mask_dim):
        self.capacity = int(capacity)
        self.size = 0
        self.position = 0
        self.max_priority = 1.0
        self.state = np.zeros((capacity, state_dim), dtype=np.float16)
        self.next_state = np.zeros_like(self.state)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_mask = np.zeros((capacity, mask_dim), dtype=np.bool_)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.discount = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.bool_)
        self.priority = np.zeros(capacity, dtype=np.float32)

    def add(self, state, action, reward, next_state, done, discount, next_mask):
        i = self.position
        self.state[i] = np.asarray(state, dtype=np.float32)
        self.next_state[i] = np.asarray(next_state, dtype=np.float32)
        self.action[i] = np.asarray(action, dtype=np.float32)
        self.next_mask[i] = np.asarray(next_mask, dtype=np.bool_)
        self.reward[i] = float(reward)
        self.done[i] = bool(done)
        self.discount[i] = float(discount)
        self.priority[i] = self.max_priority
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch, prioritized, beta):
        if self.size < batch:
            raise ValueError('Not enough replay transitions')
        if prioritized:
            p = np.maximum(self.priority[:self.size], 1e-06).astype(np.float64) ** 0.6
            p /= p.sum()
            indices = np.random.choice(self.size, batch, replace=True, p=p)
            weights = (self.size * p[indices]) ** (-beta)
            weights /= weights.max()
        else:
            indices = np.random.randint(0, self.size, size=batch)
            weights = np.ones(batch)
        return (indices, weights.astype(np.float32))

    def snapshot(self):
        return dict(capacity=self.capacity, size=self.size, position=self.position, max_priority=self.max_priority, **{k: getattr(self, k) for k in ('state', 'next_state', 'action', 'next_mask', 'reward', 'discount', 'done', 'priority')})

    def restore(self, payload):
        if payload['capacity'] != self.capacity or not 0 <= payload['size'] <= self.capacity or (not 0 <= payload['position'] < self.capacity):
            raise ValueError('Incompatible replay checkpoint')
        for name in ('state', 'next_state', 'action', 'next_mask', 'reward', 'discount', 'done', 'priority'):
            source = np.asarray(payload[name])
            target = getattr(self, name)
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(f'Wrong replay array {name}')
            target[:] = source
        self.size = int(payload['size'])
        self.position = int(payload['position'])
        self.max_priority = float(payload['max_priority'])
        if not math.isfinite(self.max_priority) or self.max_priority <= 0:
            raise ValueError('Invalid replay priority checkpoint')

class ZeroValue(nn.Module):

    def __init__(self):
        super().__init__()
        self.placeholder = nn.Parameter(torch.zeros(()))

    def forward(self, state):
        return state.new_zeros((len(state), 1))

def mlp(input_dim, output_dim, hidden=128):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, output_dim))

class NoisyLinear(nn.Module):

    def __init__(self, input_dim, output_dim, sigma=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_mu = nn.Parameter(torch.empty(output_dim, input_dim))
        self.weight_sigma = nn.Parameter(torch.empty(output_dim, input_dim))
        self.bias_mu = nn.Parameter(torch.empty(output_dim))
        self.bias_sigma = nn.Parameter(torch.empty(output_dim))
        self.register_buffer('weight_noise', torch.zeros(output_dim, input_dim))
        self.register_buffer('bias_noise', torch.zeros(output_dim))
        bound = 1 / math.sqrt(input_dim)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, sigma / math.sqrt(input_dim))
        nn.init.constant_(self.bias_sigma, sigma / math.sqrt(output_dim))
        self.reset_noise()

    @staticmethod
    def noise(size, device):
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        x = self.noise(self.input_dim, self.weight_mu.device)
        y = self.noise(self.output_dim, self.weight_mu.device)
        self.weight_noise.copy_(y.outer(x))
        self.bias_noise.copy_(y)

    def forward(self, value):
        weight = self.weight_mu + self.weight_sigma * self.weight_noise if self.training else self.weight_mu
        bias = self.bias_mu + self.bias_sigma * self.bias_noise if self.training else self.bias_mu
        return F.linear(value, weight, bias)

class BranchRainbow(nn.Module):

    def __init__(self, state_dim, dims, hidden, atoms):
        super().__init__()
        self.dims = dims
        self.atoms = atoms
        self.shared = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.value = nn.Sequential(NoisyLinear(hidden, hidden), nn.ReLU(), NoisyLinear(hidden, atoms))
        self.advantages = nn.ModuleDict({name: nn.Sequential(NoisyLinear(hidden, hidden), nn.ReLU(), NoisyLinear(hidden, dim * atoms)) for name, dim in dims.items()})

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(self, state):
        feature = self.shared(state)
        value = self.value(feature).view(-1, 1, self.atoms)
        result = {}
        for name, branch in self.advantages.items():
            advantage = branch(feature).view(-1, self.dims[name], self.atoms)
            result[name] = value + advantage - advantage.mean(dim=1, keepdim=True)
        return result

def categorical_projection(next_prob, reward, discount, done, support):
    atoms = support.numel()
    low = float(support[0])
    high = float(support[-1])
    delta = (high - low) / (atoms - 1)
    unbounded = reward[:, None] + discount[:, None] * (1 - done[:, None]) * support[None, :]
    target = unbounded.clamp(low, high)
    coordinate = (target - low) / delta
    lower = coordinate.floor().long().clamp(0, atoms - 1)
    upper = coordinate.ceil().long().clamp(0, atoms - 1)
    offset = (torch.arange(len(reward), device=reward.device) * atoms)[:, None]
    projected = torch.zeros_like(next_prob).view(-1)
    low_weight = upper.float() - coordinate + (lower == upper).float()
    high_weight = coordinate - lower.float()
    projected.index_add_(0, (lower + offset).reshape(-1), (next_prob * low_weight).reshape(-1))
    projected.index_add_(0, (upper + offset).reshape(-1), (next_prob * high_weight).reshape(-1))
    clipped = float(((unbounded < low) | (unbounded > high)).float().mean().detach())
    return (projected.view(-1, atoms), clipped)

class FairOffPolicy(nn.Module):

    def __init__(self, env, state_dim, base_state_dim, state_stack_len, hidden_dim, variant, device, gamma=0.995, capacity=16384, batch_size=128, reward_scale=10.0, update_every=4):
        super().__init__()
        self.variant = 'rainbow_branch'
        self.device = torch.device(device)
        self.env = env
        self.state_dim = int(state_dim)
        self.base_state_dim = int(base_state_dim)
        self.state_stack_len = int(state_stack_len)
        self.hidden_dim = int(hidden_dim)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.reward_scale = float(reward_scale)
        self.update_every = int(update_every)
        self.gradient_steps = 0
        self.use_state_normalization = True
        self.use_dynamic_mask = True
        from model.common.execution_signatures import DEVICE_ORDER
        self.device_order = tuple((n for n in DEVICE_ORDER if n in env.action_space))
        self.actions = {n: list(env.action_space[n]) for n in self.device_order}
        self.dims = {n: len(v) for n, v in self.actions.items()}
        self.action_dim = len(self.dims)
        if self.action_dim != len(env.action_space):
            raise ValueError('Unsupported device action order')
        self.mask_dim = sum(self.dims.values())
        self.critic = ZeroValue()
        self.actor = BranchRainbow(state_dim, self.dims, hidden_dim, 101)
        self.target_actor = copy.deepcopy(self.actor)
        self.register_buffer('support', torch.linspace(-100.0, 10.0, 101))
        self.to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.0001)
        critic_params = [self.critic.placeholder]
        self.critic_optim = torch.optim.Adam(critic_params, lr=0.0003)
        self.replay = Replay(capacity, state_dim, self.action_dim, self.mask_dim)
        self.last_update_diagnostics = {}
        self.last_update_minibatch_diagnostics = []
        self.last_update_epoch_diagnostics = []

    @property
    def shared_backbone(self):
        return self.actor.shared

    @property
    def actor_branches(self):
        return self.actor.advantages

    def configuration(self):
        return dict(variant='rainbow_branch', state_dim=self.state_dim, base_state_dim=self.base_state_dim, state_stack_len=self.state_stack_len, hidden_dim=self.hidden_dim, device_order=list(self.device_order), actions=self.actions, gamma=self.gamma, replay_capacity=self.replay.capacity, replay_state_storage='float16', state_normalization='fixed train-split physical-baseline statistics and declared ranges', batch_size=self.batch_size, reward_scale=self.reward_scale, update_every=self.update_every, continuous_to_discrete='nearest valid command on each branch', rainbow_scope='factorized branch C51/Double/Dueling/PER/3-step/NoisyNet')

    def take_action(self, state_tensor, action_mask=None, *, raw_state=None, deterministic=False, return_info=False, decoder=None):
        del raw_state
        if decoder not in (None, 'native'):
            raise ValueError('Decoder override is not defined for off-policy methods')
        masks = {n: list(action_mask[n]) if action_mask and n in action_mask else [True] * self.dims[n] for n in self.device_order}
        if any((len(masks[n]) != self.dims[n] or not any(masks[n]) for n in self.device_order)):
            raise ValueError('Invalid or empty action mask')
        with torch.no_grad():
            old_training = self.actor.training
            self.actor.train(not deterministic)
            if not deterministic:
                self.actor.reset_noise()
            distributions = self.actor(state_tensor.to(self.device))
            indices = []
            for name in self.device_order:
                q = (distributions[name].softmax(-1) * self.support).sum(-1)
                valid = torch.as_tensor(masks[name], dtype=torch.bool, device=self.device)
                indices.append(int(q.masked_fill(~valid[None, :], -torch.inf).argmax(-1).item()))
            self.actor.train(old_training)
            latent = [float(i) for i in indices]
            actions = {n: self.actions[n][i] for n, i in zip(self.device_order, indices)}
            zero = state_tensor.new_zeros(1)
            info = dict(execution_classes={}, offpolicy_action=latent, offpolicy_indices=indices)
            result = (actions, zero, zero, 0.0)
            return (*result, info) if return_info else result

    def append_rollouts(self, rows):
        horizon = 3
        if not rows:
            return
        states = torch.stack([r['stacked_state'].detach().reshape(-1) for r in rows]).cpu().numpy()
        next_states = torch.stack([r['stacked_next_state'].detach().reshape(-1) for r in rows]).cpu().numpy()
        for i, row in enumerate(rows):
            reward = 0.0
            last = row
            count = 0
            for nxt in rows[i:i + horizon]:
                reward += self.gamma ** count * float(nxt['rewards']) / self.reward_scale
                count += 1
                last = nxt
                if nxt['dones']:
                    break
            if count < horizon and (not last['dones']):
                raise ValueError('Incomplete n-step rollout at batch boundary')
            masks = last['next_mask']
            flat = [v for name in self.device_order for v in masks[name]]
            action = row['offpolicy_indices']
            self.replay.add(states[i], action, reward, next_states[i + count - 1], last['dones'], self.gamma ** count, flat)

    def update(self, rows, *args, **kwargs):
        del args, kwargs
        self.append_rollouts(rows)
        count = max(1, len(rows) // self.update_every)
        actor_losses = []
        critic_losses = []
        clipped = []
        for _ in range(count):
            if self.replay.size < self.batch_size:
                break
            loss = self._rainbow_step()
            actor_losses.append(loss[0])
            critic_losses.append(loss[1])
            clipped.append(loss[2])
            self.gradient_steps += 1
        actor = float(np.mean(actor_losses)) if actor_losses else 0.0
        critic = float(np.mean(critic_losses)) if critic_losses else 0.0
        self.last_update_diagnostics = dict(replay_size=self.replay.size, learner_updates=len(actor_losses), learner_steps_total=self.gradient_steps, support_clipped_fraction=float(np.mean(clipped)) if clipped else 0.0)
        self.last_update_minibatch_diagnostics = []
        self.last_update_epoch_diagnostics = []
        return dict(actor_loss=actor, critic_loss=critic, total_loss=actor + critic)

    def _batch(self, prioritized=False):
        beta = min(1.0, 0.4 + 0.6 * self.gradient_steps / 200000.0)
        indices, weights = self.replay.sample(self.batch_size, prioritized, beta)
        r = self.replay
        tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)
        return (indices, tensor(weights), tensor(r.state[indices]), tensor(r.action[indices]), tensor(r.reward[indices]), tensor(r.next_state[indices]), tensor(r.done[indices]), tensor(r.discount[indices]), torch.as_tensor(r.next_mask[indices], dtype=torch.bool, device=self.device))

    def _rainbow_step(self):
        indices, weights, states, actions, rewards, next_states, dones, discounts, masks = self._batch(prioritized=True)
        self.actor.reset_noise()
        self.target_actor.reset_noise()
        current = self.actor(states)
        with torch.no_grad():
            online_next = self.actor(next_states)
            target_next = self.target_actor(next_states)
        losses = []
        clipped = []
        offset = 0
        for j, name in enumerate(self.device_order):
            dim = self.dims[name]
            valid = masks[:, offset:offset + dim]
            offset += dim
            if not valid.any(dim=1).all():
                raise ValueError('Empty next-state mask')
            q = (online_next[name].softmax(-1) * self.support).sum(-1)
            selected = q.masked_fill(~valid, -torch.inf).argmax(-1)
            target_distribution = target_next[name].softmax(-1)[torch.arange(len(states), device=self.device), selected]
            projected, fraction = categorical_projection(target_distribution, rewards, discounts, dones, self.support)
            chosen = actions[:, j].long()
            logp = current[name].log_softmax(-1)[torch.arange(len(states), device=self.device), chosen]
            losses.append(-(projected * logp).sum(-1))
            clipped.append(fraction)
        per_sample = torch.stack(losses).mean(0)
        loss = (per_sample * weights).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite Rainbow loss')
        self.actor_optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 2.0, error_if_nonfinite=True)
        self.actor_optim.step()
        priorities = per_sample.detach().cpu().numpy() + 1e-06
        self.replay.priority[indices] = priorities
        self.replay.max_priority = max(self.replay.max_priority, float(np.max(priorities)))
        if (self.gradient_steps + 1) % 500 == 0:
            self.target_actor.load_state_dict(self.actor.state_dict())
        return (0.0, float(loss.detach()), float(np.mean(clipped)))

def build_agent(env, device, blocks):
    import json
    from pathlib import Path
    cfg = json.loads((Path(__file__).resolve().parents[1] / 'common/settings.json').read_text(encoding='utf8'))
    options = cfg['offpolicy_config']
    n = len(env.state_space)
    length = cfg['training']['state_stack_len']
    return FairOffPolicy(env, n * length, n, length, options['hidden_dim'], 'rainbow_branch', device, capacity=options['replay_capacity'], **{k: options[k] for k in ('gamma', 'batch_size', 'reward_scale', 'update_every')})
