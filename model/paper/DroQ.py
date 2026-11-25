"""DroQ algorithm."""
from __future__ import annotations
import copy, csv, os, random, sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from other.environment.physics import HomeEnergyManagementEnv
from model.common.action_grid import build_wash_schedule_action_tuple

@dataclass
class DroQConfig:
    gamma: float = 0.96
    tau: float = 0.005
    lr_actor: float = 0.0003
    lr_critic: float = 0.0003
    lr_alpha: float = 0.0003
    batch_size: int = 256
    buffer_size: int = 500000
    min_buffer: int = 10000
    utd_ratio: int = 4
    dropout_p: float = 0.1
    hidden_dim: int = 512
    grad_clip: float = 10.0
    reward_scale: float = 0.05

def _sanitize_vec_np(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

def state_to_vector(state: Any, state_keys: Sequence[str]) -> np.ndarray:
    if isinstance(state, dict):
        v = np.array([float(state[k]) for k in state_keys], dtype=np.float32)
    else:
        v = np.asarray(state, dtype=np.float32)
    return _sanitize_vec_np(v)

class RunningStats:

    def __init__(self, shape: int):
        self.mean = torch.zeros(shape, device=device)
        self.std = torch.ones(shape, device=device)
        self.count = 0.0001

    def update(self, x: torch.Tensor) -> None:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        batch_mean = x.mean(dim=0)
        n_batch = x.size(0)
        if n_batch <= 1:
            batch_std = torch.zeros_like(batch_mean)
        else:
            batch_std = x.std(dim=0, unbiased=False)
        batch_count = n_batch
        batch_mean = torch.nan_to_num(batch_mean, nan=0.0, posinf=0.0, neginf=0.0)
        batch_std = torch.nan_to_num(batch_std, nan=0.0, posinf=1.0, neginf=0.0)
        batch_std = torch.clamp(batch_std, min=1e-06)
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / (self.count + batch_count)
        self.std = torch.sqrt((self.std ** 2 * self.count + batch_std ** 2 * batch_count + delta ** 2 * self.count * batch_count / (self.count + batch_count)) / (self.count + batch_count))
        self.std = torch.nan_to_num(self.std, nan=1.0, posinf=1.0, neginf=1e-06)
        self.std = torch.clamp(self.std, min=1e-06)
        self.count += batch_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.clamp(self.std + 1e-08, min=1e-06)
        out = (x - self.mean) / s
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

class ActionConverter:

    def __init__(self) -> None:
        self.action_map = {'ev_power': [-6.6, -3.3, 0, 3.3, 6.6], 'battery_power': [-4.4, -2.2, 0, 2.2, 4.4], 'wash_machine': list(build_wash_schedule_action_tuple(48, (6, 20), 1.0)), 'ac_temp': [16, 18, 20, 22, 24, 26, 28, 30], 'ac_temp2': [16, 18, 20, 22, 24, 26, 28, 30], 'ewh_temp': [40, 45, 50, 55, 60, 65, 70]}

    @staticmethod
    def _convert_single(value: float, options: List[Any]) -> Any:
        scaled = (float(value) + 1.0) / 2.0
        idx = int(round(scaled * (len(options) - 1)))
        return options[max(0, min(idx, len(options) - 1))]

    def continuous_to_discrete(self, continuous_action: np.ndarray) -> Dict[str, Any]:
        ca = np.asarray(continuous_action).reshape(-1)
        return {'ev_power': self._convert_single(ca[0], self.action_map['ev_power']), 'battery_power': self._convert_single(ca[1], self.action_map['battery_power']), 'wash_machine_schedule': self._convert_single(ca[2], self.action_map['wash_machine']), 'Air_conditioner_set_temp': self._convert_single(ca[3], self.action_map['ac_temp']), 'Air_conditioner_set_temp2': self._convert_single(ca[4], self.action_map['ac_temp2']), 'ewh_set_temp': self._convert_single(ca[5], self.action_map['ewh_temp'])}

def action_dict_to_vector(action: Dict[str, Any]) -> np.ndarray:
    return np.array([float(action['ev_power']), float(action['battery_power']), float(action['wash_machine_schedule']), float(action['Air_conditioner_set_temp']), float(action['Air_conditioner_set_temp2']), float(action['ewh_set_temp'])], dtype=np.float32)

def apply_physics_constraints(env: HomeEnergyManagementEnv, state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(action)
    if hasattr(env, 'is_ev_at_home') and (not env.is_ev_at_home()):
        out['ev_power'] = 0.0
    ess = float(state.get('ess_state', 0.0))
    if ess > 23.5:
        out['battery_power'] = min(float(out['battery_power']), 0.0)
    elif ess < 0.5:
        out['battery_power'] = max(float(out['battery_power']), 0.0)
    evs = float(state.get('ev_battery_state', 0.0))
    if evs > 23.5:
        out['ev_power'] = min(float(out['ev_power']), 0.0)
    elif evs < 0.5:
        out['ev_power'] = max(float(out['ev_power']), 0.0)
    return out

def _safe_normal(mean: torch.Tensor, std: torch.Tensor) -> Normal:
    mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)
    std = torch.nan_to_num(std, nan=1e-06, posinf=1.0, neginf=1e-06).clamp(min=1e-06, max=10.0)
    try:
        return Normal(mean, std, validate_args=False)
    except TypeError:
        return Normal(mean, std)

class DroQActor(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int=512, dev: torch.device=device):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.action_scale = torch.ones(action_dim, device=dev)
        self.action_bias = torch.zeros(action_dim, device=dev)
        nn.init.uniform_(self.mean.weight, -0.003, 0.003)
        nn.init.uniform_(self.mean.bias, -0.003, 0.003)
        nn.init.uniform_(self.log_std.weight, -0.003, 0.003)
        nn.init.uniform_(self.log_std.bias, -0.003, 0.003)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state = torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.net(state)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mean = self.mean(x)
        log_std = self.log_std(x)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)
        log_std = torch.nan_to_num(log_std, nan=0.0, posinf=2.0, neginf=-20.0)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return (mean, log_std)

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp().clamp(min=1e-06, max=10.0)
        normal = _safe_normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-06)
        return (action, log_prob.sum(1, keepdim=True))

class DroQCritic(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int=512, dropout_p: float=0.1):
        super().__init__()
        in_dim = state_dim + action_dim
        self.dropout_p = dropout_p
        self.q1 = self._make_trunk(in_dim, hidden_dim)
        self.q2 = self._make_trunk(in_dim, hidden_dim)

    def _make_trunk(self, in_dim: int, hidden_dim: int) -> nn.Module:
        return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(self.dropout_p), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(self.dropout_p), nn.Linear(hidden_dim, 1))

    def _set_dropout(self, enabled: bool) -> None:
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train(enabled)

    def forward(self, state: torch.Tensor, action: torch.Tensor, use_dropout: bool=True) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=1)
        self._set_dropout(use_dropout)
        return (self.q1(x), self.q2(x))

class DroQAgent:

    def __init__(self, state_dim: int, action_dim: int, cfg: DroQConfig, dev: torch.device=device, use_state_normalization: bool=True):
        self.cfg = cfg
        self.device = dev
        self.use_state_normalization = use_state_normalization
        self.action_dim = action_dim
        self.actor = DroQActor(state_dim, action_dim, cfg.hidden_dim, dev).to(dev)
        self.critic = DroQCritic(state_dim, action_dim, cfg.hidden_dim, cfg.dropout_p).to(dev)
        self.target_critic = DroQCritic(state_dim, action_dim, cfg.hidden_dim, cfg.dropout_p).to(dev)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=dev)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
        self.target_entropy = -float(action_dim) * 0.8
        self.converter = ActionConverter()

    def select_action(self, state_vec: np.ndarray, running_stats: Optional[RunningStats]=None, deterministic: bool=False) -> Tuple[Dict[str, Any], np.ndarray]:
        state_vec = _sanitize_vec_np(np.asarray(state_vec).reshape(-1))
        s = torch.FloatTensor(state_vec).to(self.device).unsqueeze(0)
        if self.use_state_normalization and running_stats is not None:
            s = running_stats.normalize(s).clamp(-5, 5)
        s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor.forward(s)
                a = torch.tanh(mean)
            else:
                a, _ = self.actor.sample(s)
        a_np = a.cpu().numpy().reshape(-1)
        discrete = self.converter.continuous_to_discrete(a_np)
        return (discrete, a_np)

    def update(self, batch: Tuple, running_stats: Optional[RunningStats]) -> Dict[str, float]:
        states, actions, next_states, rewards, dones = batch
        states = np.nan_to_num(np.asarray(states), nan=0.0, posinf=0.0, neginf=0.0)
        actions = np.nan_to_num(np.asarray(actions), nan=0.0, posinf=0.0, neginf=0.0)
        next_states = np.nan_to_num(np.asarray(next_states), nan=0.0, posinf=0.0, neginf=0.0)
        rewards = np.nan_to_num(np.asarray(rewards), nan=0.0, posinf=0.0, neginf=0.0)
        dones = np.nan_to_num(np.asarray(dones), nan=0.0, posinf=1.0, neginf=0.0)
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        if self.use_state_normalization and running_stats is not None:
            states_t = running_stats.normalize(states_t).clamp(-5, 5)
            next_states_t = running_stats.normalize(next_states_t).clamp(-5, 5)
        states_t = torch.nan_to_num(states_t, nan=0.0, posinf=0.0, neginf=0.0)
        next_states_t = torch.nan_to_num(next_states_t, nan=0.0, posinf=0.0, neginf=0.0)
        stats: Dict[str, float] = {}
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states_t)
            tq1, tq2 = self.target_critic.forward(next_states_t, next_actions, use_dropout=False)
            target_q = torch.min(tq1, tq2) - self.log_alpha.exp().detach() * next_log_probs
            target_value = rewards_t + (1.0 - dones_t) * self.cfg.gamma * target_q
            target_value = torch.nan_to_num(target_value, nan=0.0, posinf=10000.0, neginf=-10000.0).clamp(-10000.0, 10000.0)
        q1, q2 = self.critic.forward(states_t, actions_t, use_dropout=True)
        critic_loss = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)
        if not torch.isfinite(critic_loss):
            return {'critic_loss': float('nan'), 'actor_loss': float('nan'), 'alpha': float(self.log_alpha.exp().item())}
        self.critic_optim.zero_grad()
        critic_loss.backward()
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.critic_optim.step()
        stats['critic_loss'] = float(critic_loss.item())
        new_actions, log_probs = self.actor.sample(states_t)
        q1n, q2n = self.critic.forward(states_t, new_actions, use_dropout=False)
        actor_loss = (self.log_alpha.exp().detach() * log_probs - torch.min(q1n, q2n)).mean()
        if not torch.isfinite(actor_loss):
            return stats
        self.actor_optim.zero_grad()
        actor_loss.backward()
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.actor_optim.step()
        stats['actor_loss'] = float(actor_loss.item())
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        stats['alpha'] = float(self.log_alpha.exp().item())
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                tp.data.copy_(self.cfg.tau * p.data + (1.0 - self.cfg.tau) * tp.data)
        return stats

class ReplayBuffer:

    def __init__(self, capacity: int):
        self.buffer: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = []
        self.capacity = capacity

    def push(self, s: np.ndarray, a: np.ndarray, s2: np.ndarray, r: float, done: float) -> None:
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s, a, s2, r, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, next_states, rewards, dones = zip(*batch)
        return (np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32), np.array(next_states, dtype=np.float32), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.buffer)

def build_agent(env, device, blocks):
    from model.common.continuous_adapter import ContinuousDevicePolicy
    return ContinuousDevicePolicy('droq', env, device)
