"""CrossQ algorithm."""
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
from . import DroQ as dq

@dataclass
class CrossQConfig:
    gamma: float = 0.96
    lr_actor: float = 0.0003
    lr_critic: float = 0.0001
    lr_alpha: float = 0.0003
    batch_size: int = 256
    buffer_size: int = 500000
    min_buffer: int = 10000
    utd_ratio: int = 1
    hidden_dim: int = 512
    grad_clip: float = 10.0
    reward_scale: float = 0.05
    policy_delay: int = 3
    bn_momentum: float = 0.1
    bn_eps: float = 1e-05
    critic_loss_type: str = 'huber'
    huber_beta: float = 1.0
    q_target_clip: float = 300.0
    initial_log_alpha: float = -1.6
    log_alpha_min: float = -10.0
    log_alpha_max: float = 2.0
    target_entropy_scale: float = 1.0
    actor_q_in_critic_eval: bool = True

class CrossQCritic(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int=512, bn_momentum: float=0.01, bn_eps: float=1e-05):
        super().__init__()
        in_dim = state_dim + action_dim
        self.q1 = self._make_trunk(in_dim, hidden_dim, bn_momentum, bn_eps)
        self.q2 = self._make_trunk(in_dim, hidden_dim, bn_momentum, bn_eps)

    @staticmethod
    def _make_trunk(in_dim: int, hidden_dim: int, momentum: float, eps: float) -> nn.Sequential:
        return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim, momentum=momentum, eps=eps), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim, momentum=momentum, eps=eps), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return (self.q1(x), self.q2(x))

class CrossQAgent:

    def __init__(self, state_dim: int, action_dim: int, cfg: CrossQConfig, dev: torch.device=device, use_state_normalization: bool=True):
        self.cfg = cfg
        self.device = dev
        self.use_state_normalization = use_state_normalization
        self.action_dim = action_dim
        self.actor = dq.DroQActor(state_dim, action_dim, cfg.hidden_dim, dev).to(dev)
        self.critic = CrossQCritic(state_dim, action_dim, cfg.hidden_dim, bn_momentum=cfg.bn_momentum, bn_eps=cfg.bn_eps).to(dev)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self.log_alpha = torch.tensor([float(cfg.initial_log_alpha)], requires_grad=True, device=dev)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
        self.target_entropy = -float(action_dim) * float(cfg.target_entropy_scale)
        self.converter = dq.ActionConverter()
        self._critic_updates = 0

    def select_action(self, state_vec: np.ndarray, running_stats: Optional[dq.RunningStats]=None, deterministic: bool=False) -> Tuple[Dict[str, Any], np.ndarray]:
        state_vec = dq._sanitize_vec_np(np.asarray(state_vec).reshape(-1))
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

    def update(self, batch: Tuple, running_stats: Optional[dq.RunningStats]) -> Dict[str, float]:
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
            self.critic.eval()
            tq1, tq2 = self.critic(next_states_t, next_actions)
            self.critic.train()
            target_q = torch.min(tq1, tq2) - self.log_alpha.exp().detach() * next_log_probs
            target_value = rewards_t + (1.0 - dones_t) * self.cfg.gamma * target_q
            qc = float(self.cfg.q_target_clip)
            target_value = torch.nan_to_num(target_value, nan=0.0, posinf=qc, neginf=-qc).clamp(-qc, qc)
        q1, q2 = self.critic(states_t, actions_t)
        if self.cfg.critic_loss_type.lower() == 'mse':
            critic_loss = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)
        else:
            beta = float(self.cfg.huber_beta)
            critic_loss = F.smooth_l1_loss(q1, target_value, beta=beta) + F.smooth_l1_loss(q2, target_value, beta=beta)
        if not torch.isfinite(critic_loss):
            return {'critic_loss': float('nan'), 'actor_loss': float('nan'), 'alpha': float(self.log_alpha.exp().item())}
        self.critic_optim.zero_grad()
        critic_loss.backward()
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.critic_optim.step()
        stats['critic_loss'] = float(critic_loss.item())
        self._critic_updates += 1
        do_actor = self._critic_updates % max(1, int(self.cfg.policy_delay)) == 0
        new_actions, log_probs = self.actor.sample(states_t)
        if do_actor:
            if self.cfg.actor_q_in_critic_eval:
                self.critic.eval()
                q1n, q2n = self.critic(states_t, new_actions)
                self.critic.train()
            else:
                q1n, q2n = self.critic(states_t, new_actions)
            actor_loss = (self.log_alpha.exp().detach() * log_probs - torch.min(q1n, q2n)).mean()
            if torch.isfinite(actor_loss):
                self.actor_optim.zero_grad()
                actor_loss.backward()
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
                self.actor_optim.step()
                stats['actor_loss'] = float(actor_loss.item())
            else:
                stats['actor_loss'] = float('nan')
        else:
            stats['actor_loss'] = 0.0
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        with torch.no_grad():
            self.log_alpha.clamp_(self.cfg.log_alpha_min, self.cfg.log_alpha_max)
        stats['alpha'] = float(self.log_alpha.exp().item())
        return stats

def build_agent(env, device, blocks):
    from model.common.continuous_adapter import ContinuousDevicePolicy
    return ContinuousDevicePolicy('crossq', env, device)
