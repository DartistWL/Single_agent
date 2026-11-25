"""Reference PPO algorithm."""
import copy
import csv
import os
from typing import Optional
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from model.common.Reference_PPO_networks import SharedFeatureExtractor, ActionBranch, ValueNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HomeEnergyPPO:

    def __init__(self, env, state_dim, hidden_dim, action_space_config, gamma=0.99, lmbda=0.98, eps=0.1, epochs=3, ent_coef=0.006, max_grad_norm=0.8, device=device, base_state_dim: Optional[int]=None, state_stack_len: int=8, auto_entropy: bool=True, actor_lr: float=8e-05, critic_lr: float=0.0003, ent_coef_lr: float=5e-05, ent_coef_min: float=0.0001, ent_coef_max: float=0.018, critic_target_clip: float=500.0, critic_coef: float=0.1, target_entropy_ratio: float=0.7, soft_policy_regularization: bool=False, action_magnitude_regularizer_weight: float=0.0, ev_ready_regularizer_weight: float=0.0, ev_ready_soc_floor: float=0.28, ev_charge_target_kw: float=3.3, ewh_low_temp_regularizer_weight: float=0.0, ewh_temp_floor_c: float=42.0, ewh_setpoint_floor_c: float=45.0, use_dynamic_mask: bool=False, mask_query_apply_soc: bool=False, mask_mode: str='none'):
        self.state_stack_len = max(1, int(state_stack_len))
        if base_state_dim is None:
            self.base_state_dim = int(state_dim) // self.state_stack_len
        else:
            self.base_state_dim = int(base_state_dim)
        self.use_dynamic_mask = bool(use_dynamic_mask)
        self.mask_query_apply_soc = bool(mask_query_apply_soc)
        self.mask_mode = str(mask_mode or 'none').lower()
        self.use_action_penalty = False
        self.use_state_normalization = True
        self.use_advantage_normalization = True
        self.action_mapping = {name: {idx: val for idx, val in enumerate(values)} for name, values in action_space_config.items()}
        self.action_dims = {name: len(values) for name, values in action_space_config.items()}
        self.shared_backbone = SharedFeatureExtractor(state_dim, hidden_dim).to(device)
        self.actor_branches = nn.ModuleDict({name: ActionBranch(hidden_dim, dim).to(device) for name, dim in self.action_dims.items()})
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optim = torch.optim.AdamW([{'params': self.shared_backbone.parameters()}, {'params': self.actor_branches.parameters()}], lr=actor_lr)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.epochs = epochs
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.ess_capacity = env.ess_capacity
        self.ev_capacity = env.ev_capacity
        self.auto_entropy = auto_entropy
        self.ent_coef_lr = float(ent_coef_lr)
        self.ent_coef_min = float(ent_coef_min)
        self.ent_coef_max = float(ent_coef_max)
        self.critic_target_clip = float(critic_target_clip or 0.0)
        self.critic_coef = float(critic_coef)
        self.soft_policy_regularization = bool(soft_policy_regularization)
        self.action_magnitude_regularizer_weight = float(action_magnitude_regularizer_weight or 0.0)
        self.ev_ready_regularizer_weight = float(ev_ready_regularizer_weight or 0.0)
        self.ev_ready_soc_floor = float(ev_ready_soc_floor)
        self.ev_charge_target_kw = float(ev_charge_target_kw)
        self.ewh_low_temp_regularizer_weight = float(ewh_low_temp_regularizer_weight or 0.0)
        self.ewh_temp_floor_c = float(ewh_temp_floor_c)
        self.ewh_setpoint_floor_c = float(ewh_setpoint_floor_c)
        branch_entropy_max = [np.log(max(1, dim)) for dim in self.action_dims.values()]
        self.target_entropy = float(target_entropy_ratio) * float(np.mean(branch_entropy_max))

    def _policy_regularization(self, branch_name, logits, raw_states, state_keys):
        if not self.soft_policy_regularization:
            return torch.tensor(0.0, device=self.device)
        action_values = torch.tensor(list(self.action_mapping[branch_name].values()), dtype=torch.float32, device=self.device)
        probs = torch.softmax(logits, dim=-1)
        regularizer = torch.tensor(0.0, device=self.device)
        if branch_name in {'battery_power', 'ev_power'} and self.action_magnitude_regularizer_weight > 0:
            max_abs = torch.clamp(action_values.abs().max(), min=1e-06)
            magnitude_cost = action_values.abs() / max_abs
            regularizer = regularizer + self.action_magnitude_regularizer_weight * (probs * magnitude_cost.unsqueeze(0)).sum(dim=1).mean()
        if branch_name == 'ev_power' and self.ev_ready_regularizer_weight > 0:
            try:
                ev_idx = state_keys.index('ev_battery_state')
                ev_soc = raw_states[:, ev_idx] / max(1e-08, self.ev_capacity)
                if 'ev_at_home' in state_keys:
                    home_mask = (raw_states[:, state_keys.index('ev_at_home')] > 0.5).float()
                else:
                    home_mask = torch.ones_like(ev_soc)
                urgency = torch.relu(self.ev_ready_soc_floor - ev_soc) / max(1e-08, self.ev_ready_soc_floor)
                urgency = torch.clamp(urgency * home_mask, 0.0, 1.0)
                positive_power = torch.clamp(action_values, min=0.0)
                target_kw = max(1e-06, self.ev_charge_target_kw)
                charge_gap = torch.relu(target_kw - positive_power) / target_kw
                expected_gap = (probs * charge_gap.unsqueeze(0)).sum(dim=1)
                regularizer = regularizer + self.ev_ready_regularizer_weight * (urgency * expected_gap).mean()
            except ValueError:
                pass
        if branch_name == 'ewh_set_temp' and self.ewh_low_temp_regularizer_weight > 0:
            try:
                temp_idx = state_keys.index('ewh_temp')
                temp_gap = torch.relu(self.ewh_temp_floor_c - raw_states[:, temp_idx]) / 10.0
                temp_gap = torch.clamp(temp_gap, 0.0, 1.0)
                denom = max(1e-06, self.ewh_setpoint_floor_c - float(action_values.min().item()))
                setpoint_gap = torch.relu(self.ewh_setpoint_floor_c - action_values) / denom
                expected_gap = (probs * setpoint_gap.unsqueeze(0)).sum(dim=1)
                regularizer = regularizer + self.ewh_low_temp_regularizer_weight * (temp_gap * expected_gap).mean()
            except ValueError:
                pass
        return regularizer

    def _safe_action_index(self, name):
        values = np.asarray(list(self.action_mapping[name].values()), dtype=float)
        zero = np.where(np.isclose(values, 0.0, atol=0.0001))[0]
        return int(zero[0]) if len(zero) else 0

    def _apply_action_mask(self, name, logits, action_mask=None):
        if not self.use_dynamic_mask or not action_mask or name not in action_mask:
            return logits
        mask = torch.as_tensor(action_mask[name], dtype=torch.bool, device=logits.device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        mask = mask.expand_as(logits).clone()
        invalid_rows = ~mask.any(dim=1)
        if bool(invalid_rows.any()):
            mask[invalid_rows, self._safe_action_index(name)] = True
        return torch.where(mask, logits, torch.full_like(logits, torch.finfo(logits.dtype).min))

    def take_action(self, state_tensor, action_mask=None, action_penalty=None, deterministic=False):
        shared_features = self.shared_backbone(state_tensor)
        actions = {}
        log_probs = []
        entropies = []
        for name, branch in self.actor_branches.items():
            logits = branch(shared_features)
            logits = self._apply_action_mask(name, logits, action_mask)
            dist = torch.distributions.Categorical(logits=logits)
            action_idx = torch.argmax(logits, dim=1) if deterministic else dist.sample()
            actions[name] = self.action_mapping[name][int(action_idx.item())]
            log_probs.append(dist.log_prob(action_idx))
            entropies.append(dist.entropy())
        total_log_prob = torch.stack(log_probs).sum(dim=0)
        total_entropy = torch.stack(entropies).sum()
        value = self.critic(state_tensor).item()
        return (actions, total_log_prob, total_entropy, value)

    def compute_gae(self, rewards, values, next_values, dones):
        batch_size = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(batch_size)):
            next_non_terminal = 1.0 - dones[t]
            next_value = next_values[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lmbda * next_non_terminal * last_gae
        returns = advantages + values
        if self.use_advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-08)
        return (advantages.detach(), returns.detach())

    def update(self, batch_data_list, running_stats, state_keys, episode, total_episodes):
        batch_data = {'states': [t['state'] for t in batch_data_list], 'actions': {name: [t['actions'][name] for t in batch_data_list] for name in self.action_mapping.keys()}, 'rewards': [t['rewards'] for t in batch_data_list], 'next_states': [t['next_state'] for t in batch_data_list], 'dones': [t['dones'] for t in batch_data_list], 'log_probs': [t['log_probs'] for t in batch_data_list], 'values': [t['values'] for t in batch_data_list], 'next_values': [t['next_values'] for t in batch_data_list], 'action_masks': [t.get('action_mask', {}) for t in batch_data_list]}
        normalized_states = torch.cat([t['stacked_state'] for t in batch_data_list], dim=0).to(device)
        normalized_next_states = torch.cat([t['stacked_next_state'] for t in batch_data_list], dim=0).to(device)
        raw_states = torch.stack([torch.tensor([t['state'][key] for key in state_keys], dtype=torch.float32, device=device) for t in batch_data_list])
        action_indices = {}
        for name in self.action_mapping:
            indices = []
            action_values = list(self.action_mapping[name].values())
            for val in batch_data['actions'][name]:
                close_idx = np.where(np.isclose(action_values, val, atol=0.0001))[0]
                if len(close_idx) > 0:
                    indices.append(close_idx[0])
                else:
                    idx = int(np.argmin(np.abs(np.array(action_values) - val)))
                    indices.append(idx)
            action_indices[name] = torch.tensor(indices, dtype=torch.long, device=device)
        rewards = torch.tensor(batch_data['rewards'], dtype=torch.float32, device=device)
        dones = torch.tensor(batch_data['dones'], dtype=torch.float32, device=device)
        old_log_probs = torch.stack([lp.squeeze() for lp in batch_data['log_probs']]).to(device)
        values = torch.tensor(batch_data['values'], dtype=torch.float32, device=device)
        next_values = torch.tensor(batch_data['next_values'], dtype=torch.float32, device=device)
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)
        epoch_losses = []
        for _ in range(self.epochs):
            critic_values = self.critic(normalized_states)
            with torch.no_grad():
                target_values = returns
                if self.critic_target_clip > 0:
                    target_values = torch.clamp(target_values, -self.critic_target_clip, self.critic_target_clip)
            critic_loss = F.smooth_l1_loss(critic_values, target_values)
            shared_features = self.shared_backbone(normalized_states)
            new_log_probs = []
            entropies = []
            policy_regularizer = torch.tensor(0.0, device=device)
            for name, branch in self.actor_branches.items():
                logits = branch(shared_features)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
                if self.use_dynamic_mask:
                    default_mask = [True] * len(self.action_mapping[name])
                    branch_mask = torch.tensor([masks.get(name, default_mask) for masks in batch_data['action_masks']], dtype=torch.bool, device=device)
                    logits = self._apply_action_mask(name, logits, action_mask={name: branch_mask})
                policy_regularizer = policy_regularizer + self._policy_regularization(name, logits, raw_states, state_keys)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs.append(dist.log_prob(action_indices[name]))
                entropies.append(dist.entropy())
            total_entropy = torch.stack(entropies).mean(dim=0).mean()
            new_log_probs = torch.stack(new_log_probs).sum(dim=0).squeeze()
            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * total_entropy + policy_regularizer
            if self.auto_entropy and total_episodes > 0:
                with torch.no_grad():
                    ent_error = total_entropy.item() - self.target_entropy
                    new_ent_coef = self.ent_coef - self.ent_coef_lr * ent_error
                    self.ent_coef = float(np.clip(new_ent_coef, self.ent_coef_min, self.ent_coef_max))
            total_loss = actor_loss + self.critic_coef * critic_loss
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            total_loss.backward()
            clip_grad_norm_(self.shared_backbone.parameters(), self.max_grad_norm)
            clip_grad_norm_(self.actor_branches.parameters(), self.max_grad_norm)
            clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.actor_optim.step()
            self.critic_optim.step()
            epoch_losses.append(total_loss.item())
        return (actor_loss.item(), critic_loss.item(), total_loss.item(), total_entropy.item())

def build_agent(env, device, blocks):
    import inspect
    from other.environment.config import CONFIG, apply_episode_stage_to_agent
    globals()['device'] = device
    alg = dict(CONFIG['algorithms']['ppo_base'])
    signature = inspect.signature(HomeEnergyPPO)
    params = {k: v for k, v in alg.items() if k in signature.parameters}
    n = len(env.state_space)
    params.update(env=env, state_dim=n * alg['state_stack_len'], hidden_dim=alg['hidden_dim'], action_space_config=env.action_space, base_state_dim=n, device=device, lmbda=alg['gae_lambda'], eps=alg['clip_param'], epochs=alg['n_epochs'])
    for prefix, section in [('ess', 'ess_health'), ('ev', 'ev')]:
        for edge in ('low', 'high'):
            key = f'{prefix}_{edge}_soc_weight'
            if key in signature.parameters:
                params[key] = CONFIG['environment'][section].get(f'{edge}_soc_weight', signature.parameters[key].default)
    agent = HomeEnergyPPO(**params)
    agent.mask_mode = alg.get('mask_mode') or 'none'
    agent.mask_query_apply_soc = alg.get('mask_query_apply_soc', alg.get('use_dynamic_mask', False))
    agent.ref_config = alg
    apply_episode_stage_to_agent(agent, alg)
    return agent
