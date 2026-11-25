"""PPO core."""
from __future__ import annotations
import copy
import math
import time
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from .PPO_networks import ActionBranch, SharedFeatureExtractor, ValueNet
from .execution_classes import DEVICE_ORDER, SEMANTICS_VERSION, SIGNATURE_SIZE, build_classes
VARIANT_ALIASES = {'A': 'independent', 'A+': 'deduplicated', 'C': 'independent_class', 'E': 'execution_class'}
VARIANTS = tuple(VARIANT_ALIASES.values())

def class_log_probability(logits, mask, class_ids, selected_class_ids):
    mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    ids = torch.as_tensor(class_ids, dtype=torch.long, device=logits.device)
    chosen = torch.as_tensor(selected_class_ids, dtype=torch.long, device=logits.device).reshape(-1, 1)
    members = mask & (ids == chosen)
    if not mask.any(dim=-1).all() or not members.any(dim=-1).all():
        raise ValueError('A selected execution class must contain a feasible command')
    return torch.logsumexp(logits.masked_fill(~members, -torch.inf), dim=-1) - torch.logsumexp(logits.masked_fill(~mask, -torch.inf), dim=-1)

def all_class_log_probabilities(logits, mask, class_ids, class_count=None):
    ids = torch.as_tensor(class_ids, dtype=torch.long, device=logits.device)
    mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    if class_count is None:
        class_count = int(ids.max().item()) + 1
    denominator = torch.logsumexp(logits.masked_fill(~mask, -torch.inf), dim=-1)
    classes = torch.arange(class_count, device=logits.device)
    members = mask[:, None, :] & (ids[:, None, :] == classes[None, :, None])
    present = members.any(-1)
    safe = torch.where(present[:, :, None], members, mask[:, None, :])
    masses = torch.logsumexp(logits[:, None, :].masked_fill(~safe, -torch.inf), dim=-1)
    return (masses - denominator[:, None]).masked_fill(~present, -torch.inf)

def gae_scan(rewards, values, next_values, dones, gamma, lmbda):
    alive = (1 - dones).reshape((-1,) + (1,) * (rewards.ndim - 1))
    result = rewards + gamma * alive * next_values - values
    weight = gamma * lmbda * alive
    stride = 1
    while stride < len(rewards):
        result = torch.cat((result[:-stride] + weight[:-stride] * result[stride:], result[-stride:]))
        weight = torch.cat((weight[:-stride] * weight[stride:], weight[-stride:]))
        stride *= 2
    return result

def saved_probabilities(rows, key, name, logs):
    width = logs.shape[1]
    rows_at_head = [row[key][name] for row in rows]
    return logs.new_tensor([p + [0.0] * (width - len(p)) for p in rows_at_head])

class PrefixResidual(nn.Module):

    def __init__(self, input_dim, hidden_dim, rank):
        super().__init__()
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.orthogonal_(self.down.weight)
        nn.init.orthogonal_(self.up.weight, gain=0.01)

    def forward(self, prefix):
        return self.up(F.relu(self.down(prefix)))

class ConditionalExecutionPPO(nn.Module):

    def __init__(self, env, state_dim, hidden_dim=64, action_space_config=None, gamma=0.96, lmbda=0.98, eps=0.2, epochs=3, ent_coef=0.0, max_grad_norm=2.0, device=None, constraint_mode='none', use_state_normalization=True, use_advantage_normalization=True, use_dynamic_mask=True, constraint_config=None, base_state_dim: Optional[int]=None, state_stack_len=1, use_shared_backbone=True, variant='execution_class', prefix_rank=16, device_order=None, actor_lr=1e-05, critic_lr=3e-05, deduplicated_prior='uniform_class', minibatch_size=128, target_kl=0.02, initialize_actor_uniform=True, evaluation_decoder='native'):
        super().__init__()
        self.variant = VARIANT_ALIASES.get(variant, variant)
        if self.variant not in VARIANTS:
            raise ValueError(f'Unknown conditional PPO variant: {variant}')
        if constraint_mode != 'none':
            raise ValueError("Execution-class comparisons require constraint_mode='none'")
        if not use_shared_backbone and self.variant != 'independent':
            raise ValueError('Independent encoders are an ablation of variant A only')
        if deduplicated_prior not in ('uniform_class', 'class_size'):
            raise ValueError('deduplicated_prior must be uniform_class or class_size')
        self.deduplicated_prior = deduplicated_prior
        self.state_stack_len = max(1, int(state_stack_len))
        self.base_state_dim = int(state_dim) // self.state_stack_len if base_state_dim is None else int(base_state_dim)
        if self.base_state_dim * self.state_stack_len != int(state_dim):
            raise ValueError('state_dim must equal base_state_dim * state_stack_len')
        self.state_dim, self.hidden_dim = (int(state_dim), int(hidden_dim))
        action_space_config = env.action_space if action_space_config is None else action_space_config
        self.action_space = {name: list(values) for name, values in action_space_config.items()}
        self.action_mapping = {name: dict(enumerate(values)) for name, values in self.action_space.items()}
        self.action_dims = {name: len(values) for name, values in self.action_space.items()}
        if not self.action_dims or any((size < 1 for size in self.action_dims.values())):
            raise ValueError('Every action branch must be nonempty')
        default_order = [name for name in DEVICE_ORDER if name in self.action_dims]
        default_order += [name for name in self.action_dims if name not in default_order]
        self.device_order = tuple(default_order if device_order is None else device_order)
        if len(self.device_order) != len(self.action_dims) or set(self.device_order) != set(self.action_dims):
            raise ValueError('device_order must contain every branch exactly once')
        self.conditional = self.variant in ('command_ar', 'semantic_command', 'execution_class')
        self.class_objective = self.variant in ('independent_class', 'execution_class')
        self.semantic_prefix = self.variant in ('semantic_command', 'execution_class')
        self.canonical_execution = self.variant in ('deduplicated', 'independent_class', 'semantic_command', 'execution_class')
        self.prefix_rank = int(prefix_rank)
        if self.prefix_rank < 1:
            raise ValueError('prefix_rank must be positive')
        self.prefix_dim = max(sum((size + 1 for size in self.action_dims.values())), len(self.action_dims) * SIGNATURE_SIZE)
        self.shared_backbone = SharedFeatureExtractor(self.state_dim, self.hidden_dim) if use_shared_backbone else nn.Identity()
        self.actor_branches = nn.ModuleDict({name: ActionBranch(self.hidden_dim, dim) if use_shared_backbone else nn.Sequential(SharedFeatureExtractor(self.state_dim, self.hidden_dim), ActionBranch(self.hidden_dim, dim)) for name, dim in self.action_dims.items()})
        self.critic = ValueNet(self.state_dim, self.hidden_dim)
        self.prefix_adapters = nn.ModuleDict({name: PrefixResidual(self.prefix_dim, self.hidden_dim, self.prefix_rank) for name in self.device_order} if self.conditional else {})
        self.initialize_actor_uniform = bool(initialize_actor_uniform)
        if self.initialize_actor_uniform:
            for branch in self.actor_branches.values():
                output = [layer for layer in branch.modules() if isinstance(layer, nn.Linear)][-1]
                nn.init.zeros_(output.weight)
                nn.init.zeros_(output.bias)
        if evaluation_decoder not in ('native', 'command', 'class'):
            raise ValueError('evaluation_decoder must be native, command or class')
        if evaluation_decoder == 'class' and self.variant == 'command_ar':
            raise ValueError('Class decoding is undefined for alias-conditioned command_ar')
        self.evaluation_decoder = evaluation_decoder
        self.device = torch.device(device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.actor_optim = torch.optim.AdamW(self.actor_parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=critic_lr)
        self.actor_scheduler = self.critic_scheduler = None
        self.gamma, self.lmbda, self.eps = (float(gamma), float(lmbda), float(eps))
        if not 0 <= self.gamma <= 1 or not 0 <= self.lmbda <= 1 or (not 0 < self.eps < 1):
            raise ValueError('gamma/lmbda must be in [0,1] and PPO clip eps in (0,1)')
        self.epochs, self.max_grad_norm = (int(epochs), float(max_grad_norm))
        self.minibatch_size, self.target_kl = (int(minibatch_size), float(target_kl))
        if self.epochs < 1 or self.minibatch_size < 1:
            raise ValueError('epochs and minibatch_size must be positive')
        if not math.isfinite(self.target_kl) or self.target_kl < 0:
            raise ValueError('target_kl must be finite and nonnegative; zero disables stopping')
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError('max_grad_norm must be positive and finite')
        self.requested_ent_coef = self.ent_coef = float(ent_coef)
        if not math.isfinite(self.ent_coef) or self.ent_coef < 0:
            raise ValueError('ent_coef must be finite and nonnegative')
        self.entropy_regularization_enabled = self.ent_coef > 0
        self.constraint_mode, self.use_shared_backbone = ('none', bool(use_shared_backbone))
        self.use_state_normalization = bool(use_state_normalization)
        self.use_advantage_normalization = bool(use_advantage_normalization)
        self.use_dynamic_mask = bool(use_dynamic_mask)
        self.constraint_config = dict(constraint_config or {'soc_lower': 0.1, 'soc_upper': 0.9})
        self.ess_capacity = getattr(env, 'ess_capacity', 1.0)
        self.ev_capacity = getattr(env, 'ev_capacity', 1.0)

    def actor_parameters(self):
        return (parameter for module in (self.shared_backbone, self.actor_branches, self.prefix_adapters) for parameter in module.parameters())

    def _masks(self, masks):
        result = {}
        for name, width in self.action_dims.items():
            value = [True] * width if masks is None or name not in masks else masks[name]
            tensor = torch.as_tensor(value, dtype=torch.bool).reshape(-1)
            if tensor.numel() != width or not tensor.any():
                raise ValueError(f'Invalid or empty action mask for {name}')
            result[name] = tensor.cpu().tolist()
        return result

    def _validate_classes(self, classes, masks):
        for name, width in self.action_dims.items():
            table = classes[name]
            ids, reps, signatures = (table['class_ids'], table['representatives'], table['signatures'])
            if len(ids) != width or not reps or len(reps) != len(signatures):
                raise ValueError(f'Invalid execution-class shape for {name}')
            if any(((ids[i] < 0) != (not valid) for i, valid in enumerate(masks[name]))):
                raise ValueError(f'Execution classes disagree with physical mask for {name}')
            if set((ids[i] for i, valid in enumerate(masks[name]) if valid)) != set(range(len(reps))):
                raise ValueError(f'Execution-class IDs must be contiguous for {name}')
            for class_id, representative in enumerate(reps):
                if not 0 <= representative < width or ids[representative] != class_id:
                    raise ValueError(f'Illegal representative for {name}')
                signature = signatures[class_id]
                if len(signature) != SIGNATURE_SIZE or not all((math.isfinite(v) for v in signature)):
                    raise ValueError(f'Invalid semantic signature for {name}')

    def _prefix(self, sampled, selected, classes):
        result = [0.0] * self.prefix_dim
        offset = 0
        for name in self.device_order:
            if self.semantic_prefix:
                if name in selected:
                    result[offset:offset + SIGNATURE_SIZE] = classes[name]['signatures'][selected[name]]
                offset += SIGNATURE_SIZE
            else:
                if name in sampled:
                    result[offset] = 1.0
                    result[offset + 1 + sampled[name]] = 1.0
                offset += self.action_dims[name] + 1
        return result

    def _head(self, features, name, prefix, physical_mask, tables):
        if self.conditional:
            features = features + self.prefix_adapters[name](prefix)
        logits = self.actor_branches[name](features)
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f'Non-finite logits for {name}')
        distribution_mask = physical_mask
        if self.variant == 'deduplicated':
            counts = []
            for table in tables:
                row = [0] * self.action_dims[name]
                for rep, members in zip(table[name]['representatives'], table[name]['members']):
                    row[rep] = len(members)
                counts.append(row)
            sizes = logits.new_tensor(counts)
            distribution_mask = sizes > 0
            if self.deduplicated_prior == 'class_size':
                logits = logits + sizes.clamp_min(1).log()
        return (logits, distribution_mask)

    @torch.no_grad()
    def take_action(self, state_tensor, action_mask=None, deterministic=False, return_info=False, *, raw_state=None, decoder=None, actions_only=False):
        if state_tensor.ndim != 2 or state_tensor.shape != (1, self.state_dim):
            raise ValueError('take_action expects one complete stacked network input')
        if raw_state is None:
            raise ValueError('raw_state is required for causal physical execution classes')
        if actions_only and (not deterministic):
            raise ValueError('Actions-only decoding is for deterministic evaluation')
        requested_decoder = self.evaluation_decoder if decoder is None else decoder
        if requested_decoder not in ('native', 'command', 'class'):
            raise ValueError('decoder must be native, command or class')
        if decoder not in (None, 'native') and (not deterministic):
            raise ValueError('A decoder override is only valid for deterministic evaluation')
        class_decode = self.canonical_execution if requested_decoder == 'native' else requested_decoder == 'class'
        if deterministic and class_decode and (self.variant == 'command_ar'):
            raise ValueError('Class decoding is undefined for alias-conditioned command_ar')
        state_tensor = state_tensor.to(self.device)
        masks = self._masks(action_mask)
        class_started = time.perf_counter()
        classes = build_classes(self.household, raw_state, self.action_space, masks)
        self._validate_classes(classes, masks)
        class_seconds = time.perf_counter() - class_started
        features = self.shared_backbone(state_tensor)
        sampled, representatives, selected, prefixes, actions = ({}, {}, {}, {}, {})
        command_logps, class_logps, command_entropies, class_entropies = ([], [], [], [])
        branch_command_probabilities, branch_class_probabilities = ({}, {})
        for name in self.device_order:
            prefix = self._prefix(sampled, selected, classes) if self.conditional else [0.0] * self.prefix_dim
            prefixes[name] = prefix
            physical_mask = torch.tensor([masks[name]], dtype=torch.bool, device=self.device)
            logits, mask = self._head(features, name, torch.tensor([prefix], device=self.device), physical_mask, [classes])
            masked_logits = logits.masked_fill(~mask, -torch.inf)
            if not actions_only:
                distribution = torch.distributions.Categorical(logits=masked_logits)
            if not actions_only or class_decode:
                logq_all = all_class_log_probabilities(logits, mask, [classes[name]['class_ids']], len(classes[name]['members']))
            if deterministic and class_decode:
                class_id = int(logq_all.argmax(dim=-1).item())
                representative = classes[name]['representatives'][class_id]
                index = torch.tensor([representative], device=self.device)
                sampled_index = representative
            else:
                index = masked_logits.argmax(dim=-1) if deterministic else distribution.sample()
                sampled_index = int(index.item())
                class_id = classes[name]['class_ids'][sampled_index]
                representative = classes[name]['representatives'][class_id]
            sampled[name] = sampled_index
            selected[name] = class_id
            representatives[name] = int(representative)
            execution_index = representative if self.canonical_execution else sampled[name]
            actions[name] = self.action_mapping[name][execution_index]
            if actions_only:
                continue
            if return_info:
                branch_command_probabilities[name] = distribution.probs[0]
                branch_class_probabilities[name] = logq_all.exp()[0]
            command_logps.append(distribution.log_prob(index))
            class_logps.append(logq_all[:, class_id])
            command_entropies.append(distribution.entropy())
            class_entropies.append(-(logq_all.exp() * logq_all.masked_fill(~torch.isfinite(logq_all), 0)).sum(-1))
        if actions_only:
            return actions
        command_logp = torch.stack(command_logps).sum(0)
        class_logp = None if self.variant == 'command_ar' else torch.stack(class_logps).sum(0)
        logp = class_logp if self.class_objective else command_logp
        entropy = torch.stack(class_entropies if self.class_objective else command_entropies).sum()
        result = (actions, logp, entropy, float(self.critic(state_tensor).item()))
        if return_info:

            def snapshot(probabilities):
                widths = [p.numel() for p in probabilities.values()]
                packed = torch.cat(tuple(probabilities.values())).cpu().split(widths)
                return {name: p.tolist() for name, p in zip(probabilities, packed)}
            return (*result, {'sampled_action_indices': sampled, 'representative_action_indices': representatives, 'selected_class_ids': selected, 'action_masks': masks, 'execution_classes': classes, 'prefix_features': prefixes, 'old_command_log_prob': float(command_logp.item()), 'old_class_log_prob': None if class_logp is None else float(class_logp.item()), 'old_branch_command_probabilities': snapshot(branch_command_probabilities), 'old_branch_class_probabilities': snapshot(branch_class_probabilities), 'policy_variant': self.variant, 'semantics_version': SEMANTICS_VERSION, 'deterministic_action': bool(deterministic), 'evaluation_decoder': ('class' if class_decode else 'command') if deterministic else 'stochastic', 'class_construction_seconds': class_seconds})
        return result

    def evaluate_actions(self, state_tensor, sampled_action_indices, action_masks, execution_classes=None, prefix_features=None, selected_class_ids=None, return_details=False):
        batch_size = state_tensor.shape[0]
        if execution_classes is None or prefix_features is None:
            raise ValueError('Saved execution_classes and prefix_features are required')
        tables = [execution_classes] if isinstance(execution_classes, dict) else execution_classes
        prefixes = [prefix_features] if isinstance(prefix_features, dict) else prefix_features
        selected = [selected_class_ids] if isinstance(selected_class_ids, dict) else selected_class_ids
        if len(tables) != batch_size or len(prefixes) != batch_size:
            raise ValueError('Saved metadata batch size differs from network input')
        features = self.shared_backbone(state_tensor.to(self.device))
        command_logps, class_logps, command_entropies, class_entropies = ([], [], [], [])
        branch_command_logs, branch_class_logs = ({}, {})
        for name in self.device_order:
            mask = torch.as_tensor(action_masks[name], dtype=torch.bool, device=self.device).reshape(batch_size, -1)
            indices = torch.as_tensor(sampled_action_indices[name], dtype=torch.long, device=self.device).reshape(-1)
            if mask.shape != (batch_size, self.action_dims[name]) or indices.numel() != batch_size:
                raise ValueError(f'Wrong rollout dimensions for {name}')
            if not mask.any(-1).all() or ((indices < 0) | (indices >= self.action_dims[name])).any():
                raise ValueError(f'Invalid saved command or mask for {name}')
            if not mask.gather(1, indices[:, None]).all():
                raise ValueError(f'Saved latent command violates mask for {name}')
            prefix = torch.tensor([p[name] for p in prefixes], dtype=features.dtype, device=self.device)
            if prefix.shape != (batch_size, self.prefix_dim) or not torch.isfinite(prefix).all():
                raise ValueError(f'Invalid saved prefix for {name}')
            logits, distribution_mask = self._head(features, name, prefix, mask, tables)
            if not distribution_mask.gather(1, indices[:, None]).all():
                raise ValueError(f'Saved latent command violates representative mask for {name}')
            distribution = torch.distributions.Categorical(logits=logits.masked_fill(~distribution_mask, -torch.inf))
            ids = torch.tensor([table[name]['class_ids'] for table in tables], dtype=torch.long, device=self.device)
            chosen = ids.gather(1, indices[:, None]).squeeze(1)
            if selected is not None and chosen.tolist() != [row[name] for row in selected]:
                raise ValueError(f'Saved class and latent command disagree for {name}')
            command_logps.append(distribution.log_prob(indices))
            command_entropies.append(distribution.entropy())
            class_count = max((len(table[name]['members']) for table in tables))
            logq_all = all_class_log_probabilities(logits, distribution_mask, ids, class_count)
            class_logps.append(logq_all.gather(1, chosen[:, None]).squeeze(1))
            branch_command_logs[name] = distribution.logits
            branch_class_logs[name] = logq_all
            class_entropies.append(-(logq_all.exp() * logq_all.masked_fill(~torch.isfinite(logq_all), 0)).sum(-1))
        command_logp = torch.stack(command_logps).sum(0)
        class_logp = None if self.variant == 'command_ar' else torch.stack(class_logps).sum(0)
        objective_logp = class_logp if self.class_objective else command_logp
        entropy = torch.stack(class_entropies if self.class_objective else command_entropies).sum(0)
        if return_details:
            return {'log_prob': objective_logp, 'entropy': entropy, 'command_log_prob': command_logp, 'class_log_prob': class_logp, 'branch_command_logs': branch_command_logs, 'branch_class_logs': branch_class_logs}
        return (objective_logp, entropy)

    def compute_gae(self, rewards, values, next_values, dones, normalize=None):
        advantages = gae_scan(rewards, values, next_values, dones, self.gamma, self.lmbda)
        normalize = self.use_advantage_normalization if normalize is None else bool(normalize)
        if normalize and len(rewards) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-08)
        return advantages

    def critic_targets(self, rows, states, dones, scalar_targets):
        return scalar_targets

    def critic_loss(self, states, targets):
        return F.mse_loss(self.critic(states), targets)

    def update(self, batch_data_list, running_stats=None, state_keys=None, episode=0, total_episodes=1):
        del running_stats, episode, total_episodes
        if not batch_data_list:
            raise ValueError('PPO update requires at least one transition')
        required = ('stacked_state', 'sampled_action_indices', 'action_masks', 'execution_classes', 'prefix_features', 'selected_class_ids', 'old_command_log_prob', 'old_class_log_prob')
        for row in batch_data_list:
            if any((key not in row for key in required)):
                raise ValueError('Rollout lacks exact network input or execution-class sampling metadata')
            if row.get('policy_variant', self.variant) != self.variant:
                raise ValueError('Rollout belongs to a different policy variant')
            if row.get('semantics_version', SEMANTICS_VERSION) != SEMANTICS_VERSION:
                raise ValueError('Rollout uses a different execution-semantics version')
            if row.get('deterministic_action', False):
                raise ValueError('Deterministic evaluation actions cannot be used as PPO samples')
            self._validate_classes(row['execution_classes'], row['action_masks'])
        states = torch.cat([row['stacked_state'].detach() for row in batch_data_list]).to(self.device)
        if states.shape != (len(batch_data_list), self.state_dim) or not torch.isfinite(states).all():
            raise ValueError('Saved normalized stacked states have wrong shape or non-finite values')

        def scalars(key):
            items = [row[key] for row in batch_data_list]
            if isinstance(items[0], torch.Tensor):
                return torch.stack(items).reshape(-1).to(device=self.device, dtype=torch.float32)
            return torch.tensor(items, dtype=torch.float32, device=self.device)
        rewards, dones = (scalars('rewards'), scalars('dones'))
        values, next_values = (scalars('values'), scalars('next_values'))
        old_key = 'old_class_log_prob' if self.class_objective else 'old_command_log_prob'
        old_log_probs = scalars(old_key)
        if not all((torch.isfinite(tensor).all() for tensor in (rewards, dones, values, next_values, old_log_probs))):
            raise FloatingPointError('Non-finite rollout reward, value or probability')
        if not ((dones == 0) | (dones == 1)).all():
            raise ValueError('Saved termination flags must be zero or one')
        if 'log_probs' in batch_data_list[0] and (not torch.allclose(scalars('log_probs'), old_log_probs, atol=1e-05)):
            raise ValueError("Saved PPO probability differs from the variant's sampling objective")
        indices = {name: [row['sampled_action_indices'][name] for row in batch_data_list] for name in self.action_dims}
        masks = {name: [row['action_masks'][name] for row in batch_data_list] for name in self.action_dims}
        tables = [row['execution_classes'] for row in batch_data_list]
        prefixes = [row['prefix_features'] for row in batch_data_list]
        selected = [row['selected_class_ids'] for row in batch_data_list]
        raw_advantages = self.compute_gae(rewards, values, next_values, dones, normalize=False).detach()
        targets = (raw_advantages + values).detach()
        advantages = raw_advantages.clone()
        if self.use_advantage_normalization and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-08)
        if not torch.isfinite(targets).all() or not torch.isfinite(advantages).all():
            raise FloatingPointError('Non-finite GAE advantage or lambda-return target')
        critic_targets = self.critic_targets(batch_data_list, states, dones, targets)

        def evaluate_subset(selected_indices, details=False):
            positions = selected_indices.detach().cpu().tolist()
            return self.evaluate_actions(states[selected_indices], {name: [indices[name][i] for i in positions] for name in self.action_dims}, {name: [masks[name][i] for i in positions] for name in self.action_dims}, [tables[i] for i in positions], [prefixes[i] for i in positions], [selected[i] for i in positions], return_details=details)
        all_indices = torch.arange(len(states), device=self.device)
        with torch.no_grad():
            initial_logp, _ = evaluate_subset(all_indices)
            self.last_update_initial_ratios = (initial_logp - old_log_probs).exp()
            initial_error = float((self.last_update_initial_ratios - 1).abs().max())
            if not math.isfinite(initial_error) or initial_error > 0.0005:
                raise ValueError('Rollout likelihood no longer matches its saved policy/input/mask')
        actor_parameters = list(self.actor_parameters())
        critic_parameters = list(self.critic.parameters())
        minibatch_records, epoch_records = ([], [])
        actor_losses, critic_losses, total_losses = ([], [], [])
        optimizer_steps, stopped = (0, False)
        for epoch in range(self.epochs):
            permutation = torch.randperm(len(states), device=self.device)
            for start in range(0, len(states), self.minibatch_size):
                batch_indices = permutation[start:start + self.minibatch_size]
                new_log_probs, entropy = evaluate_subset(batch_indices)
                log_ratio = new_log_probs - old_log_probs[batch_indices]
                ratios = log_ratio.exp()
                if not torch.isfinite(ratios).all() or not torch.isfinite(entropy).all():
                    raise FloatingPointError('Non-finite joint PPO ratio or entropy')
                approx_kl = float((ratios - 1 - log_ratio).mean().detach())
                clip_fraction = float(((ratios - 1).abs() > self.eps).float().mean().detach())
                if self.target_kl and approx_kl > 1.5 * self.target_kl:
                    minibatch_records.append(dict(epoch=epoch + 1, minibatch=start // self.minibatch_size + 1, samples=len(batch_indices), approx_kl=approx_kl, clip_fraction=clip_fraction, skipped_for_kl=True))
                    stopped = True
                    break
                actor_loss = -torch.minimum(ratios * advantages[batch_indices], ratios.clamp(1 - self.eps, 1 + self.eps) * advantages[batch_indices]).mean()
                actor_loss = actor_loss - self.ent_coef * entropy.mean()
                critic_loss = self.critic_loss(states[batch_indices], critic_targets[batch_indices])
                total_loss = actor_loss + 0.5 * critic_loss
                if not torch.isfinite(total_loss):
                    raise FloatingPointError('Non-finite PPO minibatch loss')
                self.actor_optim.zero_grad(set_to_none=True)
                self.critic_optim.zero_grad(set_to_none=True)
                total_loss.backward()
                actor_norm = clip_grad_norm_(actor_parameters, self.max_grad_norm, error_if_nonfinite=True)
                critic_norm = clip_grad_norm_(critic_parameters, self.max_grad_norm, error_if_nonfinite=True)
                self.actor_optim.step()
                self.critic_optim.step()
                optimizer_steps += 1
                actor_losses.append(float(actor_loss.detach()))
                critic_losses.append(float(critic_loss.detach()))
                total_losses.append(float(total_loss.detach()))
                minibatch_records.append(dict(epoch=epoch + 1, minibatch=start // self.minibatch_size + 1, samples=len(batch_indices), approx_kl=approx_kl, clip_fraction=clip_fraction, actor_grad_norm=float(actor_norm), critic_grad_norm=float(critic_norm), actor_loss=actor_losses[-1], critic_loss=critic_losses[-1], skipped_for_kl=False))
            with torch.no_grad():
                epoch_logp, _ = evaluate_subset(all_indices)
                epoch_log_ratio = epoch_logp - old_log_probs
                epoch_ratio = epoch_log_ratio.exp()
                if not torch.isfinite(epoch_ratio).all():
                    raise FloatingPointError('Non-finite epoch PPO probability ratio')
                epoch_kl = float((epoch_ratio - 1 - epoch_log_ratio).mean())
                epoch_records.append(dict(epoch=epoch + 1, approx_kl=epoch_kl, clip_fraction=float(((epoch_ratio - 1).abs() > self.eps).float().mean()), completed=not stopped))
            if stopped or (self.target_kl and epoch_kl > 1.5 * self.target_kl):
                stopped = True
                break
        if not optimizer_steps:
            raise ValueError('No optimizer steps completed; saved policy differs from rollout')
        with torch.no_grad():
            final = self.evaluate_actions(states, indices, masks, tables, prefixes, selected, True)
            prediction = self.critic(states)
            variance = targets.var(unbiased=False)
            explained_variance = 1 - (targets - prediction).var(unbiased=False) / variance if float(variance) > 1e-12 else torch.zeros((), device=self.device)
            completed = [row for row in minibatch_records if not row['skipped_for_kl']]
            average = lambda field: sum((row[field] for row in completed)) / len(completed)
            diagnostics = {'initial_ratio_max_error': initial_error, 'optimizer_steps': optimizer_steps, 'epochs_completed': sum((int(row['completed']) for row in epoch_records)), 'epochs_attempted': len(epoch_records), 'samples_optimized': sum((row['samples'] for row in completed)), 'kl_early_stop': int(stopped), 'explained_variance': float(explained_variance), 'minibatch_approx_kl_mean': average('approx_kl'), 'minibatch_approx_kl_max': max((row['approx_kl'] for row in minibatch_records)), 'minibatch_clip_fraction_mean': average('clip_fraction'), 'minibatch_clip_fraction_max': max((row['clip_fraction'] for row in minibatch_records)), 'epoch_approx_kl_max': max((row['approx_kl'] for row in epoch_records)), 'epoch_clip_fraction_max': max((row['clip_fraction'] for row in epoch_records)), 'actor_grad_norm_mean': average('actor_grad_norm'), 'critic_grad_norm_mean': average('critic_grad_norm'), 'value_target_mean': float(targets.mean()), 'value_target_std': float(targets.std(unbiased=False)), 'effective_entropy_coefficient': self.ent_coef}
            for unit in ('command', 'class'):
                if unit == 'class' and self.variant == 'command_ar':
                    continue
                key = f'old_branch_{unit}_probabilities'
                if not all((key in row for row in batch_data_list)):
                    continue
                conditional_kl = torch.zeros(len(batch_data_list), device=self.device)
                for name in self.device_order:
                    logs = final[f'branch_{unit}_logs'][name]
                    probs = saved_probabilities(batch_data_list, key, name, logs)
                    positive = probs > 0
                    conditional_kl += (probs * (probs.clamp_min(1e-30).log() - logs.masked_fill(~positive, 0))).sum(-1)
                diagnostics[f'{unit}_conditional_kl'] = float(conditional_kl.mean())
                old = scalars(f'old_{unit}_log_prob')
                ratio = (final[f'{unit}_log_prob'] - old).exp()
                diagnostics[f'{unit}_clip_fraction'] = float(((ratio - 1).abs() > self.eps).float().mean())
            if not all((math.isfinite(value) for value in diagnostics.values())):
                raise FloatingPointError('Non-finite PPO update diagnostics')
            self.last_update_diagnostics = diagnostics
            self.last_update_minibatch_diagnostics = minibatch_records
            self.last_update_epoch_diagnostics = epoch_records
        rates, means = ([], [])
        for field, capacity in (('ess_state', self.ess_capacity), ('ev_battery_state', self.ev_capacity)):
            if all((field in row.get('state', {}) for row in batch_data_list)):
                soc = torch.tensor([row['state'][field] / capacity for row in batch_data_list])
                violation = F.relu(self.constraint_config.get('soc_lower', 0.1) - soc) + F.relu(soc - self.constraint_config.get('soc_upper', 0.9))
                rates.append(float((violation > 0).float().mean().item()))
                means.append(float(violation.mean().item()))
            else:
                rates.append(0.0)
                means.append(0.0)
        return (sum(actor_losses) / len(actor_losses), sum(critic_losses) / len(critic_losses), torch.zeros((), device=self.device), sum(total_losses) / len(total_losses), 1.0, *rates, *means)

    def configuration(self):
        return {'variant': self.variant, 'state_dim': self.state_dim, 'base_state_dim': self.base_state_dim, 'state_stack_len': self.state_stack_len, 'hidden_dim': self.hidden_dim, 'prefix_rank': self.prefix_rank, 'use_shared_backbone': self.use_shared_backbone, 'prefix_dim': self.prefix_dim, 'device_order': list(self.device_order), 'deduplicated_prior': self.deduplicated_prior, 'initialize_actor_uniform': self.initialize_actor_uniform, 'evaluation_decoder': self.evaluation_decoder, 'gamma': self.gamma, 'lmbda': self.lmbda, 'eps': self.eps, 'epochs': self.epochs, 'minibatch_size': self.minibatch_size, 'target_kl': self.target_kl, 'ent_coef': self.ent_coef, 'max_grad_norm': self.max_grad_norm, 'critic_target': 'unnormalized_gae_plus_old_value', 'semantics_version': SEMANTICS_VERSION, 'action_space_config': copy.deepcopy(self.action_space)}
