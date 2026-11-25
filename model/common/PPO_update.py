"""PPO update."""
import copy
import math
import torch
from .PPO_core import ConditionalExecutionPPO, saved_probabilities

class FairConditionalPPO(ConditionalExecutionPPO):

    def __init__(self, *args, refinement_start_episode=6000, selective_policy_mode='protected', **kwargs):
        super().__init__(*args, **kwargs)
        if selective_policy_mode not in ('protected', 'unprotected', 'unguarded'):
            raise ValueError('Invalid selective policy mode')
        self.refinement_start_episode = int(refinement_start_episode)
        self.selective_policy_mode = selective_policy_mode
        self._phase_two = False

    def enter_refinement(self):
        if self._phase_two:
            return
        self._phase_two = True
        if self.selective_policy_mode != 'unprotected':
            modules = [self.shared_backbone]
            for name in ('ewh_set_temp', 'wash_machine_schedule'):
                if name in self.actor_branches:
                    modules.append(self.actor_branches[name])
                if name in self.prefix_adapters:
                    modules.append(self.prefix_adapters[name])
            for module in modules:
                module.requires_grad_(False)
        for group in self.actor_optim.param_groups:
            group['lr'] = min(group['lr'], 3e-05)
        for group in self.critic_optim.param_groups:
            group['lr'] = min(group['lr'], 0.0001)

    @torch.no_grad()
    def replay_conditional_kl(self, rows):
        states = torch.cat([r['stacked_state'] for r in rows]).to(self.device)
        result = self.evaluate_actions(states, {n: [r['sampled_action_indices'][n] for r in rows] for n in self.action_dims}, {n: [r['action_masks'][n] for r in rows] for n in self.action_dims}, [r['execution_classes'] for r in rows], [r['prefix_features'] for r in rows], [r['selected_class_ids'] for r in rows], return_details=True)
        kind = 'command' if self.variant == 'command_ar' else 'class'
        total = 0.0
        for name in self.device_order:
            logs = result[f'branch_{kind}_logs'][name]
            probs = saved_probabilities(rows, f'old_branch_{kind}_probabilities', name, logs)
            safe = logs.masked_fill(probs == 0, 0)
            value = float((probs * (probs.clamp_min(1e-30).log() - safe)).sum(-1).mean())
            if not math.isfinite(value) or value < -1e-05:
                raise FloatingPointError('Invalid policy KL')
            total += max(0.0, value)
        return total

    def update(self, rows, running_stats=None, state_keys=None, episode=0, total_episodes=1):
        if episode >= self.refinement_start_episode:
            self.enter_refinement()
        guarded = self._phase_two and self.selective_policy_mode != 'unguarded'
        parameters = list(self.actor_parameters())
        before = [p.detach().clone() for p in parameters] if guarded else None
        optimizer_before = copy.deepcopy(self.actor_optim.state_dict()) if guarded else None
        losses = super().update(rows, running_stats, state_keys, episode, total_episodes)
        attempted = self.replay_conditional_kl(rows) if self._phase_two else 0.0
        accepted = not guarded or attempted <= 0.02
        if not accepted:
            with torch.no_grad():
                for parameter, old in zip(parameters, before):
                    parameter.copy_(old)
            self.actor_optim.load_state_dict(optimizer_before)
            for group in self.actor_optim.param_groups:
                group['lr'] = max(1e-06, group['lr'] * 0.5)
        effective = attempted if accepted else self.replay_conditional_kl(rows)
        self.last_update_diagnostics.update(refinement_phase=int(self._phase_two), guarded_kl_attempted=attempted, guarded_kl_effective=effective, guarded_actor_accepted=int(accepted), actor_lr_next=self.actor_optim.param_groups[0]['lr'])
        return losses
