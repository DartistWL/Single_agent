"""continuous adapter."""
import importlib
import random
from collections import deque
import numpy as np
import torch
ORDER = ('ev_power', 'battery_power', 'wash_machine_schedule', 'Air_conditioner_set_temp', 'Air_conditioner_set_temp2', 'ewh_set_temp')

class ContinuousDevicePolicy:

    def __init__(self, name, env, device):
        if name not in ('crossq', 'droq'):
            raise ValueError(name)
        self.name, self.device, self.env = (name, device, env)
        self.state_stack_len = 1
        self.use_state_normalization = True
        self.memory = deque(maxlen=100000)
        module = importlib.import_module('model.paper.' + ('CrossQ' if name == 'crossq' else 'DroQ'))
        module.device = device
        cfg = (module.CrossQConfig if name == 'crossq' else module.DroQConfig)()
        self.agent = (module.CrossQAgent if name == 'crossq' else module.DroQAgent)(len(env.state_space), 6, cfg, dev=device, use_state_normalization=False)
        self.cfg = cfg

    def critic(self, x):
        return x.new_zeros(1)

    def choose(self, env, state, tensor, deterministic):
        mask, _ = env.get_action_mask(state, mask_mode='physical')
        zero = tensor.new_zeros(1)
        x = tensor.detach().cpu().numpy().reshape(-1)
        _, latent = self.agent.select_action(x, deterministic=deterministic)
        actions = {}
        for v, n in zip(latent, ORDER):
            options = env.action_space[n]
            idx = (float(v) + 1.0) * 0.5 * (len(options) - 1)
            legal = [i for i, valid in enumerate(mask[n]) if valid]
            actions[n] = options[min(legal, key=lambda i: abs(i - idx))]
        return (actions, zero, 0.0, dict(latent=np.asarray(latent, dtype=np.float32)))

    def update(self, batch, stats, keys, index, blocks):
        states = torch.cat([row['stacked_state'] for row in batch] + [batch[-1]['stacked_next_state']]).detach().cpu()
        for row, s, n in zip(batch, states[:-1], states[1:]):
            r, d = (row['rewards'], float(row['dones']))
            a = row['latent']
            self.memory.append((s.numpy(), a, n.numpy(), r, d))
            cfg = self.cfg
            if len(self.memory) < cfg.min_buffer:
                continue
            for _ in range(cfg.utd_ratio):
                S, A, N, R, D = [np.asarray(v, dtype=np.float32) for v in zip(*random.sample(self.memory, cfg.batch_size))]
                self.agent.update((S, A, N, R * cfg.reward_scale, D), None)
