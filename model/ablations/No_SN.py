"""No SN ablation."""
from model.common.PPO_policy import HouseholdPPO
from model.factory import ppo_parameters
POLICY_OPTIONS = {'variant': 'independent', 'use_state_normalization': False, 'use_advantage_normalization': True, 'use_dynamic_mask': True, 'use_shared_backbone': True, 'selective_policy_mode': 'unprotected'}

class No_SN(HouseholdPPO):

    def __init__(self, env, device, blocks):
        super().__init__(**ppo_parameters(env, device, blocks, POLICY_OPTIONS))
        self.treatment = 'no_state_norm'
        self.household = env
        self.priors = None

def build_agent(env, device, blocks):
    return No_SN(env, device, blocks)
