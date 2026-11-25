"""No Share ablation."""
from model.common.PPO_policy import HouseholdPPO
from model.factory import ppo_parameters
POLICY_OPTIONS = {'variant': 'independent', 'use_state_normalization': True, 'use_advantage_normalization': True, 'use_dynamic_mask': True, 'use_shared_backbone': False, 'selective_policy_mode': 'unprotected'}

class No_Share(HouseholdPPO):

    def __init__(self, env, device, blocks):
        super().__init__(**ppo_parameters(env, device, blocks, POLICY_OPTIONS))
        self.treatment = 'no_sharing'
        self.household = env
        self.priors = None

def build_agent(env, device, blocks):
    return No_Share(env, device, blocks)
