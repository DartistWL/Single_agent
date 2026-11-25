"""AR Class PPO algorithm."""
from model.common.PPO_policy import HouseholdPPO
from model.factory import ppo_parameters
POLICY_OPTIONS = {'variant': 'execution_class', 'use_state_normalization': True, 'use_advantage_normalization': True, 'use_dynamic_mask': True, 'use_shared_backbone': True, 'selective_policy_mode': 'unprotected'}

class AR_Class_PPO(HouseholdPPO):

    def __init__(self, env, device, blocks):
        super().__init__(**ppo_parameters(env, device, blocks, POLICY_OPTIONS))
        self.treatment = 'execution_class_unprotected'
        self.household = env
        self.priors = None

def build_agent(env, device, blocks):
    return AR_Class_PPO(env, device, blocks)
