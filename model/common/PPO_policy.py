"""PPO policy."""
from .PPO_update import FairConditionalPPO

class HouseholdPPO(FairConditionalPPO):

    def take_action(self, state_tensor, action_mask=None, deterministic=False, return_info=False, *, raw_state, decoder=None, actions_only=False):
        self.priors = None
        result = super().take_action(state_tensor, action_mask, deterministic, True, raw_state=raw_state, decoder=decoder, actions_only=actions_only)
        if actions_only:
            return result
        result[-1]['prefix_features']['_household_prior'] = {}
        return result if return_info else result[:4]

    def evaluate_actions(self, state_tensor, sampled_action_indices, action_masks, execution_classes=None, prefix_features=None, selected_class_ids=None, return_details=False):
        self.priors = None
        return super().evaluate_actions(state_tensor, sampled_action_indices, action_masks, execution_classes, prefix_features, selected_class_ids, return_details)
