"""Resolve checkpoint class names."""
import pickle
MODULE_MAP = {'model.household.agents': 'model.common.PPO_policy', 'model.household.networks': 'model.common.PPO_networks', 'model.household.reference.networks': 'model.common.Reference_PPO_networks', 'model.household.reference.ppo_base': 'model.paper.Reference_PPO', 'model.household.offpolicy': 'model.paper.Branching_Rainbow', 'model.household.legacy': 'model.common.continuous_adapter', 'model.crossq': 'model.paper.CrossQ', 'model.droq': 'model.paper.DroQ', 'model.household.conditional': 'model.common.PPO_core', 'model.household.fair': 'model.common.PPO_update'}

class Unpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if module == 'model.household.legacy' and name == 'Legacy':
            name = 'ContinuousDevicePolicy'
        return super().find_class(MODULE_MAP.get(module, module), name)
Pickler = pickle.Pickler
load = pickle.load
loads = pickle.loads
dump = pickle.dump
dumps = pickle.dumps
HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL
