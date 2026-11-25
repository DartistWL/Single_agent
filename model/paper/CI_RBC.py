"""CI RBC algorithm."""
from typing import Any, Dict
import numpy as np
from other.environment.physics import HomeEnergyManagementEnv
from other.environment.config import CONFIG
from .TOU_RBC import RuleBasedController

class CarbonRule(RuleBasedController):

    def __init__(self, env, train_intensity):
        super().__init__(env)
        self.reference = train_intensity
        self.eur_per_kg = env.w_carbon * env.reward_scaler.scale('carbon', 1.0) / (env.w_cost * env.reward_scaler.scale('cost', 1.0))

    def take_action(self, state):
        adjusted = dict(state)
        adjusted['electricity_price'] += self.eur_per_kg * (self.env._carbon_intensity_kg_per_kwh_at_now() - self.reference)
        return super().take_action(adjusted)
