import numpy as np

from .bandit_env_base import BanditEnvBase

class BanditRandomEnv(BanditEnvBase):
    def __init__(self, n_actions, random_seed=42):
        self.n_actions = n_actions
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)

    def _step_awake_experts(self):
        size = self.rng.choice(self.n_actions)
        awake_experts = self.rng.choice(self.n_actions, size=size, replace=False)
        self.last_awake_experts = awake_experts    
    
    def _step_signal(self):
        return self.rng.uniform(size=self.n_actions)
