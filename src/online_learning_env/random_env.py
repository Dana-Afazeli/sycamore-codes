import numpy as np

from .env_base import EnvBase

class RandomEnv(EnvBase):
    def __init__(self, n_actions, random_seed=42):
        self.n_actions = n_actions
        self.random_seed = random_seed

        self.reset()

    def step_awake_experts(self):
        size = self.rng.choice(np.arange(1,self.n_actions))
        awake_experts = np.sort(self.rng.choice(self.n_actions, size=size, replace=False))
        self.last_awake_experts = awake_experts  
        return awake_experts  

    def step_signal(self):
        return self.rng.uniform(size=self.n_actions)

    def reset(self):
        self.rng = np.ranodm.default_rng(self.random_seed)