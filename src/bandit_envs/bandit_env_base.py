from abc import ABC, abstractmethod

class BanditEnvBase(ABC):
    def __init__(self, n_actions):
        self.n_actions = n_actions