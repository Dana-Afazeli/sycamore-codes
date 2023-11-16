from abc import ABC, abstractmethod

class EnvBase(ABC):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    @abstractmethod
    def reset(self):
        pass