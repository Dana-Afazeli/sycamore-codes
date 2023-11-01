from abc import ABC, abstractmethod

class BanditAgentBase(ABC):
    
    def __init__(self, n_actions, is_full_info=True):
        self.n_actions = n_actions
        self.is_full_info = is_full_info

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def update_rewards(self, rewards):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass