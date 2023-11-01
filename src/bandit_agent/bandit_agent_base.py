from abc import ABC, abstractmethod

class BanditAgentBase(ABC):
    
    def __init__(n_actions, is_full_info=True):
        self.n_actions = n_actions
        self.is_full_info = is_full_info

    @abstractmethod
    def step():
        pass

    @abstractmethod
    def update_rewards(rewards):
        pass

    @abstractmethod
    def save(path):
        pass

    @abstractmethod
    def load(path):
        pass