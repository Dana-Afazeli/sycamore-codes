from abc import ABC, abstractmethod
import numpy as np

class BanditAgentBase(ABC):
    
    def __init__(self, n_actions, prior = None, signal_type='reward'):
        self.n_actions = n_actions

        if prior is None:
            self.prior = np.ones((self.n_actions,))
        else:
            self.prior = prior
            
        if signal_type in ['reward', 'loss']:
            self.signal_type = signal_type
        else:
            raise Exception('invalid signal type. either "reward" or "loss"')
       
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def update(self, signals):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass