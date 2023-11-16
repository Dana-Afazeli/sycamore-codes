import json
import numpy as np

from .agent_base import AgentBase


class AdaptiveNormalHedge(AgentBase):
    def __init__(self, n_actions, prior = None, signal_type='reward', random_seed=42):
        self.n_actions = n_actions
        if prior is None:
            self.prior = np.ones((self.n_actions,))
        else:
            self.prior = prior
            
        if signal_type in ['reward', 'loss']:
            self.signal_type = signal_type
        else:
            raise Exception('invalid signal type. either "reward" or "loss"')
        
        self.random_seed = random_seed

        self.reset()
        
    @staticmethod
    def _phi(R, C):
        return np.exp(
            (np.max(R, 0) * np.max(R, 0)) / 
            (3 * C)
        )
    
    @staticmethod
    def _mask_weights(weights, awake_experts):
        mask_array = np.zeros(weights.shape)
        mask_array[awake_experts] = 1
        return weights * mask_array

    def _calculate_weights(self):
        weights = 0.5 * (
            AdaptiveNormalHedge._phi(self.R + 1, self.C + 1) -
            AdaptiveNormalHedge._phi(self.R - 1, self.C + 1)
        )

        if np.array_equal(weights, np.zeros(weights.shape)):
            return np.ones(weights.shape)
        
        return weights

    def _calculate_probabilities(self, awake_experts):
        weights = self._calculate_weights()
        masked_weights = AdaptiveNormalHedge._mask_weights(weights, awake_experts)
        proportions = masked_weights * self.prior
        
        return proportions / np.sum(proportions)

    def step(self, awake_experts=None):
        if awake_experts is None:
            awake_experts = np.arange(self.n_actions)

        probabilities = self._calculate_probabilities(awake_experts)
        self.last_probabilities = probabilities
        self.last_awake_experts = awake_experts

        return self.rng.choice(self.n_actions, p=probabilities)
    
    def _calculate_regrets(self, signals):
        signals_arr = np.array(signals)
        signals_average = np.dot(signals_arr, self.last_probabilities)
        signal_gain = signals_arr - signals_average
        if self.signal_type == 'reward':
            return signal_gain
        elif self.signal_type == 'loss':
            return -signal_gain
        else:
            raise Exception('invalid signal type. either "reward" or "loss"')

    def _update_regrets(self, regrets):
        self.R += regrets
        self.C += np.abs(regrets)

    def update(self, signals):
        regrets = self._calculate_regrets(signals)
        self._update_regrets(regrets)
        return regrets

    def save(self, path):
        params = {}
        params['n_actions'] = self.n_actions
        params['prior'] = self.prior.tolist()
        params['random_seed'] = self.random_seed
        params['signal_type'] = self.signal_type
        params['R'] = self.R.tolist()
        params['C'] = self.C.tolist()
        params['last_probabilities'] = self.last_probabilities.tolist()
        params['last_awake_experts'] = self.last_awake_experts.tolist()
        json.dump(params, open(path, 'w'))

    def load(self, path):
        params = json.load(open(path, 'r'))
        self.n_actions = params['n_actions']
        self.prior = np.array(params['prior'])
        self.random_seed = params['random_seed']
        self.signal_type = params['signal_type']
        self.R = np.array(params['R'])
        self.C = np.array(params['C'])
        self.last_probabilities = np.array(params['last_probabilities'])
        self.last_awake_experts = np.array(params['last_awake_experts'])

    def reset(self):
        self.rng = np.random.default_rng(self.random_seed)

        self.last_probabilities = None
        self.last_awake_experts = None

        self.R = np.zeros((self.n_actions,))
        self.C = np.zeros((self.n_actions,))
