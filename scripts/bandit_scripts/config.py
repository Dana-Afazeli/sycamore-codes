from src.bandit_agent import AdaptiveNormalHedge
from src.bandit_env import BanditRandomEnv

def config_generator():
    return {
        'n_actions': 10,
        'agent_class': AdaptiveNormalHedge,
        'env_class': BanditRandomEnv,
        'random_seed': 42,
        'timesteps': 1000,
    }