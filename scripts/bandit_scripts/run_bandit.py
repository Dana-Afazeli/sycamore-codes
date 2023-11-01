import matplotlib.pyplot as plt
import numpy as np

from config import config_generator
from src.loops import bandit_loop

def get_agent(config):
    return config['agent_class'](
        config['n_actions'], 
        signal_type='reward', 
        random_seed=config['random_seed']
    )

def get_env(config):
    return config['env_class'](
        config['n_actions'],
        random_seed=config['random_seed']
    )

def get_best_picker(log):
    awake_to_total_regret = {}
    for i in range(len(log['awake_experts'])):
        awake_expert_str = np.array2string(log['awake_experts'][i])
        total_regret = awake_to_total_regret.get(awake_expert_str, np.zeros(log['regrets'][i].shape))
        awake_to_total_regret[awake_expert_str] = total_regret + log['regrets'][i]


    awake_to_best_action = {}
    for k, v in awake_to_total_regret.items():
        awake_to_best_action[k] = np.argmax(v)

    def phi(awake_experts):
        awake_experts_str = np.array2string(awake_experts)
        return awake_to_best_action[awake_experts_str]
    
    return phi

def calculate_sleeping_regret_array(log, best_picker):
    cumulative_sleeping_regret = 0
    sleeping_regret_array = []
    
    for i in range(len(log['awake_experts'])):
        cumulative_sleeping_regret += log['regrets'][i][best_picker(log['awake_experts'][i])]
        sleeping_regret_array.append(cumulative_sleeping_regret)
    
    return np.array(sleeping_regret_array)

def process_log(log):
    best_picker = get_best_picker(log)
    sleeping_regret_arr = calculate_sleeping_regret_array(log, best_picker)

    plt.plot(np.arange(sleeping_regret_arr.shape[0]), sleeping_regret_arr)
    plt.title('Sleeping Regret trend')
    plt.xlabel('timesteps')
    plt.ylabel('total sleeping regret')
    plt.show()

def run_bandit(config):
    agent = get_agent(config)
    env = get_env(config)
    log = bandit_loop(agent, env, config)
    process_log(log)

def main():
    run_bandit(config_generator())

if __name__ == "__main__":
    main()