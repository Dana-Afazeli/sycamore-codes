from tqdm import tqdm

def bandit_loop(agent, env, config):
    log = {
        'awake_experts': [],
        'actions': [],
        'regrets': []
    }

    for t in tqdm(range(config['timesteps'])):
        awake_experts = env.step_awake_experts()
        action = agent.step(awake_experts)
        signals = env.step_signal()
        regrets = agent.update(signals)

        log['awake_experts'].append(awake_experts)
        log['actions'].append(action)
        log['regrets'].append(regrets)

    return log
