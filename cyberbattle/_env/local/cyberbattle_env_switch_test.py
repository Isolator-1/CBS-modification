import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import pytest
from cyberbattle._env.local.cyberbattle_env_switch import RandomSwitchEnv
from cyberbattle.env_generation.chain_utils import generate_chains
import copy

@pytest.fixture
def envs():
    envs = generate_chains(100, [10, 15], 10)
    return envs

@pytest.mark.parametrize('num_episodes,num_iterations,switch_interval', [
    (100, 10, 10),
    (100, 10, 25),
    (500, 10, 50),
    (1000, 10, 100),
])
def test_random(envs, num_episodes, num_iterations, switch_interval):
    env = RandomSwitchEnv(envs, switch_interval)
    env_id = copy.deepcopy(env.current_env_index)
    num_switches = 0
    for i in range(num_episodes):
        env.reset()
        if (i+1) % switch_interval == 0:
            num_switches += (env.current_env_index != env_id)
            env_id = env.current_env_index
        for t in range(num_iterations):
            action = env.sample_random_action()
            observation, reward, done, truncated, info = env.step(action)
            if done:
                if env.episode_count == 0:
                    print(f"Environment switched at episode {i+1}")
                break
    env.close()
    # since it can be switched with replacement, assess that there has been switches at least the 50% of the time (heuristic)
    assert num_switches > num_episodes / switch_interval / 2
