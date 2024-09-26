import gymnasium
import numpy as np
import gymnasium.spaces as gymnasium_spaces
from cyberbattle._env.local.cyberbattle_env_utils import convert_to_gymnasium

# Wrapper that randomly switch the environment given a certain interval to ensure that the agent is robust to environment changes
class RandomSwitchEnv(gymnasium.Env):
    # Decide for a current environment and wraps it accordingly, then switch to a new one every switch_interval episodes
    def __init__(self, envs, switch_interval=50):
        self.envs = envs
        self.current_env = np.random.choice(envs)
        self.current_env_index = self.envs.index(self.current_env)
        self.episode_count = 0
        self.switch_interval = switch_interval
        self.steps_in_current_episode = 0
        self.current_observation = None
        self.local_attacks_count = self.current_env.get_local_attacks_count()
        self.remote_attacks_count = self.current_env.get_remote_attacks_count()
        self.ports_count = self.current_env.get_port_count()
        self.action_space = gymnasium_spaces.Discrete(self.current_env.action_space.n)
        self.observation_space = convert_to_gymnasium(self.current_env.observation_space)
        self.done = False
        self.num_envs = 1

    def render(self, mode='human'):
        return self.current_env.render(mode)


    def close(self):
        self.current_env.close()

    def sample_random_action(self):
        return self.current_env.sample_random_action()

    # Switch to a random new environment among the list
    def _switch_environment(self):
        self.current_env = np.random.choice(self.envs)
        self.current_env_index = self.envs.index(self.current_env)
        print("Switched to environment {}".format(self.current_env_index))
        self.episode_count = 0

    def _check_switch(self):
        if (self.episode_count + 1) % (self.switch_interval + 1) == 0:
            print("Switching environment...")
            self._switch_environment()

    def step(self, action, state=None):
        if state:
            # MultiInputDQN case
            observation, reward, self.done, info = self.current_env.step(action, state)
        else:
            observation, reward, self.done, info = self.current_env.step(action)
        self.truncated = self.current_env.truncated
        self.steps_in_current_episode += 1
        self.current_observation = observation
        if self.done:
            # memorize metrics in inner fields in order to be easily read
            self.owned_nodes, self.discovered_nodes, self.not_discovered_nodes, self.num_nodes, self.percentage_discovered_credentials = self.current_env.get_statistics()
            self.overall_reimaged = self.current_env.overall_reimaged
            self.defender_constraints_broken = self.current_env.defender_constraints_broken
            self.evicted = self.current_env.evicted
            self.episode_count += 1
            self._check_switch()
        return observation, reward, self.done, self.truncated, info

    def reset(self, seed=None):
        self.steps_in_current_episode = 0
        self.current_observation = self.current_env.reset()
        # determine whether a switch is necessary
        self._check_switch()
        self.done = False
        self.current_env.seed(seed)
        return self.current_observation, {}

    def seed(self, seed):
        for env in self.envs:
            env.seed(seed)

    # Provide statistics of the last iteration (called at the end of the episode)
    def get_statistics(self):
        return self.owned_nodes, self.discovered_nodes, self.num_nodes, self.not_discovered_nodes, self.percentage_discovered_credentials

    def set_cut_off(self, cut_off):
        self.current_env.set_cut_off(cut_off)
