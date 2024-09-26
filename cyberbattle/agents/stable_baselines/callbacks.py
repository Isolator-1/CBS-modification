import copy
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


# Callback logging some training statistics additionals to the one provided by stable-baselines3
class TrainingCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.env = env
        self.success_actions = []
        self.action_counts = []
        self.reset = True


    def _on_training_start(self) -> None:
        self.number_actions = self.model.action_space.n
        self.action_counts = np.zeros(self.model.action_space.n)
        self.success_actions = np.zeros(self.model.action_space.n)

    def _on_step(self) -> bool:
        actions = self.locals["actions"]
        rewards = self.locals["rewards"]
        done = self.locals["dones"][0]
        info = self.locals["infos"][0]

        if self.reset:
            pass

        self.reset = False

        for i in range(len(actions)):
            self.action_counts[actions[i]] += 1
            if rewards[i] > 0:
                self.success_actions[actions[i]] += 1
        # Log additional information related to the current training environment when episode finished
        if done:
            # TODO: 0 to fix in case of multiple envs stacked in DummyVecEnv in parallel
            owned_nodes, discovered_nodes, num_nodes, not_discovered_nodes, percentage_discovered_credentials = self.env.envs[0].get_statistics()
            self.logger.record("train/Owned nodes", owned_nodes)
            self.logger.record("train/Discovered nodes", discovered_nodes)
            self.logger.record("train/Not discovered nodes", not_discovered_nodes)
            owned_percentage = owned_nodes / num_nodes
            discovered_percentage = discovered_nodes / num_nodes
            self.logger.record("train/Owned nodes percentage", owned_percentage)
            self.logger.record("train/Discovered nodes percentage", discovered_percentage)
            self.logger.record("train/Discovered credentials percentage", percentage_discovered_credentials)
            # the minimum between the ratio and 1 will be used: credentials may be duplicated in the topology based on a probability
            # hence the ratio may be greater than 1 since the agent tries all credentials when accessing a port, we discard this additional number of nodes owned in this way
            if self.env.envs[0].current_env.env_type == "random_env":
                self.logger.record("train/Owned nodes percentage among reachable nodes",
                                   min(owned_nodes / (self.env.envs[0].current_env.reachable_count+1),1)) #adding ther starter node

                self.logger.record("train/Owned nodes in addition to reachable due to password reuse",
                                   max(owned_nodes - (self.env.envs[0].current_env.reachable_count+1), 0))

            self.logger.record("train/Owned-discovered ratio", owned_nodes/discovered_nodes)
            self.logger.record("train/Network availability", info['network_availability'])
            self.logger.record("train/Evicted", self.env.envs[0].evicted)
            self.logger.record("train/Number of reimaged nodes", len(self.env.envs[0].overall_reimaged))
            self.logger.record("train/Defender constraints broken", self.env.envs[0].defender_constraints_broken)
            self.reset=True
            overall_success = 0
            success = 0
            local_counts = 0
            remote_counts = 0
            connection_counts = 0
            movement_counts = 0
            for i in range(self.env.envs[0].local_attacks_count):
                success += self.success_actions[i]
                local_counts += self.action_counts[i]
            overall_success += success
            if local_counts != 0:
                ratio = success / local_counts
            else:
                ratio = 0
            self.logger.record(f"actions/train/Success rate for local attacks", ratio)
            success = 0
            for i in range(self.env.envs[0].local_attacks_count, self.env.envs[0].local_attacks_count+self.env.envs[0].remote_attacks_count):
                success += self.success_actions[i]
                remote_counts += self.action_counts[i]
            overall_success += success
            if remote_counts != 0:
                ratio = success / remote_counts
            else:
                ratio = 0
            self.logger.record(f"actions/train/Success rate for remote attacks", ratio)
            success = 0
            for i in range(self.env.envs[0].local_attacks_count+self.env.envs[0].remote_attacks_count, self.env.envs[0].local_attacks_count+self.env.envs[0].remote_attacks_count+self.env.envs[0].ports_count):
                success += self.success_actions[i]
                connection_counts += self.action_counts[i]
            overall_success += success
            if connection_counts != 0:
                ratio = success / connection_counts
            else:
                ratio = 0
            self.logger.record(f"actions/train/Success rate for port connections", ratio)
            self.logger.record(f"actions/train/Success rate for actions", overall_success / (local_counts + remote_counts + connection_counts))
            for i in range(self.env.envs[0].local_attacks_count + self.env.envs[0].remote_attacks_count + self.env.envs[0].ports_count,
                               self.env.envs[0].local_attacks_count + self.env.envs[0].remote_attacks_count + self.env.envs[0].ports_count + 4):
                movement_counts += self.action_counts[i]

            self.logger.record(f"actions/train/Rate of local attacks", local_counts / (local_counts + remote_counts + connection_counts + movement_counts))
            self.logger.record(f"actions/train/Rate of remote attacks",
                               remote_counts / (local_counts + remote_counts + connection_counts + movement_counts))
            self.logger.record(f"actions/train/Rate of port connections",
                               connection_counts / (local_counts + remote_counts + connection_counts + movement_counts))
            self.logger.record(f"actions/train/Rate of movements",
                                 movement_counts / (local_counts + remote_counts + connection_counts + movement_counts))

            self.action_counts = np.zeros(self.number_actions)
            self.success_actions = np.zeros(self.number_actions)
        return True  # Continue training


# Callback logging some validation statistics
class ValidationCallback(BaseCallback):
    def __init__(self, val_env, n_val_episodes, val_freq, log_path, early_stopping=False, patience=10, verbose=0):
        super(ValidationCallback, self).__init__(verbose)
        self.val_env = val_env
        self.n_val_episodes = n_val_episodes
        self.val_freq = val_freq  # in time steps
        self.log_path = log_path
        self.val_timesteps = 0
        self.early_stopping = bool(early_stopping)
        self.best_mean_reward = -np.inf
        self.patience = patience
        self.current_patience = 0

    def _on_step(self) -> bool:
        # Perform validation every val_freq timesteps
        if ((self.val_timesteps % self.val_freq) == 0):
            print("Validation phase")
            val_reward, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = self._evaluate_model()

            if val_reward > self.best_mean_reward:
                # save model checkpoint if overcomes best validation mean reward known
                self.model.save(os.path.join(self.log_path, f"checkpoint_{val_reward}_reward.zip"))
                print("New best validation reward found!")
                self.best_mean_reward = val_reward
                self.current_patience = 0
            else:
                # otherwise increase patience
                print("Increasing patience..")
                self.current_patience += 1
            if self.early_stopping:
                print("Checking early stopping condition..")
                # early stopping condition reached
                if self.current_patience >= self.patience:
                    print("Stopping training due to lack of improvement in validation reward.")
                    return False  # Stop training

        self.val_timesteps += 1
        return True  # Continue training

    def _evaluate_model(self):
        # Evaluate the model and log custom metrics after each validation episode
        custom_metrics = self._run_evaluation()
        self._log_custom_metrics(custom_metrics)
        return custom_metrics

    def _run_evaluation(self):
        owned_list = []
        discovered_list = []
        not_discovered_list = []
        owned_percentage_list = []
        discovered_percentage_list = []
        credentials_percentage_list = []
        episode_reward_list = []
        reachable_list = []

        local_ratios = []
        remote_ratios = []
        connection_ratios = []
        movement_ratios = []

        local_success = []
        remote_success = []
        connection_success = []
        overall_success_list = []
        number_steps_list = []
        evicted_list = []
        defender_constraints_broken_list = []

        for _ in range(self.n_val_episodes):
            obs = self.val_env.reset()
            episode_rewards = 0.0
            done = False

            self.action_counts = np.zeros(self.val_env.action_space.n)
            self.success_actions = np.zeros(self.val_env.action_space.n)
            number_steps = 0
            while not done:
                action_index, _ = self.model.predict(obs)
                original_index = copy.deepcopy(action_index) # may be modified by step
                obs, reward, done, _ = self.val_env.step(action_index)
                episode_rewards += reward
                self.action_counts[original_index] += 1
                number_steps += 1
                if reward > 0:
                    self.success_actions[original_index] += 1


            owned_nodes, discovered_nodes, num_nodes, not_discovered_nodes, percentage_discovered_credentials = self.val_env.envs[0].get_statistics()
            if getattr(self.val_env.envs[0], "env_type", None) is not None:
                if self.val_env.envs[0].env_type == "random_env":
                    reachable_list.append(self.val_env.envs[0].current_env.reachable_count+1) #adding ther starter node
            else:
                if self.val_env.envs[0].current_env.env_type == "random_env":
                    reachable_list.append(self.val_env.envs[0].current_env.reachable_count+1) #adding ther starter node
            owned_percentage = owned_nodes / num_nodes
            discovered_percentage = discovered_nodes / num_nodes

            owned_list.append(owned_nodes)
            discovered_list.append(discovered_nodes)
            not_discovered_list.append(not_discovered_nodes)
            episode_reward_list.append(episode_rewards)
            owned_percentage_list.append(owned_percentage)
            discovered_percentage_list.append(discovered_percentage)
            credentials_percentage_list.append(percentage_discovered_credentials)
            number_steps_list.append(number_steps)
            evicted_list.append(int(self.val_env.envs[0].evicted))
            defender_constraints_broken_list.append(int(self.val_env.envs[0].defender_constraints_broken))

            success = 0
            overall_success = 0

            local_counts = 0
            remote_counts = 0
            connection_counts = 0
            movement_counts = 0

            for i in range(self.val_env.envs[0].local_attacks_count):
                success += self.success_actions[i]
                local_counts += self.action_counts[i]
            overall_success += success

            if local_counts != 0:
                local_success.append(success/local_counts)
            else:
                local_success.append(0)

            success = 0
            for i in range(self.val_env.envs[0].local_attacks_count, self.val_env.envs[0].local_attacks_count + self.val_env.envs[0].remote_attacks_count):
                success += self.success_actions[i]
                remote_counts += self.action_counts[i]
            overall_success += success

            if remote_counts != 0:
                remote_success.append(success/remote_counts)
            else:
                remote_success.append(0)

            success = 0
            for i in range(self.val_env.envs[0].local_attacks_count + self.val_env.envs[0].remote_attacks_count,
                           self.val_env.envs[0].local_attacks_count + self.val_env.envs[0].remote_attacks_count + self.val_env.envs[0].ports_count):
                success += self.success_actions[i]
                connection_counts += self.action_counts[i]
            overall_success += success

            for i in range(self.val_env.envs[0].local_attacks_count + self.val_env.envs[0].remote_attacks_count + self.val_env.envs[0].ports_count,
                               self.val_env.envs[0].local_attacks_count + self.val_env.envs[0].remote_attacks_count + self.val_env.envs[0].ports_count + 4):
                movement_counts += self.action_counts[i]


            if connection_counts != 0:
                connection_success.append(success/connection_counts)
            else:
                connection_success.append(0)

            if local_counts + remote_counts + connection_counts != 0:
                overall_success_list.append(overall_success/ (local_counts + remote_counts + connection_counts))
            local_ratios.append(local_counts / (local_counts + remote_counts + connection_counts + movement_counts))
            remote_ratios.append(remote_counts / (local_counts + remote_counts + connection_counts + movement_counts))
            connection_ratios.append(connection_counts / (local_counts + remote_counts + connection_counts + movement_counts))
            movement_ratios.append(movement_counts / (local_counts + remote_counts + connection_counts + movement_counts))


        return np.mean(episode_reward_list), np.mean(owned_list), np.mean(discovered_list), np.mean(not_discovered_list), np.mean(owned_percentage_list), np.mean(discovered_percentage_list), np.mean(credentials_percentage_list), np.mean(local_success), np.mean(remote_success), np.mean(connection_success), np.mean(overall_success_list), np.mean(local_ratios), np.mean(remote_ratios), np.mean(connection_ratios), np.mean(movement_ratios), np.mean(number_steps_list), np.mean(evicted_list), np.mean(defender_constraints_broken_list), np.mean(reachable_list)

    def _log_custom_metrics(self, custom_metrics):
        info = self.locals["infos"][0]
        reward_mean, owned, discovered, not_discovered, owned_percentage, discovered_percentage, credentials_percentage, local_success, remote_success, connection_success, overall_success, local_ratio, remote_ratio, connection_ratio, movement_ratio, number_steps, evicted, defender_constraints_broken, reachable = custom_metrics
        self.logger.record("validation/Average reward", reward_mean)
        self.logger.record("validation/Average owned nodes", owned)
        self.logger.record("validation/Average discovered nodes", discovered)
        self.logger.record("validation/Average not discovered nodes", not_discovered)
        self.logger.record("validation/Average owned percentage", owned_percentage)
        self.logger.record("validation/Average discovered percentage", discovered_percentage)
        self.logger.record("validation/Average discovered credentials percentage", credentials_percentage)
        self.logger.record("validation/Owned-discovered ratio", owned / discovered)

        # the minimum between the ratio and 1 will be used: credentials may be duplicated in the topology based on a probability
        # hence the ratio may be greater than 1 since the agent tries all credentials when accessing a port, we discard this additional number of nodes owned in this way
        if self.val_env.envs[0].current_env.env_type == "random_env":
            self.logger.record("validation/Average owned percentage among reachable nodes",
                               min(owned / reachable,1))
            self.logger.record("validation/Owned nodes in addition to reachable due to password reuse",
                               max(owned - reachable, 0))

        self.logger.record("validation/Network availability", info['network_availability'])
        self.logger.record("validation/Number of steps", number_steps)
        self.logger.record("validation/Evicted", evicted)
        self.logger.record("validation/Defender constraints broken", defender_constraints_broken)
        self.logger.record("actions/validation/Success rate for local attacks", local_success)
        self.logger.record("actions/validation/Success rate for remote attacks", remote_success)
        self.logger.record("actions/validation/Success rate for port connections", connection_success)
        self.logger.record("actions/validation/Success rate for actions", overall_success)
        self.logger.record("actions/validation/Rate of local attacks", local_ratio)
        self.logger.record("actions/validation/Rate of remote attacks", remote_ratio)
        self.logger.record("actions/validation/Rate of port connections", connection_ratio)
        self.logger.record("actions/validation/Rate of movements", movement_ratio)
