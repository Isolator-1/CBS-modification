import copy
import torch
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import random
import re
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pickle
from stable_baselines3 import DQN, PPO
from sb3_contrib import RecurrentPPO, QRDQN, TRPO
from stable_baselines3.a2c import A2C
import matplotlib
import pandas as pd
matplotlib.use('TkAgg')
import csv
import datetime
from cyberbattle.simulation.model import RulePermission
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from cyberbattle.agents.train_utils import get_box_variables
from cyberbattle._env.local.cyberbattle_env_switch import RandomSwitchEnv


def bootstrap_ci(data, confidence=0.95, n_iterations=10000):
    rng = np.random.default_rng()
    bootstrap_means = np.array([np.mean(rng.choice(data, replace=True, size=len(data))) for _ in range(n_iterations)])
    lower_bound = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    upper_bound = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)
    original_mean = np.mean(data)
    return original_mean, lower_bound, upper_bound

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model

def predict_action(model, state, gym_env, random_percentage):
    if random_percentage > 0 and random.random() < random_percentage:
        action = gym_env.action_space.sample()
    else:
        action, _ = model.predict(state)
    return action

# Run episodes and save action choices statistics
def run_and_save_actions(model, gym_env, num_episodes=500, num_iterations=100, random_percentage=0):
    actions_chosen = []
    for _ in range(num_episodes):
        state = gym_env.reset()
        for t in range(num_iterations):
            with torch.no_grad():
                action = predict_action(model, state, gym_env, random_percentage)
            actions_chosen.append(int(action))
            next_state, reward, done, _ = gym_env.step(action)
            state = next_state
            if done:
                break
    return actions_chosen

# Line plot for an agent compared to a random agent for the owned, discovered, and credentials metrics
def plot_live_performance(performance_data, agent_color='blue', random_agent_color='black', checkpoint_name="No Info"):
    if not hasattr(plot_live_performance, 'fig'):
        plot_live_performance.fig, axs = plt.subplots(3, 1, figsize=(10, 6))
        plot_live_performance.axs = axs
        plot_live_performance.lines = {
            'agent_owned': axs[0].plot([], [], color=agent_color, label='Agent', alpha=0.7)[0],
            'random_agent_owned': axs[0].plot([], [], color=random_agent_color, label='Random Agent', alpha=0.7)[0],
            'agent_discovered': axs[1].plot([], [], color=agent_color, label='Agent', alpha=0.7)[0],
            'random_agent_discovered': axs[1].plot([], [], color=random_agent_color, label='Random Agent',alpha=0.7)[0],
            'agent_credentials': axs[2].plot([], [], color=agent_color, label='Agent',alpha=0.7)[0],
            'random_agent_credentials': axs[2].plot([], [], color=random_agent_color, label='Random Agent',alpha=0.7)[0],
        }
        for ax in axs:
            ax.legend()
    plot_live_performance.fig.suptitle(f'Average performances - Checkpoint {checkpoint_name}')

    episode_list = performance_data['episode']
    for metric, line in plot_live_performance.lines.items():
        line.set_xdata(episode_list)
        line.set_ydata(performance_data[f'{metric}'])

    for ax, metric in zip(plot_live_performance.axs, ['Average Owned Percentage', 'Average Discovered Percentage', 'Average Number Credentials']):
        ax.set_title(f'{metric}')
        if metric.endswith("Percentage"):
            ax.set_ylim(bottom=0, top=1.05)

    plt.subplots_adjust(hspace=0.5)
    for ax in plot_live_performance.axs:
        ax.relim()
        ax.autoscale_view()
    plt.draw()
    return plt

def plot_checkpoints_live_performance(performance_data, owned_color='blue', discovered_color='black', credentials_color='green'):
    if not hasattr(plot_checkpoints_live_performance, 'fig'):
        plot_checkpoints_live_performance.fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        plot_checkpoints_live_performance.ax = ax
        plot_checkpoints_live_performance.lines = {
            'average_owned': ax.plot([], [], color=owned_color, label='Mean owned percentage', alpha=0.7)[0],
            'average_discovered': ax.plot([], [], color=discovered_color, label='Mean discovered percentage', alpha=0.7)[0],
            'average_credentials': ax.plot([], [], color=credentials_color, label='Mean credentials percentage',alpha=0.7)[0]
        }
        ax.legend()

    checkpoints_list = performance_data['checkpoint']
    for metric, line in plot_checkpoints_live_performance.lines.items():
        print(metric)
        line.set_xdata(checkpoints_list)
        line.set_ydata(performance_data[f'{metric}'])

    plot_checkpoints_live_performance.ax.set_title("Checkpoint metrics")
    plot_checkpoints_live_performance.ax.set_ylim(bottom=0, top=1.05)

    plt.subplots_adjust(hspace=0.5)
    plot_checkpoints_live_performance.ax.relim()
    plot_checkpoints_live_performance.ax.autoscale_view()
    plt.draw()
    plt.pause(0.5)
    return plt

# Average performances calculation
def calculate_average_performance(model, gym_env, num_episodes=500, num_iterations=1000, avoid_random=False, checkpoint_name = "No Info", random_percentage=0.0):
    episode_list = []
    agent_owned_list = []
    agent_discovered_list = []
    agent_credentials_list = []
    random_agent_owned_list = []
    random_agent_discovered_list = []
    random_agent_credentials_list = []

    stats_data = []

    for episode in range(num_episodes):
        # Set the cut off for the environment
        gym_env.envs[0].set_cut_off(num_iterations)

        # Play random actions to calculate random agent performance
        if not avoid_random:
            random_agent_owned, random_agent_percentage_discovered, random_agent_credentials = play_random_agent_episode(gym_env)

            stats_data.append({
                'episode': episode,
                'agent': 'random',
                'owned_nodes': random_agent_owned,
                'discovered_nodes': random_agent_percentage_discovered,
                'num_discovered_credentials': random_agent_credentials
            })

        state = gym_env.reset()
        while True:
            with torch.no_grad():
                action = predict_action(model, state, gym_env, random_percentage)

            next_state, reward, done, _ = gym_env.step(action)
            state = next_state
            if done:
                break

        owned_nodes, discovered_nodes, num_nodes, _, percentage_discovered_credentials = gym_env.envs[0].get_statistics()
        # Append data for live plotting
        episode_list.append(episode)

        min(owned_nodes / (gym_env.envs[0].current_env.reachable_count + 1), 1)

        agent_owned_list.append(min(owned_nodes / (gym_env.envs[0].current_env.reachable_count + 1), 1))
        agent_discovered_list.append((discovered_nodes / num_nodes))
        agent_credentials_list.append(percentage_discovered_credentials)
        if not avoid_random:
            random_agent_owned_list.append(min(random_agent_owned / (gym_env.envs[0].current_env.reachable_count + 1), 1))
            random_agent_discovered_list.append(random_agent_percentage_discovered)
            random_agent_credentials_list.append(random_agent_credentials)

        stats_data.append({
            'episode': episode,
            'agent': 'agent',
            'owned_nodes': owned_nodes / num_nodes,
            'discovered_nodes': discovered_nodes / num_nodes,
            'num_discovered_credentials': percentage_discovered_credentials
        })

        if not avoid_random:
            # Periodically plot live performance
            performance_data = {
                'episode': episode_list,
                'agent_owned': agent_owned_list,
                'agent_discovered': agent_discovered_list,
                'agent_credentials': agent_credentials_list,
                'random_agent_owned': random_agent_owned_list,
                'random_agent_discovered': random_agent_discovered_list,
                'random_agent_credentials': random_agent_credentials_list,
            }
            plt = plot_live_performance(performance_data, checkpoint_name=checkpoint_name)
    if not avoid_random:
        plt.clf()
    df = pd.DataFrame(stats_data, columns=['episode', 'agent', 'owned_nodes', 'discovered_nodes', 'num_discovered_credentials'])
    return df, plt, agent_owned_list, agent_discovered_list, agent_credentials_list, random_agent_owned_list, random_agent_discovered_list, random_agent_credentials_list

# One episode of a random agent
def play_random_agent_episode(env):
    env.reset()
    while(True):
        action = env.action_space.sample()
        next_state, _, done, _ = env.step([action])
        if done:
            break
    owned_nodes, discovered_nodes, num_nodes, _, percentage_discovered_credentials = env.envs[0].get_statistics()
    return owned_nodes, discovered_nodes / num_nodes, percentage_discovered_credentials

# Bar plot of the count of actions chosen separated by type
def plot_action_distribution(config, actions_chosen, checkpoint_name):
    local_vulnerabilities = config['local_vulnerabilities']
    remote_vulnerabilities = config['remote_vulnerabilities']
    ports = config['ports']
    print(local_vulnerabilities, remote_vulnerabilities, ports, ['SN Backward', 'SN Forward', 'TN Backward', 'TN Forward'])
    all_actions = local_vulnerabilities + remote_vulnerabilities + ports + ["SN Backward", "SN Forward", "TN Backward", "TN Forward"]

    action_names = all_actions
    action_indices = np.arange(len(action_names))

    # different color according to the type of action
    local_indices = np.arange(len(local_vulnerabilities))
    remote_indices = np.arange(len(local_vulnerabilities), len(local_vulnerabilities) + len(remote_vulnerabilities))
    connection_indices = np.arange(len(local_vulnerabilities) + len(remote_vulnerabilities), len(local_vulnerabilities) + len(remote_vulnerabilities) + len(ports))
    movements_indices = np.arange(len(local_vulnerabilities) + len(remote_vulnerabilities) + len(ports), len(all_actions))

    # Count the occurrences of each action
    action_counts = np.zeros(len(all_actions))
    for action in actions_chosen:
        action_counts[action] += 1
    print(action_counts)
    local_counts = action_counts[local_indices]
    remote_counts = action_counts[remote_indices]
    connection_counts = action_counts[connection_indices]
    movements_counts = action_counts[movements_indices]

    plt.bar(local_indices, local_counts, color='red', label='Local')
    plt.bar(remote_indices, remote_counts, color='blue', label='Remote')
    plt.bar(connection_indices, connection_counts, color='green', label='Connection')
    plt.bar(movements_indices, movements_counts, color='orange', label='Movement')

    plt.xticks(action_indices, action_names, rotation=45, ha='right')
    plt.legend(fontsize=14)
    plt.xlabel('Actions', fontsize=14)
    plt.ylabel('Number of Times Chosen', fontsize=14)
    plt.title(f'Action Distribution - Checkpoint {checkpoint_name}', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ion()
    plt.show()
    plt.ioff()
    return plt

def calculate_average_steps(model, gym_env, num_episodes=500, random_percentage=0):
    steps_data = {'Episode': [], 'Environment size': [], 'Optimal number of steps': [], 'Steps': []}

    for episode in range(num_episodes):
        state = gym_env.reset()
        steps_taken = 0
        while True:
            with torch.no_grad():
                action = predict_action(model, state, gym_env, random_percentage)
            next_state, _, done, _ = gym_env.step(action)
            state = next_state
            steps_taken += 1
            if done:
                break
        steps_data['Episode'].append(episode)
        if gym_env.envs[0].current_env.env_type == "chain_env":
            steps_data['Environment size'].append(gym_env.envs[0].current_env.size)
            steps_data['Optimal number of steps'].append(gym_env.envs[0].current_env.size-1)
        else:
            # TODO: To implement
            steps_data['Environment size'].append(None)
            steps_data['Optimal number of steps'].append(None)
        steps_data['Steps'].append(steps_taken)

    steps_df = pd.DataFrame(steps_data)

    print(steps_df)

    return steps_df

def calculate_trajectories(model, gym_env, num_episodes=100, num_iterations=500, random_percentage=0, run_id=None):
    clean_trajectories_data = {'Run': [], 'Environment': [], 'Episode': [], 'Iteration': [], 'Source node': [], 'Target node': [], 'Action': [], 'Reward': [], 'Owned': [], 'Discovered not owned': [], 'Discovered credentials': []} # 'Source local vulnerabilities available': [], 'Target remote vulnerabilities available': [], 'Target services accessible': [], 'Target services valid accessible': [], 'Source node firewall outgoing blocked': [], 'Target node firewall incoming blocked': []}

    for episode in range(num_episodes):
        state = gym_env.reset()
        gym_env.envs[0].set_cut_off(num_iterations)
        iteration = 0
        while True:
            iteration += 1
            with torch.no_grad():
                action = predict_action(model, state, gym_env, random_percentage)
            source_node = gym_env.envs[0].current_env.source_node_index
            target_node = gym_env.envs[0].current_env.target_node_index
            #source_node_vuln_used = gym_env.envs[0].current_env.get_actuator().get_vulnerabilities_used(gym_env.envs[0].current_env.source_node_index)
            #target_node_vuln_used = gym_env.envs[0].current_env.get_actuator().get_vulnerabilities_used(gym_env.envs[0].current_env.target_node_index)
            owned_nodes = copy.deepcopy(gym_env.envs[0].current_env.get_owned_nodes())
            discovered_not_owned = copy.deepcopy(gym_env.envs[0].current_env.get_discovered_not_owned_nodes())
            accessible_services = []
            accessible_services_valid = []
            discovered_credentials = copy.deepcopy(gym_env.envs[0].current_env.get_discovered_credentials())
            for service in gym_env.envs[0].current_env.get_node(target_node).services:
                if gym_env.envs[0].current_env.is_service_accessible(service, gym_env.envs[0].current_env.target_node_index):
                    accessible_services.append(str(service.name))
                    if gym_env.envs[0].current_env.is_service_accessible_by_valid_credentials(service, gym_env.envs[0].current_env.target_node_index):
                        accessible_services_valid.append(str(service.name))

            next_state, reward, done, _ = gym_env.step(action)

            clean_trajectories_data['Run'].append(run_id)
            clean_trajectories_data['Environment'].append(gym_env.envs[0].current_env_index)
            clean_trajectories_data['Episode'].append(episode)
            clean_trajectories_data['Iteration'].append(iteration)
            clean_trajectories_data['Source node'].append(source_node)

            #local_vulns = [vulnerability for vulnerability in gym_env.envs[0].current_env.get_node(source_node).vulnerabilities if vulnerability in gym_env.envs[0].current_env.identifiers.local_vulnerabilities and gym_env.envs[0].current_env.get_vulnerability_index(vulnerability) not in source_node_vuln_used]

            #clean_trajectories_data['Source local vulnerabilities available'].append(local_vulns)
            clean_trajectories_data['Target node'].append(target_node)
            #remote_vulns = [vulnerability for vulnerability in gym_env.envs[0].current_env.get_node(target_node).vulnerabilities if vulnerability in gym_env.envs[0].current_env.identifiers.remote_vulnerabilities and gym_env.envs[0].current_env.get_vulnerability_index(vulnerability) not in target_node_vuln_used]

            #clean_trajectories_data['Target remote vulnerabilities available'].append(remote_vulns)

            #clean_trajectories_data['Target services accessible'].append(accessible_services)
            #clean_trajectories_data['Target services valid accessible'].append(accessible_services_valid)

            #clean_trajectories_data['Source node firewall outgoing blocked'].append([rule.port for rule in gym_env.envs[0].current_env.get_node(source_node).firewall.outgoing if rule.permission == RulePermission.BLOCK])
            #clean_trajectories_data['Target node firewall incoming blocked'].append([rule.port for rule in gym_env.envs[0].current_env.get_node(target_node).firewall.incoming if rule.permission == RulePermission.BLOCK])
            clean_trajectories_data['Action'].append(gym_env.envs[0].current_env.get_action_name(action[0]))
            clean_trajectories_data['Reward'].append(reward)
            clean_trajectories_data['Owned'].append(owned_nodes)
            clean_trajectories_data['Discovered not owned'].append(discovered_not_owned)
            clean_trajectories_data['Discovered credentials'].append(discovered_credentials)

            state = next_state

            if done:
                break

    clean_trajectories_df = pd.DataFrame(clean_trajectories_data)

    return clean_trajectories_df

algorithm_dict = {
    'dqn': DQN,
    'dqn_random': DQN,
    'ppo': PPO,
    'a2c': A2C,
    'recurrent_ppo': RecurrentPPO,
    'qr_dqn': QRDQN,
    'qr_dqn_random': QRDQN,
    'trpo': TRPO,
}

def load_attacks(env_sample, config):
    try:
        local_vulnerabilities = env_sample.identifiers.local_vulnerabilities
        remote_vulnerabilities = env_sample.identifiers.remote_vulnerabilities
        ports = env_sample.identifiers.ports
    except:
        local_vulnerabilities = env_sample.get_local_attacks()
        remote_vulnerabilities = env_sample.get_remote_attacks()
        ports = env_sample.get_ports()

    config['local_vulnerabilities'] = local_vulnerabilities
    config['remote_vulnerabilities'] = remote_vulnerabilities
    config['ports'] = ports
    return config


def parse_option(config, norm_obs):
    # Load test environments from specific run folder
    if config['load_test_envs']:
        if config['test_folder']:
            test_envs_path = os.path.join('logs', config['test_folder'], str(config['run_id']), 'test_envs.pkl')
        else:
            test_envs_path = os.path.join('logs', config['run_folder'], str(config['run_id']), 'test_envs.pkl')
        with open(test_envs_path, 'rb') as test_file:
            envs = pickle.load(test_file)
        print(f"Test environments loaded from {test_envs_path}")
        envs = RandomSwitchEnv(envs, config['switch_interval'])
        envs = DummyVecEnv([lambda: Monitor(envs)])
        norm_obs_keys = get_box_variables(envs.observation_space)
        if norm_obs:
            print("-> Normalization: True")
        envs = VecNormalize(envs, norm_obs=norm_obs, norm_obs_keys=norm_obs_keys,
                                norm_reward=False)

    # load attacks information, useful for some options
    config = load_attacks(envs.envs[0].current_env, config)

    # Determine where the checkpoint should be a training or a validation one
    if config['val_checkpoints']:
        print("--- Checkpoint folder: Validation")
        stats_folder = "stats/validation"
        checkpoints_folder = "validation"
    else:
        print("--- Checkpoint folder: Training")
        stats_folder = "stats/train"
        checkpoints_folder = "checkpoints"

    if not os.path.exists(os.path.join('logs', config['run_folder'], stats_folder, str(config['run_id']))):
        os.makedirs(os.path.join('logs', config['run_folder'], stats_folder, str(config['run_id'])))

    # Load checkpoints
    checkpoint_files = [file for file in os.listdir(
        os.path.join('logs', config['run_folder'], checkpoints_folder, str(config['run_id']))) if
                        file.startswith("checkpoint_")]

    episode_pattern = re.compile(r'checkpoint_(-?\d+)')

    checkpoint_files.sort(key=lambda x: int(episode_pattern.search(x).group(1)))
    checkpoints = []

    # Case of a single checkpoint: the last one
    if config['last_checkpoint']:
        # last checkpoint only
        if len(checkpoint_files) > 0:
            print("--- Checkpoint set: Last checkpoint")
            checkpoints.append(os.path.join('logs', config['run_folder'], checkpoints_folder, str(config['run_id']),
                                            checkpoint_files[-1]))
    else:
        # use all checkpoints ordered by episode
        print("--- Checkpoint set: All checkpoints")
        for checkpoint_file in checkpoint_files:
            checkpoints.append(
                os.path.join('logs', config['run_folder'], checkpoints_folder, str(config['run_id']), checkpoint_file))

    if len(checkpoints) == 0:
        print("ERROR: No checkpoints to load in the folder...")

    outcomes = {} # to be averaged with the other runs

    for checkpoint_index in range(len(checkpoints)):
        checkpoint_path = checkpoints[checkpoint_index]
        if "steps" in checkpoint_path:
            checkpoint_id = \
                checkpoint_path.split(checkpoints_folder + "/" + str(config['run_id']) + "/checkpoint_")[1].split(
                    "_steps")[0]
        elif "reward" in checkpoint_path:
            checkpoint_id = \
                checkpoint_path.split(checkpoints_folder + "/" + str(config['run_id']) + "/checkpoint_")[
                    1].split("_reward")[0]
        checkpoint_short_name = \
            checkpoint_path.split(checkpoints_folder + "/" + str(config['run_id']) + "/")[1].split('.pt')[0].split(
                '.zip')[0]

        model = algorithm_dict[config['algorithm']].load(checkpoint_path)

        print("--- Checkpoint:", checkpoint_short_name)
        # Action distribution plot
        if config['option'] == "action_distribution":
            print("--- Option: Action distribution")
            actions_chosen = run_and_save_actions(model, envs, config['num_episodes_per_checkpoint'],
                                                  config['num_iterations'], config['random_percentage'])

            with open(os.path.join('logs', config['run_folder'], stats_folder, str(config['run_id']),
                                    f"action_choices_{checkpoint_short_name}.csv"), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(actions_chosen)
            plt = plot_action_distribution(config, actions_chosen, checkpoint_id)
            fig_name = os.path.join('logs', config['run_folder'], stats_folder, str(config['run_id']),
                                    f"action_distribution_{checkpoint_short_name}.png")
            plt.tight_layout()
            plt.savefig(fig_name)
            plt.close()
            outcomes[checkpoint_short_name] = actions_chosen
        # Average performance calculation
        elif config['option'] == "average_performances":
            print("Calculating the average performance of the agent...")
            df, plt, agent_owned_list, agent_discovered_list, agent_credentials_list, random_agent_owned_list, random_agent_discovered_list, random_agent_credentials_list = calculate_average_performance(
                model, envs, config['num_episodes_per_checkpoint'], config['num_iterations'], config['no_random'], checkpoint_id, config['random_percentage'])
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            if config['test_folder']:
                save_folder = os.path.join('logs', config['run_folder'], stats_folder, str(config['run_id']), config['test_folder'])
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
            else:
                save_folder = os.path.join('logs', config['run_folder'], stats_folder, str(config['run_id']))

            df.to_csv(os.path.join(save_folder, f"average_performances_{checkpoint_short_name}_{config['num_iterations']}_{config['num_episodes_per_checkpoint']}_{current_time}.csv"),
                          index=False)

            outcomes[checkpoint_short_name] = {"Owned nodes among reachable": agent_owned_list,
                                               "Discovered nodes": agent_discovered_list,
                                               "Discovered credentials": agent_credentials_list,
                                               "Random - Owned nodes among reachable": random_agent_owned_list,
                                               "Random - Discovered nodes": random_agent_discovered_list,
                                               "Random - Discovered credentials": random_agent_credentials_list}
            print_save_performance_metrics(outcomes[checkpoint_short_name], checkpoint_short_name, save_folder, config['num_episodes_per_checkpoint'], config['num_iterations'])
        elif config['option'] == "number_steps":
            print("Calculating the average number of steps required...")
            logs_steps = calculate_average_steps(model, envs, config['num_episodes_per_checkpoint'], config['random_percentage'])
            file_name = os.path.join('logs', config['run_folder'], stats_folder, str(config['run_id']),
                                     f"number_steps_{checkpoint_short_name}.csv")
            logs_steps.to_csv(file_name, index=False)
            outcomes[checkpoint_short_name] = logs_steps
        elif config['option'] == "trajectories":
            print("Calculating the trajectories taken by the agent..")
            clean_logs_trajectories = calculate_trajectories(model, envs, config['num_episodes_per_checkpoint'], config['num_iterations'], config['random_percentage'], config['run_id'])
            file_name = os.path.join('logs', config['run_folder'], stats_folder, str(config['run_id']),
                                     f"clean_trajectories_{checkpoint_short_name}_{config['num_iterations']}_{config['num_episodes_per_checkpoint']}.csv")
            clean_logs_trajectories.to_csv(file_name, index=False)
            outcomes[checkpoint_short_name] = (clean_logs_trajectories)
    return outcomes

def print_save_performance_metrics(dict, checkpoint_short_name, logs_folder, num_episodes=100, num_iterations=500):
    average_owned_percentage, lower_bound_owned_percentage, upper_bound_owned_percentage = bootstrap_ci(dict['Owned nodes among reachable'])
    average_discovered_percentage, lower_bound_discovered_percentage, upper_bound_discovered_percentage = bootstrap_ci(dict['Discovered nodes'])
    average_number_credentials, lower_bound_number_credentials, upper_bound_number_credentials = bootstrap_ci(dict['Discovered credentials'])
    average_random_owned_percentage, lower_bound_random_owned_percentage, upper_bound_random_owned_percentage = bootstrap_ci(dict['Random - Owned nodes among reachable'])
    average_random_discovered_percentage, lower_bound_random_discovered_percentage, upper_bound_random_discovered_percentage = bootstrap_ci(dict['Random - Discovered nodes'])
    average_random_discovered_credentials, lower_bound_random_discovered_credentials, upper_bound_random_discovered_credentials = bootstrap_ci(dict['Random - Discovered credentials'])
    print("--------------------")
    print(f"Checkpoint {checkpoint_short_name}:")
    print(f"Average owned percentage among reachable: {average_owned_percentage} [{lower_bound_owned_percentage}, {upper_bound_owned_percentage}]")
    print(f"Average discovered percentage: {average_discovered_percentage} [{lower_bound_discovered_percentage}, {upper_bound_discovered_percentage}]")
    print(f"Average number of credentials: {average_number_credentials} [{lower_bound_number_credentials}, {upper_bound_number_credentials}]")
    print(f"Average random owned percentage: {average_random_owned_percentage} [{lower_bound_random_owned_percentage}, {upper_bound_random_owned_percentage}]")
    print(f"Average random discovered percentage: {average_random_discovered_percentage} [{lower_bound_random_discovered_percentage}, {upper_bound_random_discovered_percentage}]")
    print(f"Average random number of credentials: {average_random_discovered_credentials} [{lower_bound_random_discovered_credentials}, {upper_bound_random_discovered_credentials}]")
    print("--------------------")
    with open(os.path.join(logs_folder, f"average_performances_{checkpoint_short_name}_{num_iterations}_{num_episodes}.txt"), 'w') as file:
        file.write(f"Checkpoint {checkpoint_short_name}:\n")
        file.write(f"Average owned percentage: {average_owned_percentage} [{lower_bound_owned_percentage}, {upper_bound_owned_percentage}]\n")
        file.write(f"Average discovered percentage: {average_discovered_percentage} [{lower_bound_discovered_percentage}, {upper_bound_discovered_percentage}]\n")
        file.write(f"Average number of credentials: {average_number_credentials} [{lower_bound_number_credentials}, {upper_bound_number_credentials}]\n")
        file.write(f"Average random owned percentage: {average_random_owned_percentage} [{lower_bound_random_owned_percentage}, {upper_bound_random_owned_percentage}]\n")
        file.write(f"Average random discovered percentage: {average_random_discovered_percentage} [{lower_bound_random_discovered_percentage}, {upper_bound_random_discovered_percentage}]\n")
        file.write(f"Average random number of credentials: {average_random_discovered_credentials} [{lower_bound_random_discovered_credentials}, {upper_bound_random_discovered_credentials}]\n")

def merge_outcomes(run_outcomes, config):
    # Determine where the checkpoint should be a training or a validation one
    if config['val_checkpoints']:
        stats_folder = "stats/validation/merged"
    else:
        stats_folder = "stats/train/merged"

    if not os.path.exists(os.path.join('logs', config['run_folder'], stats_folder)):
        os.makedirs(os.path.join('logs', config['run_folder'], stats_folder))

    if config['option'] == "action_distribution":
        print("--- Merge: Action distribution")
        overall_action_choices = {}
        for run in run_outcomes:
            for checkpoint_short_name in run.keys():
                if checkpoint_short_name in overall_action_choices:
                    overall_action_choices[checkpoint_short_name].extend(run[checkpoint_short_name])
                else:
                    overall_action_choices[checkpoint_short_name] = run[checkpoint_short_name]

        for checkpoint_short_name in overall_action_choices.keys():
            plt = plot_action_distribution(config, overall_action_choices[checkpoint_short_name], checkpoint_short_name)
            fig_name = os.path.join('logs', config['run_folder'], stats_folder,
                                            f"action_distribution_{checkpoint_short_name}.png")
            plt.tight_layout()
            plt.savefig(fig_name)
            plt.close()
            with open(os.path.join('logs', config['run_folder'], stats_folder,
                                   f"action_choices_{checkpoint_short_name}.csv"), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(overall_action_choices[checkpoint_short_name])
    elif config['option'] == "average_performances":
        print("--- Merge: Average performances")
        overall_outcomes = {}
        overall_outcomes['Owned nodes among reachable'] = []
        overall_outcomes['Discovered nodes'] = []
        overall_outcomes['Discovered credentials'] = []
        overall_outcomes['Random - Owned nodes among reachable'] = []
        overall_outcomes['Random - Discovered nodes'] = []
        overall_outcomes['Random - Discovered credentials'] = []
        for run in run_outcomes:
            for checkpoint_short_name in run.keys():
                overall_outcomes['Owned nodes among reachable'].extend(run[checkpoint_short_name]['Owned nodes among reachable'])
                overall_outcomes['Discovered nodes'].extend(run[checkpoint_short_name]['Discovered nodes'])
                overall_outcomes['Discovered credentials'].extend(run[checkpoint_short_name]['Discovered credentials'])
                overall_outcomes['Random - Owned nodes among reachable'].extend(run[checkpoint_short_name]['Random - Owned nodes among reachable'])
                overall_outcomes['Random - Discovered nodes'].extend(run[checkpoint_short_name]['Random - Discovered nodes'])
                overall_outcomes['Random - Discovered credentials'].extend(run[checkpoint_short_name]['Random - Discovered credentials'])
        if config['test_folder']:
            save_folder = os.path.join('logs', config['run_folder'], stats_folder, config['test_folder'])
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
        else:
            save_folder = os.path.join('logs', config['run_folder'], stats_folder)
        print_save_performance_metrics(overall_outcomes, "merged", save_folder, config['num_episodes_per_checkpoint'], config['num_iterations'])
    elif config['option'] == "number_steps":
        print("Merged --- Average number of steps")
        steps_by_checkpoint = {}
        for run in run_outcomes:
            for checkpoint_short_name in run.keys():
                if checkpoint_short_name in steps_by_checkpoint:
                    df1 = steps_by_checkpoint[checkpoint_short_name]
                    df2 = run[checkpoint_short_name]
                    steps_by_checkpoint[checkpoint_short_name] = pd.concat([df1, df2], axis=0)
                    steps_by_checkpoint[checkpoint_short_name].reset_index(drop=True, inplace=True)
                else:
                    steps_by_checkpoint[checkpoint_short_name] = run[checkpoint_short_name]
        for checkpoint_short_name in steps_by_checkpoint.keys():
            steps_by_checkpoint[checkpoint_short_name].to_csv(os.path.join('logs', config['run_folder'], stats_folder,
                                     f"number_steps_{checkpoint_short_name}.csv"), index=False)
    elif config['option'] == "trajectories":
        print("Merged --- Trajectories")
        # Initialize with empty DataFrames
        overall_clean_trajectories = pd.DataFrame()
        for run in run_outcomes:
            for checkpoint_short_name in run.keys():
                print(run)
                print(checkpoint_short_name)
                clean_df = run[checkpoint_short_name]
                overall_clean_trajectories = pd.concat([overall_clean_trajectories, clean_df], axis=0)

        overall_clean_trajectories.to_csv(os.path.join('logs', config['run_folder'], stats_folder,
                                        f"clean_trajectories_merged_{config['num_episodes_per_checkpoint']}_{config['num_iterations']}.csv"), index=False)

def random_agent_stats(config):
    # Load test environments from specific run folder
    if config['load_test_envs']:
        test_envs_path = os.path.join('logs', config['run_folder'], str(config['run_id']), 'test_envs.pkl')
        with open(test_envs_path, 'rb') as test_file:
            gym_envs = pickle.load(test_file)
        print(f"Test environments loaded from {test_envs_path}")

    if not os.path.exists(os.path.join('logs', config['run_folder'], "stats", "random", str(config['run_id']))):
        os.makedirs(os.path.join('logs', config['run_folder'], "stats", "random", str(config['run_id'])))

    episode_list = []
    random_agent_owned_list = []
    random_agent_discovered_list = []
    random_agent_credentials_list = []

    stats_data = []
    for episode in range(config['num_episodes_per_checkpoint']):
        if config['switch']:
            if episode % config['switch_interval'] == 0:
                gym_env = random.choices(gym_envs, k=1)[0]
                #gym_env = copy.deepcopy(original_gym_env)
                print(f"Switching to a new env for episodes {episode} - {episode + config['switch_interval']}")
                index_of_chosen_env = gym_envs.index(gym_env)
                print(f"The chosen environment is at index: {index_of_chosen_env}")
        else:
            gym_env = gym_envs
        gym_env.reset()
        gym_env.set_cut_off(config['num_iterations'])

        while (True):
            action = gym_env.action_space.sample()
            next_state, _, done, _ = gym_env.step(action)
            if done:
                break
        owned_nodes, discovered_nodes, _, num_nodes, num_discovered_credentials = gym_env.get_statistics()
        episode_list.append(episode)
        random_agent_owned_list.append((owned_nodes / num_nodes))
        random_agent_discovered_list.append((discovered_nodes / num_nodes))
        random_agent_credentials_list.append(num_discovered_credentials)
        stats_data.append({
            'episode': episode,
            'agent': 'random',
            'owned_nodes': (owned_nodes / num_nodes),
            'discovered_nodes': (discovered_nodes / num_nodes),
            'num_discovered_credentials': num_discovered_credentials,
        })

    df = pd.DataFrame(stats_data,columns=['episode', 'agent', 'owned_nodes', 'discovered_nodes', 'num_discovered_credentials'])

    print("Calculating the average performance of the agent...")
    average_owned_percentage, lower_bound_owned_percentage, upper_bound_owned_percentage = bootstrap_ci(random_agent_owned_list)
    average_discovered_percentage, lower_bound_discovered_percentage, upper_bound_discovered_percentage = bootstrap_ci(random_agent_discovered_list)
    average_number_credentials, lower_bound_number_credentials, upper_bound_number_credentials = bootstrap_ci(random_agent_credentials_list)
    print("--------------------")
    print(f"Average owned percentage: {average_owned_percentage} [{lower_bound_owned_percentage}, {upper_bound_owned_percentage}]")
    print(f"Average discovered percentage: {average_discovered_percentage} [{lower_bound_discovered_percentage}, {upper_bound_discovered_percentage}]")
    print(f"Average number of credentials: {average_number_credentials} [{lower_bound_number_credentials}, {upper_bound_number_credentials}]")
    print("--------------------")
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    df.to_csv(os.path.join('logs', config['run_folder'], "stats", "random", str(config['run_id']), f"random_performances_{config['num_episodes_per_checkpoint']}_{current_time}.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description='Interpret and monitor metrics of a neural network in a given environment')
    parser.add_argument('--run_folder', required=True, help='Path to the run folder')
    parser.add_argument('--run_id', default=-1, type=int, help='Run ID to gather the correct run metrics')
    parser.add_argument('--algorithm', choices=['dqn', 'dqn_random', 'ppo', 'a2c', 'recurrent_ppo', 'qr_dqn', 'qr_dqn_random', 'trpo'], default='dqn', help='Algorithm to use (default: dqn)')
    parser.add_argument('--environment', type=str, choices=['random', 'chain'], default='chain',
                        help='Gym environment to test on')
    parser.add_argument('--switch', default=True, action="store_true",
                        help='Switch environment periodically during the run')
    parser.add_argument('--test_folder', required=False, help='Path to the test folder')
    parser.add_argument('--load_test_envs', default=True, action="store_true",
                        help='Load test environments instead of generating them')
    parser.add_argument('--option', default='action_distribution', choices=['random_performances', 'action_distribution','average_performances', 'number_steps', 'trajectories'],
                        help='Decide which statistics to plot')
    parser.add_argument('--last_checkpoint', default=False, action="store_true",
                        help='Load the last checkpoint only')
    parser.add_argument('--val_checkpoints', default=False, action="store_true", help='Use validation checkpoints instead of training checkpoints')
    parser.add_argument('--config', type=str, default='test_config.yaml', help='Path to the configuration YAML file')
    parser.add_argument('--no_random', default=False, action="store_true", help='Avoid calculation of average performances for the random agent')
    parser.add_argument('--random_percentage', default=0, type=float, help='Percentage of random actions to take')
    args = parser.parse_args()

    if args.random_percentage and not (args.algorithm == 'dqn_random' or args.algorithm == 'qr_dqn_random'):
        print("Random percentage is only valid for the dqn_random algorithm")
        sys.exit(1)

    if args.run_id == -1:
        print("--- Runs: Using all runs and averaging...")
        runs = [folder for folder in os.listdir(os.path.join('logs', args.run_folder)) if len(folder) == 1 or len(folder) == 2]
        num_runs = len(runs)
        print(f"--- Number of runs: {num_runs}")
        args.run_id = runs
    else:
        print("--- Run: ", args.run_id)

    # Read YAML configuration file
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
    for key, value in vars(args).items():
        config[key] = value

    train_config_file = os.path.join('logs', args.run_folder, 'train_config.yaml')
    with open(train_config_file, 'r') as train_config_file:
        train_config = yaml.safe_load(train_config_file)

    norm_obs = train_config['norm_obs']

    # independent from checkpoints
    if config['option'] == "random_performances":
        print("--- Option: Random performances")
        if isinstance(config['run_id'], list):
            for run_id in config['run_id']:
                config['run_id'] = run_id
                print("--- Run ID: ", run_id)
                random_agent_stats(config)
    else:
        if isinstance(config['run_id'], list):
            runs_outcomes = []
            for run_id in config['run_id']:
                config['run_id'] = run_id
                print("--- Run ID: ", run_id)
                outcomes = parse_option(config, norm_obs)
                runs_outcomes.append(outcomes)
            if not args.val_checkpoints or config['option'] == "average_performances" or config['option'] == "trajectories":
                merge_outcomes(runs_outcomes, config)
        else:
            _ = parse_option(config, norm_obs) # no need to average outcomes if you have one run only, it makes no sense to average among many checkpoints

if __name__ == "__main__":
    main()
