import argparse
import copy
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from cyberbattle._env.local.cyberbattle_env_switch import RandomSwitchEnv
import yaml
from datetime import datetime
from stable_baselines3 import DQN, PPO
from stable_baselines3.a2c import A2C
from sb3_contrib import RecurrentPPO, QRDQN, TRPO
from stable_baselines3.common.callbacks import CheckpointCallback
from cyberbattle.agents.stable_baselines.callbacks import TrainingCallback, ValidationCallback
from cyberbattle.env_generation.env_utils import split_graphs, save_config, save_envs, save_networks, save_net_sizes, save_seeds
from cyberbattle.env_generation.random_utils import generate_graphs, wrap_graphs
from cyberbattle.env_generation.chain_utils import generate_chains, wrap_chains
from cyberbattle.env_generation.ad_utils import generate_ads, wrap_ads
import numpy as np
from cyberbattle.agents.train_utils import linear_schedule, get_box_variables, replace_with_classes
import random
import pickle
from stable_baselines3.common.env_checker import check_env
from cyberbattle._env.local.cyberbattle_moving_env import DefenderConstraint, DefenderGoal
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import torch
from cyberbattle._env.defender import ScanAndReimageCompromisedMachines, ExternalRandomEvents

def train_rl_algorithm(envs, algorithm, use_static_seed, seed, switch, num_runs, num_iterations, early_stopping,logs_folder, config, norm_obs=False, norm_reward=False, seed_env_generation=False, seeds_loaded=None):
    seeds = []
    seeds.append(seed_env_generation)
    for run_id in range(num_runs):
        # Varying the seed if needed at each run
        if not seeds_loaded:
            if not use_static_seed:
                seed = np.random.randint(1000)
        else:
            seed = seeds_loaded[run_id]

        np.random.seed(seed)
        random.seed(seed)
        seeds.append(seed)

        net_sizes = []
        # regenerate different environments for each run in chain and ad mode
        # configuration files will provide the proper range of generation
        if args.environment == "chain":
            print("Environment: CyberBattleChain")
            envs = generate_chains(**config)
            net_sizes = []
            for env in envs:
                net_sizes.append(env.size)
            print(f"Network sizes: {net_sizes}")
        elif args.environment == "ad":
            print("Environment: CyberBattleActiveDirectory")
            envs = generate_ads(**config)
            net_sizes = []
            for env in envs:
                net_sizes.append([env.num_clients, env.num_servers, env.num_users, env.admin_probability, env.size])
            print(f"Network sizes: {net_sizes}")

        for env in envs:
            env.seed(seed)

        # save net infos
        if net_sizes:
            save_net_sizes(net_sizes, os.path.join(logs_folder, str(run_id+1)))

        print(f"\nTraining {algorithm} - Run {run_id + 1}/{num_runs}")

        # Splitting environment in case of holdout method (switch flag)
        if switch:
            train_envs, val_envs, test_envs = split_graphs(envs, seed, config['validation_ratio'], config['test_ratio'])
            print(f"Number of training envs: {len(train_envs)}")
            print(f"Number of validation envs: {len(val_envs)}")
            print(f"Number of test envs: {len(test_envs)}")
        else:
            train_envs = envs
            val_envs = None
            test_envs = None

        # Saving environments used in this run for reproducibility
        save_envs(run_id+1, train_envs, val_envs, test_envs, logs_folder)

        # Creating wrappers to randomly switch environments to use during training and validation with a certain interval
        train_envs = RandomSwitchEnv(train_envs, config['switch_interval'])
        if switch:
            val_envs = RandomSwitchEnv(val_envs, config['val_switch_interval'])

        # checker only used in this case, since in the other case we do not rely on Stable Baselines 3
        check_env(train_envs)
        train_model(algorithm, train_envs, num_iterations, early_stopping, logs_folder, config, config['algo'], run_id+1, val_envs=val_envs, norm_obs=norm_obs, norm_reward=norm_reward)


    save_seeds(seeds, logs_folder)
    print("\nTraining completed.")


def train_model(algorithm_type, train_envs, num_iterations, early_stopping, logs_folder, config, algorithm_config, run_id, val_envs=None, norm_obs=False, norm_reward=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Learning rate scheduling (potential)
    algorithm_config = copy.deepcopy(algorithm_config)
    if algorithm_config[algorithm_type]['learning_rate_type'] == "linear":
        learning_rate = linear_schedule(algorithm_config[algorithm_type]['learning_rate'], algorithm_config[algorithm_type]['learning_rate_final'])
    elif algorithm_config[algorithm_type]['learning_rate_type'] == "constant":
        learning_rate = algorithm_config[algorithm_type]['learning_rate']

    algorithm_config[algorithm_type].pop('learning_rate_type', None)
    algorithm_config[algorithm_type].pop('learning_rate', None)
    algorithm_config[algorithm_type].pop('learning_rate_final', None)

    # Mapping algorithm type to model class and additional parameters
    algorithm_models = {
        'dqn': (DQN, algorithm_config['dqn']),
        'ppo': (PPO, algorithm_config['ppo']),
        'a2c': (A2C, algorithm_config['a2c']),
        'recurrent_ppo': (RecurrentPPO, algorithm_config['recurrent_ppo']),
        'qr_dqn': (QRDQN, algorithm_config['qr_dqn']),
        'trpo': (TRPO, algorithm_config['trpo'])
    }

    reccurrent_algorithms = ["recurrent_ppo"]

    # Ensure the algorithm type is valid
    if algorithm_type not in algorithm_models:
        raise ValueError(
            f"Invalid algorithm type: {algorithm_type}. Supported types: {', '.join(algorithm_models.keys())}")

    # Create the model
    model_class, additional_params = algorithm_models[algorithm_type]

    train_envs = DummyVecEnv([lambda: Monitor(train_envs)])

    # potential normalization
    norm_obs_keys = get_box_variables(train_envs.observation_space)
    train_envs = VecNormalize(train_envs, norm_obs=norm_obs, norm_obs_keys=norm_obs_keys, norm_reward=norm_reward)
    if val_envs:
        val_envs = DummyVecEnv([lambda: Monitor(val_envs)])
        norm_obs_keys = get_box_variables(val_envs.observation_space)
        val_envs = VecNormalize(val_envs, norm_obs=norm_obs, norm_obs_keys=norm_obs_keys, norm_reward=False) # if the reward is normalized, the scale plotted will be different

    # different policy according to whether the algorithm is memory-based or not
    if algorithm_type in reccurrent_algorithms:
        model = model_class("MultiInputLstmPolicy", train_envs, policy_kwargs=config['policy_kwargs'],
                            learning_rate=learning_rate,
                            tensorboard_log=logs_folder, **additional_params, verbose=1, device=device)
    else:
        lstm_keys = ['lstm_hidden_size', 'n_lstm_layers']
        for key in lstm_keys:
            if key in config['policy_kwargs']:
                del config['policy_kwargs'][key]
        model = model_class("MultiInputPolicy", train_envs, policy_kwargs=config['policy_kwargs'],
                            learning_rate=learning_rate,
                            tensorboard_log=logs_folder,  **additional_params, verbose=1, device=device)

    # Checkpoint saving
    checkpoint_callback = CheckpointCallback(save_freq=config['checkpoints_save_freq'],
                                             save_path=os.path.join(logs_folder, "checkpoints", str(run_id)),
                                             name_prefix='checkpoint')

    # Logging additional training metrics
    train_callback = TrainingCallback(env=train_envs)

    callbacks = [checkpoint_callback, train_callback]

    if val_envs:
        # Logging validation metrics, saves validation checkpoints, and use early stopping if requested
        val_callback = ValidationCallback(
            val_env=val_envs,
            n_val_episodes=config['n_val_episodes'],
            val_freq=config['val_freq'],
            log_path=os.path.join(logs_folder, "validation", str(run_id)),
            early_stopping=early_stopping,
            patience=early_stopping,
            verbose=1
        )
        callbacks.append(val_callback)

    model.learn(total_timesteps=num_iterations, callback=callbacks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL algorithm on CyberBattleSim environment with local view!")
    parser.add_argument('--algorithm', type=str, choices=['dqn', 'ppo', 'a2c', 'recurrent_ppo', 'qr_dqn', 'trpo'], default='dqn', help='RL algorithm to train')
    parser.add_argument('--environment', type=str, choices=['random', 'chain', 'ad'], default='random', help='CyberBattleSim environments to train on')
    parser.add_argument('--static_seed', action='store_true', default=False, help='Use a static seed for training')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--switch', action='store_true', default=False, help='Use validation and test sets and switch among environments periodically on the training and validation sets')
    parser.add_argument('--config', type=str, default='train_config.yaml', help='Path to the configuration YAML file')
    parser.add_argument('--early_stopping', type=int, default=0, help='Early stopping on the validation environments setting the number of patience runs')
    parser.add_argument('--yaml', default=False,
                        help='Read configuration file from YAML file of a previous training')
    parser.add_argument('--name', default=False,
                        help='Name of the logs folder related to the run')
    parser.add_argument('--load_envs', default=False,
                        help='Path of the run folder where the environment should be loaded from')
    parser.add_argument('--load_nets', default=False,
                        help='Path of the run folder where the networks should be loaded from')
    parser.add_argument('--load_seeds', default=False,
                        help='Path of the file where the seeds should be loaded from (It also determines the number of runs)')
    parser.add_argument('--random_agent', default=False, action="store_true", help='Run random agent only to get benchmark metrics')

    args = parser.parse_args()

    # Load fixed set of seeds
    seeds_loaded = None
    if args.load_seeds:
        print("Reading seeds from folder...")
        with open(os.path.join("..", "logs", args.load_seeds, 'seeds.yaml'), 'r') as seeds_file:
            seeds_loaded = yaml.safe_load(seeds_file)

    if not seeds_loaded:
        # Potential seed setting before the generation of environment
        if not args.static_seed:
            seed = np.random.randint(1000)
        else:
            seed = 42
    else:
        seed = seeds_loaded[0] # first seed for environment generation
        seeds_loaded = seeds_loaded[1:] # one seed per run
        args.num_runs = len(seeds_loaded)

    seed_env_generation = seed

    np.random.seed(seed)
    random.seed(seed)

    # Read YAML configuration file of a previous training
    if args.yaml:
        print("Reading YAML file from a previous training for reproducibility...")
        with open(os.path.join("..", "logs", args.yaml), 'r') as config_file:
            config = yaml.safe_load(config_file)
    else:
        # Read default YAML configuration file
        with open(args.config, 'r') as config_file:
            config = yaml.safe_load(config_file)
    config.update({"seed": seed})
    config.update({"random_agent": args.random_agent})

    # Creating logs folder
    if args.name:
        logs_folder = os.path.join('../logs', args.name + "_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        logs_folder = os.path.join('../logs', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(logs_folder, exist_ok=True)

    if not args.switch:
        config['num_environments'] = 1

    envs_networks = None

    envs = []

    net_sizes = None

    # Saving configuration yaml file with all information related to the training
    save_config(config, args, logs_folder)

    # map to classes after saving the configuration
    if config['defender_agent']:
        map_dict = {
            "ExternalRandomEvents": ExternalRandomEvents(config['random_event_probability']),
            "ScanAndReimageCompromisedMachines": ScanAndReimageCompromisedMachines(config['detect_probability'], config['scan_capacity'], config['scan_frequency'])
        }
        config['defender_agent'] = map_dict[config['defender_agent']]
        config['defender_constraint'] = DefenderConstraint(config['defender_sla_min'])
        config['defender_goal'] = DefenderGoal(config['defender_eviction_goal'])

    config['environment_type'] = config['environment']
    config.pop('environment', None)

    # Generating or loading environment(s)
    if args.load_envs:
        print("Loading specified environments instead of generating them...")
        train_envs_path = os.path.join('../logs', args.load_envs, 'train_envs.pkl')
        with open(train_envs_path, 'rb') as train_file:
            train_envs = pickle.load(train_file)
        val_envs_path = os.path.join('../logs', args.load_envs, 'val_envs.pkl')
        with open(val_envs_path, 'rb') as val_file:
            val_envs = pickle.load(val_file)
        test_envs_path = os.path.join('../logs', args.load_envs, 'test_envs.pkl')
        with open(test_envs_path, 'rb') as test_file:
            test_envs = pickle.load(test_file)
        train_envs.extend(val_envs)
        train_envs.extend(test_envs)
        envs = train_envs
    elif args.load_nets: # loading only configuration parameters or random networkx
        if args.environment == "random":
            print("Loading specified networks and wrapping them instead of generating them...")
            networks_path = os.path.join('../logs',  args.load_nets, 'networks.pkl')
            with open(networks_path, 'rb') as nets_file:
                networks = pickle.load(nets_file)
            envs_networks = networks
        elif args.environment == "chain" or args.environment == "ad":
            file_path = os.path.join('../logs', args.load_nets, 'net_sizes.yaml')
            with open(file_path, 'r') as file:
                net_sizes = yaml.safe_load(file)

    if args.load_envs:
        pass
    elif args.load_nets: # random mode for sure, since chain is reproducible exactly at the same way
        # wrap in the environments
        if args.environment == "random":
            envs = wrap_graphs(networks, **config)
        elif args.environment == "chain":
            envs = wrap_chains(net_sizes,**config)
        elif args.environment == "ad":
            envs = wrap_ads(net_sizes, **config)
    elif args.environment == "random":
        # handling this before because the environment should be generated, differently from the other two cases which are deterministic
        print("Environment: CyberBattleRandom")
        envs, envs_networks = generate_graphs(**config)

    for env in envs:
        env.seed(seed)

    if envs_networks:
        save_networks(envs_networks, logs_folder)

    config['policy_kwargs'] = replace_with_classes(config['policy_kwargs'])

    train_rl_algorithm(envs, args.algorithm, args.static_seed, seed, args.switch, args.num_runs, config['train_iterations'], args.early_stopping, logs_folder, config, config['norm_obs'], config['norm_reward'], seed_env_generation, seeds_loaded)
