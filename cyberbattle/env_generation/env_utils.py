import pickle
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import yaml
import gymnasium

def calculate_observation_space_dimensions(observation_space):
    total_dimensions = 0

    if isinstance(observation_space, gymnasium.spaces.Dict):
        for space_key, space in observation_space.spaces.items():
            if isinstance(space, gymnasium.spaces.Discrete):
                total_dimensions += 1
            elif isinstance(space, gymnasium.spaces.Box):
                total_dimensions += space.shape[0]

    return total_dimensions


# Splitting environment set in training, validation and test sets
def split_graphs(graphs, seed, validation_ratio=0.25, test_ratio=0.2):
    if graphs[0].env_type == "chain_env" or graphs[0].env_type == "ad_env":
        train_graphs, val_graphs, test_graphs = split_graph_by_size(graphs, validation_ratio, test_ratio)
    else:
        if test_ratio:
            train_graphs, test_graphs = train_test_split(graphs, test_size=test_ratio, random_state=seed)
            train_graphs, val_graphs = train_test_split(train_graphs, test_size=validation_ratio, random_state=seed)
        else:
            train_graphs, val_graphs = train_test_split(graphs, test_size=validation_ratio, random_state=seed)
            test_graphs = None
    return train_graphs, val_graphs, test_graphs

def split_graph_by_size(graphs, validation_ratio=0.25, test_ratio=0.2):
    sorted_graphs = sorted(graphs, key=lambda x: x.size, reverse=True)

    total_graphs = len(sorted_graphs)
    test_size = int(total_graphs * test_ratio)
    validation_size = int(total_graphs * (1 - test_ratio) * validation_ratio)

    test_graphs, rest_graphs = sorted_graphs[:test_size], sorted_graphs[test_size:]
    val_graphs, train_graphs = rest_graphs[:validation_size], rest_graphs[validation_size:]

    return train_graphs, val_graphs, test_graphs


# Calculating the observation space and action space sizes
def get_input_output_dimensions(env):
    state_example = env.reset()
    flat_state = state_example
    input_dim = len(flat_state)
    output_dim = env.action_space.n
    print("Possible actions:", env.identifiers.local_vulnerabilities, env.identifiers.remote_vulnerabilities, env.identifiers.ports)
    print("State space dimensions:", input_dim)
    print("Action space dimensions:", output_dim)
    return input_dim, output_dim

# Save environments in the log folder
def save_envs(run_id, train_envs, val_envs, test_envs, folder_path):
    train_envs_path = os.path.join(folder_path, str(run_id), 'train_envs.pkl')
    os.makedirs(os.path.dirname(train_envs_path), exist_ok=True)

    with open(train_envs_path, 'wb') as train_file:
        pickle.dump(train_envs, train_file)
    print(f"Train environments saved to {train_envs_path}")
    if val_envs:
        val_envs_path = os.path.join(folder_path, str(run_id), 'val_envs.pkl')
        with open(val_envs_path, 'wb') as val_file:
            pickle.dump(val_envs, val_file)
        print(f"Validation environments saved to {val_envs_path}")
    if test_envs:
        test_envs_path = os.path.join(folder_path, str(run_id), 'test_envs.pkl')
        with open(test_envs_path, 'wb') as test_file:
            pickle.dump(test_envs, test_file)
        print(f"Test environments saved to {test_envs_path}")

# Save only the networks to ensure reproducibility
def save_networks(networks, folder_path, name=None):
    if name:
        networks_path = os.path.join(folder_path, 'network_'+name+'.pkl')
    else:
        networks_path = os.path.join(folder_path, 'networks.pkl')
    os.makedirs(os.path.dirname(networks_path), exist_ok=True)

    with open(networks_path, 'wb') as networks_file:
        pickle.dump(networks, networks_file)
    print(f"Networks saved to {networks_path}")

# To ensure reproducibility as well
def save_seeds(seeds, folder_path):
    seeds_path = os.path.join(folder_path, 'seeds.yaml')
    with open(seeds_path, 'w') as file:
        yaml.dump(seeds, file)
    print(f"Seeds saved to {seeds_path}")

# Save only the networks to ensure reproducibility in
def save_net_sizes(net_sizes, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, 'net_sizes.yaml')
    with open(file_path, 'w') as file:
        yaml.dump(net_sizes, file)
    print(f"Sizes saved to {file_path}")

# Save configuration in the logs folder
def save_config(config, args, folder_path, file_name="train_config.yaml"):
    if args != None:
        config.update(vars(args))
    config_path = os.path.join(folder_path, file_name)

    with open(config_path, 'w') as config_file:
        yaml.safe_dump(config, config_file)
    print(f"Configuration saved to {config_path}")
