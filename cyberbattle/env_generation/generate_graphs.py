import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from datetime import datetime
from cyberbattle.env_generation.env_utils import save_networks
from cyberbattle.env_generation.random_utils import create_default_random_network_by_range
import yaml
import pickle

def load_networks(folder_path):
    networks_path = os.path.join(folder_path, 'networks.pkl')
    if os.path.exists(networks_path):
        with open(networks_path, 'rb') as networks_file:
            networks = pickle.load(networks_file)
        print(f"Networks loaded from {networks_path}")
        return networks
    else:
        print(f"No networks file found at {networks_path}")
        return None


def custom_sort(item):
    #knows = 0.5 * item[1]["knows_connectivity"] + 0.5 * item[1]["knows_rechability"]
    #access = 0.5 * item[1]["access_connectivity"] + 0.5 * item[1]["access_reachability"]
    f = item[1]["access_connectivity"]
    return f


# generation of multiple graphs respecting the constraints
def generate_graphs(logs_folder, num_environments, **kwargs):
    graphs_networks = []
    for i in range(num_environments):
        print("Generating graph {}...".format(i))
        env, env_info = create_default_random_network_by_range(**kwargs)
        knows_connectivity = env.knows_connectivity
        knows_reachability = env.knows_reachability
        access_connectivity = env.access_connectivity
        access_reachability = env.access_reachability
        print("Knows:", knows_connectivity, knows_reachability)
        print("Access:", access_connectivity, access_reachability)

        stats = {
            'knows_connectivity': knows_connectivity,
            'knows_rechability': knows_reachability,
            'access_connectivity': access_connectivity,
            'access_reachability': access_reachability
        }
        stats.update(env_info)
        graph_stats = (env, stats)
        save_networks(graph_stats, logs_folder, str(i+1))
        graphs_networks.append(graph_stats)
    sorted_graphs_networks = sorted(graphs_networks, key=custom_sort)
    for item in sorted_graphs_networks:
        stats = item[1]
        print(stats)
    return graphs_networks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL algorithm on CyberBattleSim environment with local view!")
    parser.add_argument('--config', type=str, default='graphs_generation.yaml', help='Path to the configuration YAML file')
    parser.add_argument('--load', default=False, help='Load networks from file')
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    if args.load:
        print("Loading existing graphs...")
        envs_networks = load_networks(args.load)
        print("------------------------ Local ------------------------------------ Original ------------------------")
        for net in envs_networks:
            print(net[1], net[2])
    else:
        print("Generating new graphs...")
        print("-----------------------------------")
        logs_folder = os.path.join('./logs', "graphs_generation_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(logs_folder, exist_ok=True)
        envs_networks = generate_graphs(logs_folder, **config)
