import cyberbattle._env.local.cyberbattle_random as cyberbattle_local_random
import numpy as np
import random
from random import randint
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../agents", "..", ".."))
import cyberbattle.simulation.model as model
import cyberbattle.simulation.generate_network as g
from cyberbattle._env.local.cyberbattle_moving_env import AttackerGoal

# creation of CyberBattleRandom network based on beta distribution and number of nodes
def create_default_random_network(n_clients, n_servers, alpha, beta, seed = None, **kwargs):
    # print(alpha, beta)
    network_parameters = dict(n_clients=n_clients, seed=seed, n_servers=n_servers, alpha=alpha, beta=beta)
    env = model.Environment(network_parameters=network_parameters, vulnerability_library=dict([]),
                            identifiers=g.ENV_IDENTIFIERS, env_type="random_env", **kwargs)
    return env


def create_default_random_network_by_range(num_clients_list, num_servers_list, inter, intra, cached_rdp_password_probability_list,
        cached_smb_password_probability_list, cached_accessed_network_shares_probability_list, cached_password_has_changed_probability_list,
        traceroute_discovery_probability_list, probability_two_nodes_use_same_password_to_access_given_resource_list,
                                           firewall_rule_incoming_probability_list, firewall_rule_outgoing_probability_list, protocols, seed = None, **kwargs):

    num_clients = randint(num_clients_list[0], num_clients_list[1])
    num_servers = {}
    alpha = {}
    beta = {}

    for protocol in protocols:
        num_servers[protocol] = randint(num_servers_list[0], num_servers_list[1])
        alpha_intra = random.uniform(intra["alpha"][0], intra["alpha"][1])
        beta_intra = random.uniform(intra["beta"][0], intra["beta"][1])
        alpha_inter = random.uniform(inter["alpha"][0], inter["alpha"][1])
        beta_inter = random.uniform(inter["beta"][0], inter["beta"][1])
        alpha[protocol] = np.array([(alpha_intra, alpha_inter), (alpha_inter, alpha_intra)], dtype=float)
        beta[protocol] = np.array([(beta_intra, beta_inter), (beta_inter, beta_intra)], dtype=float)

    cached_rdp_password_probability = random.uniform(cached_rdp_password_probability_list[0], cached_rdp_password_probability_list[1])
    cached_smb_password_probability = random.uniform(cached_smb_password_probability_list[0], cached_smb_password_probability_list[1])
    cached_accessed_network_shares_probability = random.uniform(cached_accessed_network_shares_probability_list[0], cached_accessed_network_shares_probability_list[1])
    cached_password_has_changed_probability = random.uniform(cached_password_has_changed_probability_list[0], cached_password_has_changed_probability_list[1])
    traceroute_discovery_probability = random.uniform(traceroute_discovery_probability_list[0], traceroute_discovery_probability_list[1])
    probability_two_nodes_use_same_password_to_access_given_resource = random.uniform(probability_two_nodes_use_same_password_to_access_given_resource_list[0], probability_two_nodes_use_same_password_to_access_given_resource_list[1])
    firewall_rule_incoming_probability = random.uniform(firewall_rule_incoming_probability_list[0], firewall_rule_incoming_probability_list[1])
    firewall_rule_outgoing_probability = random.uniform(firewall_rule_outgoing_probability_list[0], firewall_rule_outgoing_probability_list[1])
    probabilities = {
        'cached_rdp_password_probability': cached_rdp_password_probability,
        'cached_smb_password_probability': cached_smb_password_probability,
        'cached_accessed_network_shares_probability': cached_accessed_network_shares_probability,
        'cached_password_has_changed_probability': cached_password_has_changed_probability,
        'traceroute_discovery_probability': traceroute_discovery_probability,
        'probability_two_nodes_use_same_password_to_access_given_resource': probability_two_nodes_use_same_password_to_access_given_resource,
        'firewall_rule_incoming_probability': firewall_rule_incoming_probability,
        'firewall_rule_outgoing_probability': firewall_rule_outgoing_probability
    }

    kwargs.update(probabilities)
    env = create_default_random_network(num_clients, num_servers, alpha, beta, seed, **kwargs)
    stats = {'num_clients': num_clients, 'num_servers': num_servers, 'alpha': alpha, 'beta': beta}
    stats.update(probabilities)
    return env, stats


# generation of multiple graphs respecting the constraints
def generate_graphs(num_environments, knows_reachability_range, knows_connectivity_range, access_reachability_range, access_connectivity_range, owned_threshold_winning_reward, visible_local_node_features, visible_local_global_features, **kwargs):
    graphs = []
    graphs_networks = []

    for i in range(num_environments):
        knows_reachability = 0
        knows_connectivity = 0
        access_reachability = 0
        access_connectivity = 0
        while knows_reachability < knows_reachability_range[0] or knows_reachability > knows_reachability_range[1] or knows_connectivity < knows_connectivity_range[0] or knows_connectivity > knows_connectivity_range[1] or access_reachability < access_reachability_range[0] or access_reachability > access_reachability_range[1] or access_connectivity < access_connectivity_range[0] or access_connectivity > access_connectivity_range[1]:
            env, env_info = create_default_random_network_by_range(**kwargs)
            cyber_env = cyberbattle_local_random.CyberBattleRandom(env, attacker_goal=AttackerGoal(
                                                                        own_atleast_percent=owned_threshold_winning_reward),
                                                                       visible_node_features=visible_local_node_features,
                                                                       visible_global_features=visible_local_global_features,
                                                                        **kwargs)
            knows_reachability = env.knows_reachability
            knows_connectivity = env.knows_connectivity
            access_reachability = env.access_reachability
            access_connectivity = env.access_connectivity
            print(knows_reachability, knows_connectivity, access_reachability, access_connectivity)
        print("Added {} graphs".format(i+1))
        graphs.append(cyber_env)
        graphs_networks.append(env)

    return graphs, graphs_networks


def wrap_graphs(nets, visible_local_node_features=None, visible_local_global_features=None, owned_threshold_winning_reward=0.9, **kwargs):
    graphs = []
    if isinstance(nets, tuple):
        nets = [nets[0]]
    for net in nets:
        if isinstance(net, tuple):
            net = net[0]
        cyber_env = cyberbattle_local_random.CyberBattleRandom(net,
                                                               attacker_goal=AttackerGoal(own_atleast_percent=owned_threshold_winning_reward),
                                                               visible_node_features=visible_local_node_features, visible_global_features=visible_local_global_features,
                                                               **kwargs)
        graphs.append(cyber_env)
        knows_reachability = net.knows_reachability
        knows_connectivity = net.knows_connectivity
        access_reachability = net.access_reachability
        access_connectivity = net.access_connectivity
        print(knows_reachability, knows_connectivity, access_reachability, access_connectivity)
    return graphs
