import cyberbattle._env.local.cyberbattle_chain as cyberbattle_local_chain
from random import randint
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../agents", "..", ".."))

# generation of multiple chains of different sizes
def generate_chains(num_environments, num_clients_list, episode_iterations, optimal_num_iterations=False, tolerance_factor=0.2,
                    visible_local_node_features=None, visible_local_global_features=None, **kwargs):
    chains = []
    for i in range(num_environments):
        num_clients = 1
        while num_clients % 2 == 1:
            num_clients = randint(num_clients_list[0], num_clients_list[1])
        if optimal_num_iterations:
            max_num_iterations = 3 + 3 * num_clients # start(local) + start(connect) + start(movement) + nodei(local) + nodei(connect) + nodei(movement) + ......
            max_num_iterations += max_num_iterations * tolerance_factor
        else:
            max_num_iterations = episode_iterations
        max_num_iterations = int(max_num_iterations)
        print("Num iterations", max_num_iterations)
        env = cyberbattle_local_chain.CyberBattleChain(num_clients, visible_node_features=visible_local_node_features, visible_global_features=visible_local_global_features, episode_iterations=max_num_iterations, **kwargs)
        chains.append(env)
    return chains

def wrap_chains(net_sizes, episode_iterations, optimal_num_iterations=False, tolerance_factor=0.2, visible_local_node_features=None, visible_local_global_features=None, **kwargs):
    chains = []
    for net_size in net_sizes:
        if optimal_num_iterations:
            max_num_iterations = 3 + 3 * net_size # start(local) + start(connect) + start(movement) + nodei(local) + nodei(connect) + nodei(movement) + ......
            max_num_iterations += max_num_iterations * tolerance_factor
        else:
            max_num_iterations = episode_iterations
        max_num_iterations = int(max_num_iterations)
        print("Num iterations", max_num_iterations)
        env = cyberbattle_local_chain.CyberBattleChain(net_size,
                                                       episode_iterations=max_num_iterations,
                                                       visible_node_features=visible_local_node_features,
                                                       visible_global_features=visible_local_global_features,
                                                       **kwargs)
        chains.append(env)
    return chains


