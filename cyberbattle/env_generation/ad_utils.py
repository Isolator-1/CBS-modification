import cyberbattle._env.local.active_directory as cyberbattle_local_ad
import random
from random import randint
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../agents", "..", ".."))

# generation of multiple active directories of different sizes
def generate_ads(num_environments, num_clients_list, num_servers_list, num_users_list, admin_probability_list,
                 visible_local_node_features=None, visible_local_global_features=None, **kwargs):
    ads = []
    for i in range(num_environments):
        num_clients = randint(num_clients_list[0], num_clients_list[1])
        num_servers = randint(num_servers_list[0], num_servers_list[1])
        num_users = randint(num_users_list[0], num_users_list[1])
        admin_probability = random.uniform(admin_probability_list[0], admin_probability_list[1])
        env = cyberbattle_local_ad.CyberBattleCustomActiveDirectory(num_clients, num_servers, num_users, admin_probability,
                                                       visible_node_features=visible_local_node_features, visible_global_features=visible_local_global_features,
                                                      **kwargs)
        ads.append(env)
    return ads


def wrap_ads(net_sizes, visible_local_node_features=None, visible_local_global_features=None, **kwargs):
    ads = []
    for net_size in net_sizes:
        print(net_size)
        num_clients, num_servers, num_users, admin_probability, size = net_size
        env = cyberbattle_local_ad.CyberBattleCustomActiveDirectory(num_clients, num_servers, num_users, admin_probability,
                                                       visible_node_features=visible_local_node_features,
                                                       visible_global_features=visible_local_global_features,
                                                       **kwargs)
        ads.append(env)
    return ads
