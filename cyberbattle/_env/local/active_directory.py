from cyberbattle.samples.active_directory import generate_ad
from cyberbattle._env.local import cyberbattle_moving_env as cyberbattle_moving_env
from cyberbattle.samples.active_directory import tiny_ad
from cyberbattle.simulation import model as m

class CyberBattleActiveDirectory(cyberbattle_moving_env.CyberBattleEnv):
    """CyberBattle simulation based on real world Active Directory networks"""

    def __init__(self, seed, random_starter_node=None, **kwargs):
        if random_starter_node == None:
            random_starter_node = True # this game may involve a different starter node each time
        super().__init__(initial_environment=generate_ad.new_random_environment(seed, env_type="ad_env"), random_starter_node=random_starter_node, env_type="ad_env", **kwargs)


class CyberBattleCustomActiveDirectory(cyberbattle_moving_env.CyberBattleEnv):
    """CyberBattle simulation based on real world Active Directory networks"""
    def __init__(self, num_clients, num_servers, num_users, admin_probability=0.8, random_starter_node=None, env_type="ad", **kwargs):
        if random_starter_node == None:
            random_starter_node = True # this game may involve a different starter node each time
        self.num_clients = num_clients
        self.num_servers = num_servers
        self.num_users = num_users
        self.admin_probability = admin_probability
        self.env_type = env_type
        self.size = (num_clients + num_servers + num_users) / admin_probability # defined size proportional to all the attributes
        network_ad = generate_ad.create_network_from_smb_traffic(num_clients, num_servers, num_users, admin_probability)
        env_ad = m.Environment(network=network_ad,
                      vulnerability_library=dict([]),
                      identifiers=generate_ad.ENV_IDENTIFIERS,
                               env_type=self.env_type)
        env_ad.num_credentials = generate_ad.num_credentials
        super().__init__(initial_environment=env_ad, random_starter_node=random_starter_node, env_type="ad_env", **kwargs)


class CyberBattleActiveDirectoryTiny(cyberbattle_moving_env.CyberBattleEnv):
    def __init__(self, random_starter_node=None, **kwargs):
        if random_starter_node == None:
            random_starter_node = True
        super().__init__(initial_environment=tiny_ad.new_environment(), random_starter_node=random_starter_node, env_type="ad_tiny_env", **kwargs)
