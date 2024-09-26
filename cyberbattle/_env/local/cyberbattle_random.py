# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A CyberBattle simulation over a randomly generated network"""

from cyberbattle._env.local import cyberbattle_moving_env as cyberbattle_local_env
import cyberbattle.simulation.generate_network as g

class CyberBattleRandom(cyberbattle_local_env.CyberBattleEnv):
    """A sample CyberBattle environment"""

    def __init__(self, environment=None, attacker_goal=None, random_starter_node=None, **kwargs):
        # network not provided: using the default one
        self.env_type = "random_env"

        if environment == None:
            environment = g.new_environment(n_servers_per_protocol=15, env_type=self.env_type)

        if random_starter_node == None:
            random_starter_node = True # this game may involve a different starter node each time


        # attacker goal available
        if attacker_goal:
            super().__init__(initial_environment=environment,
                             attacker_goal=attacker_goal,
                             env_type=self.env_type,
                             random_starter_node=random_starter_node,
                             **kwargs
                             )
        # if not available, not passing None, hence using the default one
        else:
            super().__init__(initial_environment=environment, env_type="random_env", random_starter_node=random_starter_node, **kwargs)
