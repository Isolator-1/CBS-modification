# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from cyberbattle.samples.toyctf import tinytoy
from cyberbattle._env.local import cyberbattle_moving_env as cyberbattle_moving_env

class CyberBattleTiny(cyberbattle_moving_env.CyberBattleEnv):
    """CyberBattle simulation on a tiny environment. (Useful for debugging purpose)"""
    def __init__(self, random_starter_node=None, **kwargs):
        if random_starter_node == None:
            random_starter_node = False # this game involves the same starter node each time
        self.env_type = "tiny_env"
        super().__init__(
            initial_environment=tinytoy.new_environment(env_type=self.env_type),
            env_type=self.env_type,
            random_starter_node=random_starter_node,
            **kwargs)
