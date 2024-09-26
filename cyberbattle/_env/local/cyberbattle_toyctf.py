# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from cyberbattle.samples.toyctf import toy_ctf
from cyberbattle._env.local import cyberbattle_moving_env as cyberbattle_moving_env

class CyberBattleToyCtf(cyberbattle_moving_env.CyberBattleEnv):
    """CyberBattle simulation based on a toy CTF exercise"""

    def __init__(self, random_starter_node=None, **kwargs):
        if random_starter_node == None:
            random_starter_node = False
        self.env_type = "toyctf_env"
        super().__init__(
            initial_environment=toy_ctf.new_environment(self.env_type),
            env_type=self.env_type,
            random_starter_node=random_starter_node,
            **kwargs)
