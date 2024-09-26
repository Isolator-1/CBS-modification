# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CyberBattle environment based on a simple chain network structure"""
from cyberbattle._env.local import cyberbattle_moving_env as cyberbattle_local_env
from cyberbattle.samples.chainpattern import chainpattern
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", ".."))
sys.path.insert(0, project_root)


class CyberBattleChain(cyberbattle_local_env.CyberBattleEnv):
    """CyberBattle environment based on a simple chain network structure"""

    def __init__(self, size, random_starter_node=None, **kwargs):
        self.size = size
        self.env_type = "chain_env"  # defined as attribute since it is also used outside
        if random_starter_node == None:
            random_starter_node = False # this game involves the same starter node each time
        super().__init__(
            initial_environment=chainpattern.new_environment(size),
            env_type=self.env_type,
            random_starter_node=random_starter_node,
            **kwargs)

    @ property
    def name(self) -> str:
        return f"CyberBattleChain-{self.size}"
