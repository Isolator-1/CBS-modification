# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Initialize CyberBattleSim module"""
from gym.envs.registration import registry, EnvSpec
from gym.error import Error

from . import simulation
from . import agents
from ._env.local.cyberbattle_moving_env import AttackerGoal, DefenderGoal
from .samples.chainpattern import chainpattern
from .samples.toyctf import toy_ctf
from .samples.active_directory import generate_ad
from .simulation import generate_network, model

__all__ = (
    'simulation',
    'agents',
)

default_kwargs = {
    'reward_coefficients': {
        'value_coefficient': 1.0,
        'cost_coefficient': 1.0,
        'property_discovered_coefficient': 2.0,
        'credential_discovered_coefficient': 3.0,
        'node_discovered_coefficient': 5.0,
        'first_success_attack_coefficient': 7.0,
        'moved_source_node_unlock': 0.0
    },
    'penalties': {
        'suspiciousness': -5.0, # penalty for generic suspiciousness
        'scanning_unopen_port': -10.0, # penalty for attempting a connection to a port that was not open
        'repeat': -1, # penalty for repeating the same exploit attempt
        'local_exploit_failed': -20,
        'failed_remote_exploit': -50,
        'machine_not_running': 0, # penalty for attempting to connect or execute an action on a node that's not in running state
        'wrong_password': -10, # penalty for attempting a connection with an invalid password
        'blocked_by_local_firewall': -10, # traffic blocked by outgoing rule in a local firewall
        'blocked_by_remote_firewall': -10, # traffic blocked by incoming rule in a local firewall
        'invalid_action': -1,  # invalid action (e.g., running an attack from a node that's not owned)
        'invalid_movement': -50,
        'movement': 0,
        'connection_to_same_node': -50
    }
}


def register(id: str, cyberbattle_env_identifiers: model.Identifiers, **kwargs):
    """ same as gym.envs.registry.register, but adds CyberBattle specs to env.spec  """
    if id in registry.env_specs:
        raise Error('Cannot re-register id: {}'.format(id))
    spec = EnvSpec(id, **kwargs)
    # Map from port number to port names : List[model.PortName]
    spec.ports = cyberbattle_env_identifiers.ports
    # Array of all possible node properties (not necessarily all used in the network) : List[model.PropertyName]
    spec.properties = cyberbattle_env_identifiers.properties
    # Array defining an index for every possible local vulnerability name : List[model.VulnerabilityID]
    spec.local_vulnerabilities = cyberbattle_env_identifiers.local_vulnerabilities
    # Array defining an index for every possible remote  vulnerability name : List[model.VulnerabilityID]
    spec.remote_vulnerabilities = cyberbattle_env_identifiers.remote_vulnerabilities

    registry.env_specs[id] = spec


if 'CyberBattleToyCtf-v0' in registry.env_specs:
    del registry.env_specs['CyberBattleToyCtf-v0']

register(
    id='CyberBattleToyCtf-v0',
    cyberbattle_env_identifiers=toy_ctf.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.cyberbattle_toyctf:CyberBattleToyCtf',
    kwargs={'defender_agent': None,
            'attacker_goal': AttackerGoal(own_atleast=6),
            'defender_goal': DefenderGoal(eviction=True)
            },
    # max_episode_steps=2600,
)

if 'CyberBattleTiny-v0' in registry.env_specs:
    del registry.env_specs['CyberBattleTiny-v0']

register(
    id='CyberBattleTiny-v0',
    cyberbattle_env_identifiers=toy_ctf.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.cyberbattle_tiny:CyberBattleTiny',
    kwargs={'defender_agent': None,
            'attacker_goal': AttackerGoal(own_atleast=6),
            'defender_goal': DefenderGoal(eviction=True),
            'maximum_total_credentials': 10,
            'maximum_node_count': 10
            },
    # max_episode_steps=2600,
)


if 'CyberBattleRandom-v0' in registry.env_specs:
    del registry.env_specs['CyberBattleRandom-v0']

register(
    id='CyberBattleRandom-v0',
    cyberbattle_env_identifiers=generate_network.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.cyberbattle_random:CyberBattleRandom',
)

if 'CyberBattleChain-v0' in registry.env_specs:
    del registry.env_specs['CyberBattleChain-v0']

register(
    id='CyberBattleChain-v0',
    cyberbattle_env_identifiers=chainpattern.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.cyberbattle_chain:CyberBattleChain',
    kwargs={'size': 50,
            'defender_agent': None,
            'attacker_goal': AttackerGoal(own_atleast_percent=1.0),
            'defender_goal': DefenderGoal(eviction=True),
            'winning_reward': 5000.0,
            'losing_reward': 0.0,
            **default_kwargs
            },
    reward_threshold=2200,
)

ad_envs = [f"ActiveDirectory-v{i}" for i in range(0, 10)]
for (index, env) in enumerate(ad_envs):
    if env in registry.env_specs:
        del registry.env_specs[env]

    register(
        id=env,
        cyberbattle_env_identifiers=generate_ad.ENV_IDENTIFIERS,
        entry_point='cyberbattle._env.active_directory:CyberBattleActiveDirectory',
        kwargs={
            'seed': index,
            'maximum_discoverable_credentials_per_action': 50000,
            'maximum_node_count': 30,
            'maximum_total_credentials': 50000,
        }
    )

if 'ActiveDirectoryTiny-v0' in registry.env_specs:
    del registry.env_specs['ActiveDirectoryTiny-v0']
register(
    id='ActiveDirectoryTiny-v0',
    cyberbattle_env_identifiers=chainpattern.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.active_directory:CyberBattleActiveDirectoryTiny',
    kwargs={'maximum_discoverable_credentials_per_action': 50000,
            'maximum_node_count': 30,
            'maximum_total_credentials': 50000
            }
)

# Probabilistic action selection

if 'CyberBattleToyCtfProbabilistic-v0' in registry.env_specs:
    del registry.env_specs['CyberBattleToyCtfProbabilistic-v0']

register(
    id='CyberBattleToyCtfProbabilistic-v0',
    cyberbattle_env_identifiers=toy_ctf.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.cyberbattle_toyctf:CyberBattleToyCtf',
    kwargs={'defender_agent': None,
            'attacker_goal': AttackerGoal(own_atleast=6),
            'defender_goal': DefenderGoal(eviction=True),
            'random_mode': "probabilistic",
            **default_kwargs
            },
    # max_episode_steps=2600,
)

if 'CyberBattleTinyProbabilistic-v0' in registry.env_specs:
    del registry.env_specs['CyberBattleTinyProbabilistic-v0']

register(
    id='CyberBattleTinyProbabilistic-v0',
    cyberbattle_env_identifiers=toy_ctf.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.cyberbattle_tiny:CyberBattleTiny',
    kwargs={'defender_agent': None,
            'attacker_goal': AttackerGoal(own_atleast=6),
            'defender_goal': DefenderGoal(eviction=True),
            'maximum_total_credentials': 10,
            'maximum_node_count': 10,
            'random_mode': "probabilistic",
            **default_kwargs
            },
    # max_episode_steps=2600,
)


if 'CyberBattleRandomProbabilistic-v0' in registry.env_specs:
    del registry.env_specs['CyberBattleRandomProbabilistic-v0']

register(
    id='CyberBattleRandomProbabilistic-v0',
    cyberbattle_env_identifiers=generate_network.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.cyberbattle_random:CyberBattleRandom',
    kwargs={'random_mode': "probabilistic",
            **default_kwargs
            },
)

if 'CyberBattleChainProbabilistic-v0' in registry.env_specs:
    del registry.env_specs['CyberBattleChainProbabilistic-v0']

register(
    id='CyberBattleChainProbabilistic-v0',
    cyberbattle_env_identifiers=chainpattern.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.cyberbattle_chain:CyberBattleChain',
    kwargs={'size': 50,
            'defender_agent': None,
            'attacker_goal': AttackerGoal(own_atleast_percent=1.0),
            'defender_goal': DefenderGoal(eviction=True),
            'winning_reward': 5000.0,
            'losing_reward': 0.0,
            'random_mode': "probabilistic",
            **default_kwargs
            },
    reward_threshold=2200,
)

ad_envs = [f"ActiveDirectoryProbabilistic-v{i}" for i in range(0, 10)]
for (index, env) in enumerate(ad_envs):
    if env in registry.env_specs:
        del registry.env_specs[env]

    register(
        id=env,
        cyberbattle_env_identifiers=generate_ad.ENV_IDENTIFIERS,
        entry_point='cyberbattle._env.active_directory:CyberBattleActiveDirectory',
        kwargs={
            'seed': index,
            'maximum_discoverable_credentials_per_action': 50000,
            'maximum_node_count': 30,
            'maximum_total_credentials': 50000,
            'random_mode': "probabilistic",
            **default_kwargs
        }
    )

if 'ActiveDirectoryTinyProbabilistic-v0' in registry.env_specs:
    del registry.env_specs['ActiveDirectoryTinyProbabilistic-v0']
register(
    id='ActiveDirectoryTinyProbabilistic-v0',
    cyberbattle_env_identifiers=chainpattern.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.active_directory:CyberBattleActiveDirectoryTiny',
    kwargs={'maximum_discoverable_credentials_per_action': 50000,
            'maximum_node_count': 30,
            'maximum_total_credentials': 50000,
            'random_mode': "probabilistic",
            **default_kwargs
            }
)

# Local View Version

if 'CyberBattleToyCtfLocal-v0' in registry.env_specs:
    del registry.env_specs['CyberBattleToyCtfLocal-v0']

register(
    id='CyberBattleToyCtfLocal-v0',
    cyberbattle_env_identifiers=toy_ctf.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.local.cyberbattle_toyctf:CyberBattleToyCtf',
    kwargs={'defender_agent': None,
            'attacker_goal': AttackerGoal(own_atleast=6),
            'defender_goal': DefenderGoal(eviction=True),
            **default_kwargs
            },
    # max_episode_steps=2600,
)

if 'CyberBattleTinyLocal-v0' in registry.env_specs:
    del registry.env_specs['CyberBattleTinyLocal-v0']

register(
    id='CyberBattleTinyLocal-v0',
    cyberbattle_env_identifiers=toy_ctf.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.local.cyberbattle_tiny:CyberBattleTiny',
    kwargs={'defender_agent': None,
            'attacker_goal': AttackerGoal(own_atleast=6),
            'defender_goal': DefenderGoal(eviction=True),
            **default_kwargs
            },
    # max_episode_steps=2600,
)

if 'CyberBattleRandomLocal-v0' in registry.env_specs:
    del registry.env_specs['CyberBattleRandomLocal-v0']

register(
    id='CyberBattleRandomLocal-v0',
    cyberbattle_env_identifiers=generate_network.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.local.cyberbattle_random:CyberBattleRandom',
    kwargs={
            'verbose': False,
            **default_kwargs
    }
)


if 'CyberBattleRandomLocalVerbose-v0' in registry.env_specs:
    del registry.env_specs['CyberBattleRandomLocal-v0']

register(
    id='CyberBattleRandomLocalVerbose-v0',
    cyberbattle_env_identifiers=generate_network.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.local.cyberbattle_random:CyberBattleRandom',
    kwargs={
            'verbose': True,
            **default_kwargs
    })


if 'CyberBattleChainLocal-v0' in registry.env_specs:
    del registry.env_specs['CyberBattleChainLocal-v0']

register(
    id='CyberBattleChainLocal-v0',
    cyberbattle_env_identifiers=chainpattern.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.local.cyberbattle_chain:CyberBattleChain',
    kwargs={'size': 50,
            'defender_agent': None,
            'attacker_goal': AttackerGoal(own_atleast_percent=1.0),
            'defender_goal': DefenderGoal(eviction=True),
            'winning_reward': 5000.0,
            'losing_reward': 0.0,
            'absolute_reward': True,
            **default_kwargs
            },
    reward_threshold=2200,
)

ad_envs = [f"ActiveDirectoryLocal-v{i}" for i in range(0, 10)]
for (index, env) in enumerate(ad_envs):
    if env in registry.env_specs:
        del registry.env_specs[env]

    register(
        id=env,
        cyberbattle_env_identifiers=generate_ad.ENV_IDENTIFIERS,
        entry_point='cyberbattle._env.local.active_directory:CyberBattleActiveDirectory',
        kwargs={
            'seed': index,
            **default_kwargs
        }
    )

if 'ActiveDirectoryTinyLocal-v0' in registry.env_specs:
    del registry.env_specs['ActiveDirectoryTinyLocal-v0']
register(
    id='ActiveDirectoryTinyLocal-v0',
    cyberbattle_env_identifiers=chainpattern.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.local.active_directory:CyberBattleActiveDirectoryTiny',
)

# Probabilistic Local View Version

if 'CyberBattleToyCtfLocalProbabilistic-v0' in registry.env_specs:
    del registry.env_specs['CyberBattleToyCtfLocalProbabilistic-v0']

register(
    id='CyberBattleToyCtfLocalProbabilistic-v0',
    cyberbattle_env_identifiers=toy_ctf.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.local.cyberbattle_toyctf:CyberBattleToyCtf',
    kwargs={'defender_agent': None,
            'attacker_goal': AttackerGoal(own_atleast=6),
            'defender_goal': DefenderGoal(eviction=True),
            'random_mode': "probabilistic",
            **default_kwargs
            },
    # max_episode_steps=2600,
)

if 'CyberBattleTinyLocalProbabilistic-v0' in registry.env_specs:
    del registry.env_specs['CyberBattleTinyLocalProbabilistic-v0']

register(
    id='CyberBattleTinyLocalProbabilistic-v0',
    cyberbattle_env_identifiers=toy_ctf.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.local.cyberbattle_tiny:CyberBattleTiny',
    kwargs={'defender_agent': None,
            'attacker_goal': AttackerGoal(own_atleast=6),
            'defender_goal': DefenderGoal(eviction=True),
            'random_mode': "probabilistic",
            **default_kwargs
            },
    # max_episode_steps=2600,
)

if 'CyberBattleRandomLocalProbabilistic-v0' in registry.env_specs:
    del registry.env_specs['CyberBattleRandomLocalProbabilistic-v0']

register(
    id='CyberBattleRandomLocalProbabilistic-v0',
    cyberbattle_env_identifiers=generate_network.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.local.cyberbattle_random:CyberBattleRandom',
    kwargs={'random_mode': "probabilistic", **default_kwargs}
)

if 'CyberBattleChainLocalProbabilistic-v0' in registry.env_specs:
    del registry.env_specs['CyberBattleChainLocalProbabilistic-v0']

register(
    id='CyberBattleChainLocalProbabilistic-v0',
    cyberbattle_env_identifiers=chainpattern.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.local.cyberbattle_chain:CyberBattleChain',
    kwargs={'size': 50,
            'defender_agent': None,
            'attacker_goal': AttackerGoal(own_atleast_percent=1.0),
            'defender_goal': DefenderGoal(eviction=True),
            'winning_reward': 5000.0,
            'losing_reward': 0.0,
            'absolute_reward': True,
            'random_mode': "probabilistic",
            **default_kwargs
            },
    reward_threshold=2200,
)

ad_envs = [f"ActiveDirectoryLocalProbabilistic-v{i}" for i in range(0, 10)]
for (index, env) in enumerate(ad_envs):
    if env in registry.env_specs:
        del registry.env_specs[env]

    register(
        id=env,
        cyberbattle_env_identifiers=generate_ad.ENV_IDENTIFIERS,
        entry_point='cyberbattle._env.local.active_directory:CyberBattleActiveDirectory',
        kwargs={
            'seed': index,
            'random_mode': "probabilistic",
            **default_kwargs
        }
    )

if 'ActiveDirectoryTinyLocalProbabilistic-v0' in registry.env_specs:
    del registry.env_specs['ActiveDirectoryTinyLocalProbabilistic-v0']
register(
    id='ActiveDirectoryTinyLocalProbabilistic-v0',
    cyberbattle_env_identifiers=chainpattern.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.local.active_directory:CyberBattleActiveDirectoryTiny',
    kwargs={'random_mode': "probabilistic",   **default_kwargs}
)

# FullyObservable Local View Version

if 'CyberBattleToyCtfLocalObservable-v0' in registry.env_specs:
    del registry.env_specs['CyberBattleToyCtfLocalObservable-v0']

register(
    id='CyberBattleToyCtfLocalObservable-v0',
    cyberbattle_env_identifiers=toy_ctf.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.local.cyberbattle_toyctf:CyberBattleToyCtf',
    kwargs={'defender_agent': None,
            'attacker_goal': AttackerGoal(own_atleast=6),
            'defender_goal': DefenderGoal(eviction=True),
            'partial_observability': False,
            **default_kwargs
            },
    # max_episode_steps=2600,
)

if 'CyberBattleTinyLocalObservable-v0' in registry.env_specs:
    del registry.env_specs['CyberBattleTinyLocalObservable-v0']

register(
    id='CyberBattleTinyLocalObservable-v0',
    cyberbattle_env_identifiers=toy_ctf.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.local.cyberbattle_tiny:CyberBattleTiny',
    kwargs={'defender_agent': None,
            'attacker_goal': AttackerGoal(own_atleast=6),
            'defender_goal': DefenderGoal(eviction=True),
            'partial_observability': False,
            **default_kwargs
            },
    # max_episode_steps=2600,
)

if 'CyberBattleRandomLocalObservable-v0' in registry.env_specs:
    del registry.env_specs['CyberBattleRandomLocalObservable-v0']

register(
    id='CyberBattleRandomLocalObservable-v0',
    cyberbattle_env_identifiers=generate_network.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.local.cyberbattle_random:CyberBattleRandom',
    kwargs={'partial_observability': False, **default_kwargs}
)

if 'CyberBattleChainLocalObservable-v0' in registry.env_specs:
    del registry.env_specs['CyberBattleChainLocalObservable-v0']

register(
    id='CyberBattleChainLocalObservable-v0',
    cyberbattle_env_identifiers=chainpattern.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.local.cyberbattle_chain:CyberBattleChain',
    kwargs={'size': 50,
            'defender_agent': None,
            'attacker_goal': AttackerGoal(own_atleast_percent=1.0),
            'defender_goal': DefenderGoal(eviction=True),
            'winning_reward': 5000.0,
            'losing_reward': 0.0,
            'absolute_reward': True,
            'partial_observability': False,
            **default_kwargs
            },
    reward_threshold=2200,
)

ad_envs = [f"ActiveDirectoryLocalObservable-v{i}" for i in range(0, 10)]
for (index, env) in enumerate(ad_envs):
    if env in registry.env_specs:
        del registry.env_specs[env]

    register(
        id=env,
        cyberbattle_env_identifiers=generate_ad.ENV_IDENTIFIERS,
        entry_point='cyberbattle._env.local.active_directory:CyberBattleActiveDirectory',
        kwargs={
            'seed': index,
            'partial_observability': False,
            **default_kwargs
        }
    )

if 'ActiveDirectoryTinyLocalObservable-v0' in registry.env_specs:
    del registry.env_specs['ActiveDirectoryTinyLocalObservable-v0']
register(
    id='ActiveDirectoryTinyLocalObservable-v0',
    cyberbattle_env_identifiers=chainpattern.ENV_IDENTIFIERS,
    entry_point='cyberbattle._env.local.active_directory:CyberBattleActiveDirectoryTiny',
    kwargs={'partial_observability': False, **default_kwargs}
)
