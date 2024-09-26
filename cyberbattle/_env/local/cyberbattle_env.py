# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Anatares OpenGym Environment"""

import time
import logging
from typing import NamedTuple, Optional, Tuple, List, Dict, TypedDict
import numpy
import random
import gym as gym
from gym import spaces
from gym.utils import seeding
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import sys
import os
import copy
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", ".."))
sys.path.insert(0, project_root)
from cyberbattle._env.defender import DefenderAgent
from cyberbattle.simulation.model import PortName, PrivilegeLevel
from cyberbattle.simulation import commandcontrol, model, actions

LOGGER = logging.getLogger(__name__)

class OutOfBoundIndexError(Exception):
    """The agent attempted to reference an entity (node or a vulnerability) with an invalid index"""

Observation = TypedDict(
    'Observation', {
        # source node feature vector
        'source_node': numpy.ndarray,
        # target node feature vector
        'target_node': numpy.ndarray,
        # global features
        'global_features': numpy.ndarray,
    })

StepInfo = TypedDict(
    'StepInfo', {
        'description': str,
        'duration_in_ms': float,
        'step_count': int,
        'network_availability': float,
        'outcome': model.VulnerabilityOutcome
    })


class AttackerGoal(NamedTuple):
    """Define conditions to be simultanesouly met for the attacker to win.
    If field values are not specified the default is to target full ownership
    of the network nodes.
    """
    # Include goal to reach at least the specifed cumulative total reward after
    reward: float = 0.0
    # Include goal to bring the availability to lower that the specified SLA value
    low_availability: float = 1.0
    # Include goal to own at least the specified number of nodes.
    own_atleast: int = 0
    # Include goal to own at least the specified percentage of the network nodes.
    # Set to 1.0 to define goal as the ownership of all network nodes.
    own_atleast_percent: float = 1.0


class DefenderGoal(NamedTuple):
    """Define conditions to be simultanesouly met for the defender to win."""
    # Met if attacker is evicted from all the network nodes
    eviction: bool


class DefenderConstraint(NamedTuple):
    """Define constraints to be maintained by the defender at all time."""
    maintain_sla: float


class CyberBattleEnv(gym.Env):
    """OpenAI Gym environment interface to the CyberBattle simulation.

    # Actions

        Run a local attack:            `(local_vulnerability_to_exploit)`
        Run a remote attack command:   `(remote_vulnerability_to_exploit)`
        Connect to a remote node:      `(target_port)`
        Simplification: try all credentials gathered for now

    # Observation
       See type `Observation` for a full description of the observation space.
       It includes:
       - Source node feature vector
       - Target node feature vector
       - Global features

    # Termination

    The simulation ends if either the attacker reaches its goal (e.g. full network ownership),
    the defender reaches its goal (e.g. full eviction of the attacker)
    or if one of the defender's constraints is not met (e.g. SLA).
    """

    metadata = {'render.modes': ['human']}
    # number of distinct privilege levels
    privilege_levels = model.PrivilegeLevel.MAXIMUM + 1

    @property
    def environment(self) -> model.Environment:
        return self.__environment

    @property
    def name(self) -> str:
        return "CyberBattleEnv"

    @property
    def identifiers(self) -> model.Identifiers:
        return self.__initial_environment.identifiers

    # Function to reset the network to its initial state and choose an entry node
    def __reset_environment(self) -> None:
        self.__environment: model.Environment = copy.deepcopy(self.__initial_environment)
        # Pick a random node as the agent entry node
        if self.random_starter_node:
            entry_node_index = random.randrange(len(self.__environment.network.nodes))
            for node in self.__environment.network.nodes:
                self.get_node(node).agent_installed = False
            entry_node_id, _ = list(self.__environment.network.nodes(data=True))[entry_node_index]
            if self.env_type == "random_env":
                self.__environment.network.nodes[entry_node_id]['data'].services = []
                self.__environment.network.nodes[entry_node_id]['data'].value = 0
                # property present only in random environment
                self.__environment.network.nodes[entry_node_id]['data'].properties = ["breach_node"]
            self.__environment.network.nodes[entry_node_id]['data'].agent_installed = True
        else:
            entry_node_id = None
            for node in self.__environment.network.nodes:
                if self.get_node(node).agent_installed == True:
                    entry_node_id = node
                    break
            if self.env_type == "random_env":
                if not entry_node_id:
                    entry_node_id = '0'
                self.__environment.network.nodes[entry_node_id]['data'].services = []
                self.__environment.network.nodes[entry_node_id]['data'].value = 0
                # property present only in random environment
                self.__environment.network.nodes[entry_node_id]['data'].properties = ["breach_node"]
        self.__environment.network.nodes[entry_node_id]['data'].agent_installed = True
        self.__discovered_nodes: List[model.NodeID] = []
        self.__owned_nodes_indices_cache: Optional[List[int]] = None
        self.__credential_cache: List[model.CachedCredential] = []
        self.__episode_rewards: List[float] = []
        # The actuator used to execute actions in the simulation environment
        self._actuator = actions.AgentActions(self.__environment, throws_on_invalid_actions=self.__throws_on_invalid_actions,
                                              value_coefficient=self.value_coefficient, cost_coefficient=self.cost_coefficient,
                                              property_discovered_coefficient=self.property_discovered_coefficient,
                                              credential_discovered_coefficient=self.credential_discovered_coefficient,
                                              node_discovered_coefficient=self.node_discovered_coefficient,
                                              first_success_attack_coefficient=self.first_success_attack_coefficient,
                                              penalty_dict=self.penalties
                                              )
        self._defender_actuator = actions.DefenderAgentActions(self.__environment)

        self.__stepcount = 0
        self.__start_time = time.time()
        self.__done = False
        self.__discovered_nodes.append(entry_node_id)
        self.source_node_index = entry_node_id
        self.target_node_index = self.__pick_random_discovered_node_id_with_probabilities(self.owned_probability, self.not_owned_probability)

    def validate_environment(self, environment: model.Environment):
        """Validate that the size of the network and associated constants fits within
        the dimensions bounds set for the CyberBattle gym environment"""
        assert environment.identifiers.ports
        assert environment.identifiers.properties
        assert environment.identifiers.local_vulnerabilities
        assert environment.identifiers.remote_vulnerabilities


        referenced_ports = model.collect_ports_from_environment(environment)
        undefined_ports = set(referenced_ports).difference(environment.identifiers.ports)
        if undefined_ports:
            raise ValueError(f"The network has references to undefined port names: {undefined_ports}")

        referenced_properties = model.collect_properties_from_nodes(model.iterate_network_nodes(environment.network))
        undefined_properties = set(referenced_properties).difference(environment.identifiers.properties)
        if undefined_properties:
            raise ValueError(f"The network has references to undefined property names: {undefined_properties}")

        local_vulnerabilities = \
            model.collect_vulnerability_ids_from_nodes_bytype(
                environment.nodes(),
                environment.vulnerability_library,
                model.VulnerabilityType.LOCAL
            )

        undefined_local_vuln = set(local_vulnerabilities).difference(environment.identifiers.local_vulnerabilities)
        if undefined_local_vuln:
            raise ValueError(f"The network has references to undefined local"
                             f" vulnerability names: {undefined_local_vuln}")

        remote_vulnerabilities = \
            model.collect_vulnerability_ids_from_nodes_bytype(
                environment.nodes(),
                environment.vulnerability_library,
                model.VulnerabilityType.REMOTE
            )

        undefined_remote_vuln = set(remote_vulnerabilities).difference(environment.identifiers.remote_vulnerabilities)
        if undefined_remote_vuln:
            raise ValueError(f"The network has references to undefined remote"
                             f" vulnerability names: {undefined_remote_vuln}")

    def __init__(self,
                 initial_environment: model.Environment,
                 env_type="random_env",
                 defender_agent: Optional[DefenderAgent] = None,
                 attacker_goal: Optional[AttackerGoal] = AttackerGoal(own_atleast_percent=1.0),
                 defender_goal=DefenderGoal(eviction=True),
                 defender_constraint=DefenderConstraint(maintain_sla=0.0),
                 winning_reward=0,
                 losing_reward=0,
                 renderer='',
                 throws_on_invalid_actions=True,
                 random_starter_node=True,
                 max_num_iterations = 100,
                 random_mode = "normal",
                 absolute_reward = False,
                 visible_node_features = None,
                 visible_global_features = None,
                 owned_probability=0,
                 partial_observability=True,
                 stop_at_goal_reached=False,
                 value_coefficient=1,
                 cost_coefficient=1,
                 property_discovered_coefficient=2,
                 credential_discovered_coefficient=3,
                 node_discovered_coefficient=5,
                 first_success_attack_coefficient=7,
                 penalties = None
                 ):
        """Arguments
        ===========
        environment               - The CyberBattle network simulation environment
        attacker_goal             - Target goal for the attacker to win and stop the simulation.
        defender_goal             - Target goal for the defender to win and stop the simulation.
        defender_constraint       - Constraint to be maintain by the defender to keep the simulation running.
        winning_reward            - Reward granted to the attacker if the simulation ends because the attacker's goal is reached.
        losing_reward             - Reward granted to the attacker if the simulation ends because the Defender's goal is reached.
        renderer                  - the matplotlib renderer (e.g. 'png')
        observation_padding       - whether to padd all the observation fields to their maximum size. For instance this will pad the credential matrix
                                    to fit in `maximum_node_count` rows. Turn on this flag for gym agent that expects observations of fixed sizes.
        throws_on_invalid_actions - whether to raise an exception if the step function attempts an invalid action (e.g., running an attack from a node that's not owned)
                                    if set to False a negative reward is returned instead.
        """


        self.fig = None
        self.render_index = 0
        self.env_type = env_type
        self.max_num_iterations = max_num_iterations
        self.validate_environment(initial_environment)

        self.stop_at_goal_reached = stop_at_goal_reached
        self.__attacker_goal: Optional[AttackerGoal] = attacker_goal
        self.__defender_goal: DefenderGoal = defender_goal
        self.__defender_constraint: DefenderConstraint = defender_constraint
        self.__WINNING_REWARD = winning_reward
        self.__LOSING_REWARD = losing_reward
        self.__renderer = renderer
        self.__throws_on_invalid_actions = throws_on_invalid_actions
        self.random_starter_node = random_starter_node
        self.viewer = None
        self.done = False
        self.absolute_reward = absolute_reward
        self.owned_probability = owned_probability
        self.not_owned_probability= 1-owned_probability
        self.random_mode = random_mode
        self.partial_observability = partial_observability

        self.value_coefficient = value_coefficient
        self.cost_coefficient = cost_coefficient
        self.property_discovered_coefficient = property_discovered_coefficient
        self.credential_discovered_coefficient = credential_discovered_coefficient
        self.node_discovered_coefficient = node_discovered_coefficient
        self.first_success_attack_coefficient = first_success_attack_coefficient
        self.penalties = penalties

        self.owned_nodes_list = []
        self.discovered_nodes_list = []
        self.discovered_credentials_list = []
        self.episode_rewards_list = []

        self.visible_node_features = visible_node_features
        self.visible_global_features = visible_global_features

        self.__initial_environment: model.Environment = initial_environment

        self.num_credentials = self.__initial_environment.num_credentials

        self.__defender_agent = defender_agent

        self.__reset_environment()

        self.__node_count = len(initial_environment.network.nodes.items())

        self.local_vulnerabilities = self.identifiers.local_vulnerabilities
        self.remote_vulnerabilities = self.identifiers.remote_vulnerabilities
        self.ports = self.identifiers.ports
        self.properties = self.identifiers.properties

        self.local_vulnerabilities_count = len(self.local_vulnerabilities)
        self.remote_vulnerabilities_count = len(self.remote_vulnerabilities)
        self.property_count = len(self.properties)
        self.port_count = len(self.ports)

        self.num_nodes = 0
        for _ in self.__environment.nodes():
            self.num_nodes += 1

        # ------- ACTION SPACE -------
        # local vulnerability OR remote vulnerability OR connect
        self.action_space = spaces.Discrete(self.local_vulnerabilities_count + self.remote_vulnerabilities_count + self.port_count)

        # ------- OBSERVATION SPACE -------
        # --- NODE FEATURE ---
        value_space = spaces.Box(low=0, high=1000, shape=(1,), dtype=numpy.float32)
        privilege_level_space = spaces.Discrete(self.privilege_levels)
        status_space = spaces.Discrete(len(model.MachineStatus))
        reimageable_space = spaces.Discrete(2)
        sla_weight_space = spaces.Box(low=0, high=1, shape=(1,), dtype=numpy.float32)
        listening_service_space = spaces.Tuple([
            spaces.Discrete(2),  # running
            spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=float), # sla weight
            # spaces.Discrete(max_port + 1)  given already by the index in the array
        ])
        listening_services_array_space = spaces.Tuple([listening_service_space] * self.port_count)

        if self.partial_observability:
            vulnerability_types = 4 # type + 2 unknowns
            vulnerability_outcomes = 12 # outcome: 10 several possible types of outcomes may be discovered during simulation + 2 unknowns
        else:
            vulnerability_types = 2
            vulnerability_outcomes = 10

        vulnerability_space = spaces.Tuple([
            spaces.Discrete(vulnerability_types),
            spaces.Discrete(vulnerability_outcomes),
            spaces.Box(low=0.0, high=100, shape=(1,), dtype=float)  # cost
        ])
        vulnerabilities_array_space = spaces.Tuple([vulnerability_space] * (self.local_vulnerabilities_count + self.remote_vulnerabilities_count))
        property_array_space = spaces.Tuple([spaces.Discrete(2) for _ in range(self.property_count)])

        firewall_config_space = spaces.Tuple([
            spaces.Discrete(2),  # incoming/outgoing
            spaces.Discrete(2) # allow/block
            # TODO: may expand to unknown
        ])
        firewall_config_array_space = spaces.Tuple([firewall_config_space] * 2*self.port_count)

        # --- GLOBAL FEATURES ---
        number_discovered_nodes_space = spaces.Box(low=0, high=self.__node_count, shape=(1,), dtype=numpy.int32) # overall
        lateral_move_space = spaces.Discrete(2) # whether last lateral move was successful
        discovered_data_space = spaces.Discrete(2) # whether customer data were just discovered
        probe_result_space = spaces.Discrete(3) # whether last probe was successful and how
        escalation_result_space = spaces.Discrete(2) # whether last escalation was successful
        number_discovered_credentials_space = spaces.Box(low=0, high=self.num_credentials, shape=(1,), dtype=numpy.int32) # overall

        dict_observation_space = {
            # source node feature vector
            'source_node_value': value_space,
            'source_node_privilege_level': privilege_level_space,
            'source_node_status': status_space,
            'source_node_reimageable': reimageable_space,
            'source_node_sla_weight': sla_weight_space,
            'source_node_listening_services_array': listening_services_array_space,
            'source_node_vulnerabilities_array': vulnerabilities_array_space,
            'source_node_property_array': property_array_space,
            'source_node_firewall_config_array': firewall_config_array_space,
            # target node feature vector
            'target_node_value': value_space,
            'target_node_privilege_level': privilege_level_space,
            'target_node_status': status_space,
            'target_node_reimageable': reimageable_space,
            'target_node_sla_weight': sla_weight_space,
            'target_node_listening_services_array': listening_services_array_space,
            'target_node_vulnerabilities_array': vulnerabilities_array_space,
            'target_node_property_array': property_array_space,
            'target_node_firewall_config_array': firewall_config_array_space,
            # global_features
            'global_features_number_discovered_nodes': number_discovered_nodes_space,
            'global_features_lateral_move': lateral_move_space,
            'global_features_customer_data_found': discovered_data_space,
            'global_features_probe_result': probe_result_space,
            'global_features_escalation': escalation_result_space,
            'global_features_number_discovered_credentials': number_discovered_credentials_space
        }

        visible_features = self.concatenate_or_none(self.visible_node_features, self.visible_global_features)
        visible_dict = self.hide_features(dict_observation_space, visible_features)

        self.observation_space = spaces.Dict(visible_dict)
        self.observation_space = self.flatten_space_dict_with_arrays(self.observation_space)

        # ----- REWARD -----
        # reward_range: A tuple corresponding to the min and max possible rewards
        self.reward_range = (-float('inf'), float('inf'))

    def __index_to_local_vulnerabilityid(self, vulnerability_index: int) -> model.VulnerabilityID:
        """Return the local vulnerability identifier from its internal encoding index"""
        return self.__initial_environment.identifiers.local_vulnerabilities[vulnerability_index]

    def __vulnerabilityid_to_index(self, vulnerability_id: model.VulnerabilityID) -> int:
        """Return the integer index from a VulnerabilityID"""
        if vulnerability_id in self.__initial_environment.identifiers.local_vulnerabilities:
            return self.__initial_environment.identifiers.local_vulnerabilities.index(vulnerability_id)
        elif vulnerability_id in self.__initial_environment.identifiers.remote_vulnerabilities:
            return self.__initial_environment.identifiers.remote_vulnerabilities.index(vulnerability_id) + self.local_vulnerabilities_count
        else:
            return -1

    def __index_to_remote_vulnerabilityid(self, vulnerability_index: int) -> model.VulnerabilityID:
        """Return the remote vulnerability identifier from its internal encoding index"""
        return self.__initial_environment.identifiers.remote_vulnerabilities[vulnerability_index]

    def __index_to_port_name(self, port_index: int) -> model.PortName:
        """Return the port name identifier from its internal encoding index"""
        return self.__initial_environment.identifiers.ports[port_index]

    def __portname_to_index(self, port_name: model.PortName) -> int:
        """Return the integer index from a PortName"""
        if port_name in self.__initial_environment.identifiers.ports:
            return self.__initial_environment.identifiers.ports.index(port_name)
        else:
            return -1

    def __property_to_index(self, property_name: model.PropertyName) -> int:
        """Return the integer index from a PortName"""
        if property_name in self.__initial_environment.identifiers.properties:
            return self.__initial_environment.identifiers.properties.index(property_name)
        else:
            return -1

    def get_property_index(self, property_name: model.PropertyName) -> int:
        """Return the integer index from a PortName"""
        return self.__property_to_index(property_name)


    def __internal_node_id_from_external_node_index(self, node_external_index: int) -> model.NodeID:
        """"Return the internal environment node ID corresponding to the specified
        external node index that is exposed to the Gym agent
                0 -> ID of inital node
                1 -> ID of first discovered node
                ...

        """
        # Ensures that the specified node is known by the agent
        if node_external_index < 0:
            raise OutOfBoundIndexError(f"Node index must be positive, given {node_external_index}")

        length = len(self.__discovered_nodes)
        if node_external_index >= length:
            raise OutOfBoundIndexError(
                f"Node index ({node_external_index}) is invalid; only {length} nodes discovered so far.")

        node_id = self.__discovered_nodes[node_external_index]
        return node_id

    def __find_external_index(self, node_id: model.NodeID) -> int:
        """Find the external index associated with the specified node ID"""
        return self.__discovered_nodes.index(node_id)

    def __agent_owns_node(self, node_id: model.NodeID) -> bool:
        node = self.__environment.get_node(node_id)
        owned: bool = node.agent_installed
        return owned

    def get_nodes(self):
        return self.__environment.nodes()

    def get_vulnerability_index(self, vulnerability_id: model.VulnerabilityID) -> int:
        return self.__vulnerabilityid_to_index(vulnerability_id)

    def get_actuator(self):
        return self._actuator

    def get_episode_reward(self):
        return numpy.sum(self.__episode_rewards)

    def get_port_index(self, port_name: PortName) -> int:
        return self.__portname_to_index(port_name)

    def get_local_attacks_count(self):
        return self.local_vulnerabilities_count

    def get_remote_attacks_count(self):
        return self.remote_vulnerabilities_count

    def get_port_count(self):
        return self.port_count

    def get_property_count(self):
        return self.property_count

    def get_node(self, node_id):
        return self.__environment.get_node(node_id)



    # Use the actuator to execute action and gather the result
    def __execute_action(self, kind: int, action_index: int) -> actions.ActionResult:
        if kind == 0: # local_vulnerability
            return self._actuator.exploit_local_vulnerability(
                self.source_node_index,
                self.__index_to_local_vulnerabilityid(action_index))
        elif kind == 1: # remote_vulnerability
            return self._actuator.exploit_remote_vulnerability(
                self.source_node_index,
                self.target_node_index,
                self.__index_to_remote_vulnerabilityid(action_index))
        elif kind == 2: # connect
            if len(self.__credential_cache) == 0:
                #print("Connect but with empty cache")
                return actions.ActionResult(reward=-1, outcome=None)
            for credential in self.__credential_cache:
                result = self._actuator.connect_to_remote_machine(
                    self.source_node_index,
                    self.target_node_index,
                    self.__index_to_port_name(action_index),
                    credential.credential)
                if result.outcome:
                    self.source_node_index = self.target_node_index
                    return result
            return result

    # Create a blank observation conforming to the observation space
    def __get_blank_observation(self):
        blank_observation = {
            'source_node': {
                'firewall_config_array': [(0, 0) for _ in range(2*self.port_count)],
                'listening_services_array': [(0, numpy.array([0.0], dtype=numpy.float32)) for _ in
                                             range(self.port_count)],
                'privilege_level': 0,
                'property_array': [0 for _ in range(self.property_count)],
                'reimageable': 0,
                'sla_weight': numpy.array([0.0], dtype=numpy.float32),
                'status': 0,
                'value': numpy.array([0.0], dtype=numpy.float32),
                'vulnerabilities_array': [(0, 0, numpy.array([0.0], dtype=numpy.float64)) for _ in
                                                range(self.local_vulnerabilities_count + self.remote_vulnerabilities_count)],
            },
            'target_node': {
                'firewall_config_array': [(0, 0) for _ in range(2*self.port_count)],
                'listening_services_array': [(0, numpy.array([0.0], dtype=numpy.float32)) for _ in
                                             range(self.port_count)],
                'privilege_level': 0,
                'property_array': [0 for _ in range(self.property_count)],
                'reimageable': 0,
                'sla_weight': numpy.array([0.0], dtype=numpy.float32),
                'status': 0,
                'value': numpy.array([0.0], dtype=numpy.float32),
                'vulnerabilities_array': [(0, 0, numpy.array([0.0], dtype=numpy.float64)) for _ in
                                                range(self.local_vulnerabilities_count + self.remote_vulnerabilities_count)],
            },
            'global_features': {
                'customer_data_found': 0,
                'escalation': 0,
                'lateral_move': 0,
                'number_discovered_credentials': numpy.array([0], dtype=numpy.int32),
                'number_discovered_nodes': numpy.array([0], dtype=numpy.int32),
                'probe_result': 0
            }
        }

        return blank_observation

    # Flatten observation into Discrete or Box supported types as requested by StableBaselines
    def flatten_dict_with_arrays(self,input_dict):
        flattened_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, dict):
                # If the value is a dictionary, flatten it recursively
                flattened_subdict = self.flatten_dict_with_arrays(value)
                flattened_dict.update(
                    {f"{key}_{sub_key}": sub_value for sub_key, sub_value in flattened_subdict.items()})
            elif isinstance(value, list):
                # If the value is a list, flatten it to individual components
                for i, sub_value in enumerate(value):
                    if isinstance(sub_value, tuple):
                        for j, inner_sub_value in enumerate(sub_value):
                            flattened_dict[f"{key}_{i}_{j}"] = inner_sub_value
                    else:
                        flattened_dict[f"{key}_{i}"] = sub_value
            else:
                # If the value is not a dictionary or list, include it as is
                flattened_dict[key] = value
        return flattened_dict

    # Flatten Discrete and Box Spaces in hierarchies in pure single hierarchy of Discrete and Box Spaces
    def flatten_space_dict_with_arrays(self, input_dict):
        flattened_dict = {}

        for key, value in input_dict.items():
            if isinstance(value, spaces.Dict):
                # If the value is a dictionary, flatten it recursively
                flattened_subdict = self.flatten_space_dict_with_arrays(value)
                flattened_dict.update(
                    {f"{key}_{sub_key}": sub_value for sub_key, sub_value in flattened_subdict.items()})
            elif isinstance(value, spaces.Tuple):
                # If the value is a tuple, flatten it to individual components
                for i, sub_space in enumerate(value):
                    if isinstance(sub_space, spaces.Tuple):
                        # If the element of the tuple is another tuple, flatten it again
                        for j, nested_sub_space in enumerate(sub_space):
                            if isinstance(nested_sub_space, spaces.Dict):
                                # Recursively flatten nested dictionary in the tuple
                                nested_flattened_dict = self.flatten_space_dict_with_arrays(
                                    {f"{key}_{i}_{j}_{sub_key}": sub_value for sub_key, sub_value in
                                     nested_sub_space.items()})
                                flattened_dict.update(nested_flattened_dict)
                            elif isinstance(nested_sub_space, spaces.Box):
                                # Flatten box components
                                flattened_dict[f"{key}_{i}_{j}"] = nested_sub_space
                            elif isinstance(nested_sub_space, spaces.Discrete):
                                # Flatten discrete components
                                flattened_dict[f"{key}_{i}_{j}"] = nested_sub_space
                    elif isinstance(sub_space, spaces.Dict):
                        # Recursively flatten nested dictionary in the tuple
                        nested_flattened_dict = self.flatten_space_dict_with_arrays(
                            {f"{key}_{i}_{sub_key}": sub_value for sub_key, sub_value in sub_space.items()})
                        flattened_dict.update(nested_flattened_dict)
                    elif isinstance(sub_space, spaces.Box):
                        # Flatten box components
                        flattened_dict.update(
                            {f"{key}_{i}_{sub_key}": sub_value for sub_key, sub_value in
                             zip(sub_space.spaces.keys(), sub_space)}
                        )
                    elif isinstance(sub_space, spaces.Discrete):
                        # Flatten discrete components
                        flattened_dict[f"{key}_{i}"] = sub_space
            else:
                # If the value is not a dictionary or tuple, include it with indices
                flattened_dict[key] = value

        return spaces.Dict(flattened_dict)

    # Convert node information to the proper feature vector used in the observation space
    def __convert_node_info_to_observation(self, node_info, node_id) -> Dict:
        listening_services_array = [
            (0, numpy.array([0.0], dtype=numpy.float32)) for _ in range(self.port_count)
        ]

        # OWNED or DISCOVERED: Services provided known
        for service in node_info.services:
            if self.__portname_to_index(service.name) != -1:
                listening_services_array[self.__portname_to_index(service.name)] = (
                    int(service.running),
                    numpy.array([service.sla_weight], dtype=numpy.float32)
                )
        if self.partial_observability:
            if self.__environment.get_node(node_id).agent_installed == True:
                # OWNED: Vulnerabilities not present by default
                vulnerabilities_array = [
                    (1, 1, numpy.array([0.0], dtype=numpy.float64)) for _ in range(self.local_vulnerabilities_count + self.remote_vulnerabilities_count)
                ]
            else:
                # DISCOVERED: Vulnerabilities not known by default
                vulnerabilities_array = [
                    (0, 0, numpy.array([0.0], dtype=numpy.float64)) for _ in range(self.local_vulnerabilities_count + self.remote_vulnerabilities_count)
                ]
            # DISCOVERED: Set discovered vulnerabilities
            # OWNED: Set all present vulnerabilities
            for vulnerability_id, vulnerability_info in node_info.vulnerabilities.items():
                if self.__vulnerabilityid_to_index(
                    vulnerability_id) in self._actuator.get_discovered_vulnerabilities(node_id):
                    if self.__vulnerabilityid_to_index(vulnerability_id) != -1:
                        vulnerabilities_array[self.__vulnerabilityid_to_index(vulnerability_id)] = (
                            vulnerability_info.type.value + 2,
                            model.map_outcome_to_index(vulnerability_info.outcome) + 2,
                            numpy.array([vulnerability_info.cost], dtype=numpy.float64)
                        )

            # DISCOVERED: set vulnerabilities tested to be not present
            if self.__environment.get_node(node_id).agent_installed == False:
                for vulnerability_id in self._actuator.get_absence_discovered_vulnerabilities(node_id):
                    vulnerabilities_array[vulnerability_id] = (
                        1,
                        1,
                        numpy.array([0.0], dtype=numpy.float64)
                    )

        else:
            vulnerabilities_array = [
                (0, 0, numpy.array([0.0], dtype=numpy.float64)) for _ in
                range(self.local_vulnerabilities_count + self.remote_vulnerabilities_count)
            ]
            for vulnerability_id, vulnerability_info in node_info.vulnerabilities.items():
                if self.__vulnerabilityid_to_index(vulnerability_id) != -1:
                    vulnerabilities_array[self.__vulnerabilityid_to_index(vulnerability_id)] = (
                        vulnerability_info.type.value,
                        model.map_outcome_to_index(vulnerability_info.outcome),
                        numpy.array([vulnerability_info.cost], dtype=numpy.float64)
                    )

        property_array = [
            0 for _ in range(self.property_count)
        ]

        if self.partial_observability:
            # OWNED: all known
            # DISCOVERED: get only discovered properties
            for property in self._actuator.get_discovered_properties(node_id):
                property_array[property] = 1
        else:
            for property in node_info.properties:
                if self.__property_to_index(property) != -1:
                    property_array[self.__property_to_index(property)] = 1


        # DISCOVERED or OWNED: all known
        firewall_config_array = [
            (0, 0) for _ in range(2*self.port_count)
        ]

        for i in range(self.port_count, 2 * self.port_count):
                firewall_config_array[i] = (1, 0)

        for config in node_info.firewall.incoming:
            incoming_outgoing = 0
            permission = config.permission.value
            if self.__portname_to_index(config.port) != -1:
                firewall_config_array[self.__portname_to_index(config.port)] = (incoming_outgoing, permission)
        for config in node_info.firewall.outgoing:
            incoming_outgoing = 1
            permission = config.permission.value
            if self.__portname_to_index(config.port) != -1:
                firewall_config_array[self.port_count + self.__portname_to_index(config.port)] = (incoming_outgoing, permission)

        node_feature = {
                'firewall_config_array': firewall_config_array,
                'listening_services_array': listening_services_array,
                'privilege_level': int(node_info.privilege_level),
                'property_array': property_array,
                'reimageable': int(node_info.reimagable),
                'sla_weight': numpy.array([node_info.sla_weight], dtype=numpy.float32),
                'status': node_info.status.value,
                'value': numpy.array([node_info.value], dtype=numpy.float32),
                'vulnerabilities_array': vulnerabilities_array
        }
        return node_feature

    # Target node selection
    def __pick_random_discovered_node_id(self):
        available_nodes = list(filter(lambda node: node != self.source_node_index, self.__discovered_nodes))

        if available_nodes:
            # If there are available nodes, select a random one
            random_node = random.choice(available_nodes)
            return random_node
        else:
            # initial setup where no other node is discovered, only local vulnerabilities possible by giving the same feature vector twice in input
            return self.source_node_index

    # Target node selection with probabilities by group
    def __pick_random_discovered_node_id_with_probabilities(self, owned_prob, not_owned_prob):
        owned_nodes = [node_id for node_id, node_data in self.__environment.nodes() if node_data.agent_installed]
        owned_nodes = list(filter(lambda node: node != self.source_node_index, owned_nodes))

        discovered_only_nodes = [node_id for node_id, node_data in self.__environment.nodes() if
                                      node_id in self.__discovered_nodes and not node_data.agent_installed]
        discovered_only_nodes = list(filter(lambda node: node != self.source_node_index, discovered_only_nodes))

        if discovered_only_nodes:
            if not owned_prob: # case (0,1)
                random_node = random.choices(discovered_only_nodes)[0]
            else:
                # If there are available nodes, select a random one
                probabilities = [owned_prob if node in owned_nodes else not_owned_prob for node in discovered_only_nodes]
                random_node = random.choices(discovered_only_nodes, probabilities)[0]
            return random_node
        else:
            # Initial setup where no other node is discovered, only local vulnerabilities
            return self.source_node_index

    def __update_statistics(self):
        owned_nodes, discovered_nodes, _, num_nodes, percentage_discovered_credentials = self.get_statistics()
        self.owned_nodes_list.append(owned_nodes / num_nodes)
        self.discovered_nodes_list.append(discovered_nodes / num_nodes)
        self.discovered_credentials_list.append(percentage_discovered_credentials)
        self.episode_rewards_list.append(self.get_episode_reward())

    def get_statistics(self):
        owned_nodes = [node_id for node_id, node_data in self.__environment.nodes() if node_data.agent_installed]
        discovered_nodes = self.__discovered_nodes
        not_discovered_nodes = [node_id for node_id, node_data in self.__environment.nodes() if node_id not in self.__discovered_nodes and node_id != self.source_node_index]
        num_discovered_credentials = len(self.__credential_cache)
        return len(owned_nodes), len(discovered_nodes), len(not_discovered_nodes), self.num_nodes, num_discovered_credentials / self.num_credentials

    # Calculate reward based on outcome
    def __observation_reward_from_action_result(self, result: actions.ActionResult) -> Tuple[Observation, float]:
        outcome = result.outcome
        if isinstance(outcome, model.LeakedNodesId):
            # update discovered nodes
            newly_discovered_nodes_count = 0
            outcome.new_nodes = []
            for node in outcome.nodes:
                if node not in self.__discovered_nodes:
                    self.__discovered_nodes.append(node)
                    newly_discovered_nodes_count += 1
                    outcome.new_nodes.append(node)
            self.current_observation['global_features']['number_discovered_nodes'] += numpy.int32(newly_discovered_nodes_count)

        elif isinstance(outcome, model.LeakedCredentials):
            # update discovered nodes and credentials
            newly_discovered_nodes_count = 0
            newly_discovered_creds: List[Tuple[int, model.CachedCredential]] = []
            newly_discovered_nodes = []
            outcome.new_credentials = []
            for cached_credential in outcome.credentials:
                if cached_credential.node not in self.__discovered_nodes:
                    self.__discovered_nodes.append(cached_credential.node)
                    newly_discovered_nodes.append(cached_credential.node)
                    newly_discovered_nodes_count += 1

                if cached_credential not in self.__credential_cache:
                    self.__credential_cache.append(cached_credential)
                    added_credential_index = len(self.__credential_cache) - 1
                    newly_discovered_creds.append((added_credential_index, cached_credential))
                    outcome.new_credentials.append((added_credential_index, cached_credential))

            outcome.new_nodes = newly_discovered_nodes
            self.current_observation['global_features']['number_discovered_nodes'] += numpy.int32(newly_discovered_nodes_count)
            self.current_observation['global_features']['number_discovered_credentials'] += numpy.int32(len(newly_discovered_creds))

        elif isinstance(outcome, model.LateralMove):
            self.current_observation['global_features']['lateral_move'] = numpy.int32(1)
        elif isinstance(outcome, model.CustomerData):
            self.current_observation['global_features']['customer_data_found'] = numpy.int32(1)
        elif isinstance(outcome, model.ProbeSucceeded):
            self.current_observation['global_features']['probe_result'] = numpy.int32(2)
        elif isinstance(outcome, model.ProbeFailed):
            self.current_observation['global_features']['probe_result'] = numpy.int32(1)
        elif isinstance(outcome, model.PrivilegeEscalation):
            self.current_observation['global_features']['escalation'] = numpy.int32(outcome.level)

        self.target_node_index = self.__pick_random_discovered_node_id_with_probabilities(self.owned_probability, self.not_owned_probability)
        self.current_observation['source_node'] = self.__convert_node_info_to_observation(
            self.__environment.get_node(self.source_node_index), self.source_node_index)
        self.current_observation['target_node'] = self.__convert_node_info_to_observation(
            self.__environment.get_node(self.target_node_index), self.target_node_index)

        return self.current_observation, outcome, result.reward

    def get_owned_nodes_feature_vectors(self):
        owned_nodes = [node_id for node_id, node_data in self.__environment.nodes() if node_data.agent_installed]
        observation_vectors = []
        for node_id in owned_nodes:
            observation_vector = self.__convert_node_info_to_observation(
                self.__environment.get_node(node_id), node_id)
            observation_vectors.append((node_id,observation_vector))
        return observation_vectors

    def get_discovered_not_owned_nodes_feature_vectors(self):
        discovered_not_owned_nodes = [node_id for node_id, node_data in self.__environment.nodes() if node_id in self.__discovered_nodes and not node_data.agent_installed]
        observation_vectors = []
        for node_id in discovered_not_owned_nodes:
            observation_vector = self.__convert_node_info_to_observation(
                self.__environment.get_node(node_id), node_id)
            observation_vectors.append((node_id,observation_vector))
        return observation_vectors

    # Determine whether a discovered node is owned
    def is_node_owned(self, node: int):
        node_id = self.__internal_node_id_from_external_node_index(node)
        node_owned = self._actuator.get_node_privilegelevel(node_id) > PrivilegeLevel.NoAccess
        return node_owned

    def sample_random_action(self):
        if self.random_mode == "probabilistic":
            return self.sample_random_probabilistic_action()
        else:
            return numpy.random.randint(0, self.action_space.n)

    # Sample action by giving same probability to each group (local, remote, connect ports)
    def sample_random_probabilistic_action(self):
        weights = [ 0.33 / self.local_vulnerabilities_count for _ in range(self.local_vulnerabilities_count) ]
        weights.extend([ 0.33 / self.remote_vulnerabilities_count for _ in range(self.remote_vulnerabilities_count) ] )
        weights.extend([0.33 / self.port_count for _ in
                                  range(self.port_count)])
        chosen_index = random.choices(
            population=range(self.action_space.n),
            weights=weights,
            k=1
        )[0]
        return chosen_index

    # Check whether the attacker goal is reached
    def __attacker_goal_reached(self) -> bool:
        goal = self.__attacker_goal

        if not goal:
            return False

        if numpy.sum(self.__episode_rewards) < goal.reward:
            return False

        nodes_owned = self.__get__owned_nodes_indices()

        owned_count = len(nodes_owned)

        if owned_count < goal.own_atleast:
            return False

        if owned_count / self.__node_count < goal.own_atleast_percent:
            return False

        if self.__defender_agent is not None and \
                self._defender_actuator.network_availability >= goal.low_availability:
            return False

        return True

    # Check if any of the defender's constraint is not met
    def __defender_constraints_broken(self) -> bool:
        constraint = self.__defender_constraint

        if self.__defender_agent is not None and \
                self._defender_actuator.network_availability < constraint.maintain_sla:
            return True

        return False

    # Get list of indices of all owned nodes
    def __get__owned_nodes_indices(self) -> List[int]:
        if self.__owned_nodes_indices_cache is None:
            owned_nodeids = self._actuator.get_nodes_with_atleast_privilegelevel(PrivilegeLevel.LocalUser)
            self.__owned_nodes_indices_cache = [self.__find_external_index(n) for n in owned_nodeids]

        return self.__owned_nodes_indices_cache

    # Check if defender's goal is reached (e.g. full eviction of attacker)
    def __defender_goal_reached(self) -> bool:
        goal = self.__defender_goal
        return goal.eviction and not (self.__get__owned_nodes_indices())

    # Convert action index to action kind and sub-action index
    def calculate_action(self, action_index):
        if action_index < self.local_vulnerabilities_count:
            return 0, action_index
        elif action_index < self.remote_vulnerabilities_count + self.local_vulnerabilities_count:
            action_index -= self.local_vulnerabilities_count
            return 1, action_index
        else:
            action_index -= (self.local_vulnerabilities_count + self.remote_vulnerabilities_count)
            return 2, action_index

    def step(self, action_index: int) -> Tuple[Observation, float, bool, StepInfo]:
        if self.__done:
            raise RuntimeError("new episode must be started with env.reset()")

        self.__stepcount += 1
        duration = time.time() - self.__start_time
        kind, action_index = self.calculate_action(action_index)
        try:
            result = self.__execute_action(kind, action_index)
            observation, outcome, reward = self.__observation_reward_from_action_result(result)

            # Execute the defender step if provided
            if self.__defender_agent:
                self._defender_actuator.on_attacker_step_taken()
                self.__defender_agent.step(self.__environment, self._defender_actuator, self.__stepcount)

            self.__owned_nodes_indices_cache = None

            if self.__attacker_goal_reached() or self.__defender_constraints_broken():
                if self.env_type != "random_env" or self.stop_at_goal_reached:
                    self.__done = True
                reward = self.__WINNING_REWARD
            elif self.__defender_goal_reached():
                self.__done = True
                reward = self.__LOSING_REWARD
            else:
                if self.absolute_reward:
                    reward = max(0, reward)

        except OutOfBoundIndexError as error:
            logging.warning('Invalid entity index: ' + error.__str__())
            observation = self.__get_blank_observation()
            reward = 0.

        info = StepInfo(
            description='CyberBattle simulation',
            duration_in_ms=duration,
            step_count=self.__stepcount,
            network_availability=self._defender_actuator.network_availability,
            outcome=outcome)
        self.__episode_rewards.append(reward)
        self.num_iterations += 1
        self.truncated = (self.num_iterations >= self.max_num_iterations)
        self.done = (self.__done == True or self.truncated == True)
        flattened_observation = self.flatten_dict_with_arrays(self.current_observation)
        visible_features = self.concatenate_or_none(self.visible_node_features, self.visible_global_features)
        visible_observation = self.hide_features(flattened_observation, visible_features)
        return visible_observation, reward, self.__done or self.truncated, info

    def reset(self) -> Observation:
        LOGGER.info("Resetting the CyberBattle environment")
        if self.done:
            self.__update_statistics()
        self.__reset_environment()
        self.current_observation = self.__get_blank_observation()
        self.current_observation['source_node'] = self.__convert_node_info_to_observation(self.__environment.get_node(self.source_node_index), self.source_node_index)
        self.current_observation['target_node'] = self.__convert_node_info_to_observation(
            self.__environment.get_node(self.target_node_index), self.target_node_index)
        self.__owned_nodes_indices_cache = None
        self.done = False
        self.num_iterations = 0
        flattened_observation = self.flatten_dict_with_arrays(self.current_observation)
        visible_features = self.concatenate_or_none(self.visible_node_features, self.visible_global_features)
        visible_observation = self.hide_features(flattened_observation, visible_features)
        return visible_observation

    def hide_features(self, input_dict, visible_prefixes=None):
        if visible_prefixes == None:
            return input_dict
        else:
            return {key: value for key, value in input_dict.items() if
                    any(key.__contains__(prefix) for prefix in visible_prefixes)}

    def concatenate_or_none(self, list1, list2):
        if list1 is not None and list2 is not None:
            # Concatenate if both lists are not None
            return list1 + list2
        elif list1 is not None:
            # Use list1 if only list2 is None
            return list1
        elif list2 is not None:
            # Use list2 if only list1 is None
            return list2
        else:
            # Both lists are None, return None
            return None

    def render_as_fig(self):
        debug = commandcontrol.EnvironmentDebugging(self._actuator)
        # plot the cumulative reward and network side by side using plotly
        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(go.Scatter(y=numpy.array(self.__episode_rewards).cumsum(),
                                 name='cumulative reward'), row=1, col=1)
        traces, layout = debug.network_as_plotly_traces(xref="x2", yref="y2")
        for t in traces:
            fig.add_trace(t, row=1, col=2)
        fig.update_layout(layout)
        return fig

    def render(self, mode: str = 'human') -> None:
        fig = self.render_as_fig()
        if self.render_index % 5 == 0:
            auto_refresh_script = """
              <script>
                  function autoRefresh() {
                      location.reload();  // Reload the current page
                  }
                  setTimeout(autoRefresh, 5000); // Auto-refresh every 5 seconds
              </script>
              """
            with open("figure.html", "w") as html_file:
                # Write the updated HTML content with the auto-refresh JavaScript code
                html_file.write(fig.to_html() + auto_refresh_script)
            if self.fig == None:
                webbrowser.open("figure.html", new=0)
                self.fig = True
        self.render_index +=1

    def print_node_info(self, node_index, node_info):
        print("WHATS")
        if self.__environment.get_node(node_index).agent_installed == True:
            print("discovery status: owned")
        elif node_index in self.__discovered_nodes:
            print("discovery status: discovered")
        else:
            print("discovery status: not discovered")
        for key, value in node_info.items():
            print(f'{key}: {value}')
        print()

    def print_nodes_info(self, mode=2):

        owned_nodes = [node_id for node_id, node_data in self.__environment.nodes() if node_data.agent_installed]
        discovered_nodes = [node_id for node_id in self.__discovered_nodes if
                            not self.__environment.get_node(node_id).agent_installed]
        not_discovered_nodes = [node_id for node_id, node_data in self.__environment.nodes() if
                                not node_data.agent_installed and node_id not in self.__discovered_nodes]

        max_width = max(len(owned_nodes), len(discovered_nodes), len(not_discovered_nodes))

        owned_texts = []
        discovered_texts = []
        not_discovered_texts = []
        for i in range(max_width):
            if i < len(owned_nodes):
                owned = owned_nodes[i]
                owned_color = "\033[32m" if owned_nodes[i] == self.source_node_index else (
                    "\033[33m" if owned_nodes[i] == self.target_node_index else "\033[0m")
                owned_texts.append(f"{owned_color}{owned}\033[0m")
            else:
                owned_texts.append("")

            if i < len(discovered_nodes):
                discovered = discovered_nodes[i]
                discovered_color = "\033[33m" if discovered_nodes[i] == self.target_node_index else "\033[0m"
                discovered_texts.append(f"{discovered_color}{discovered}\033[0m")
            else:
                discovered_texts.append("")

            if i < len(not_discovered_nodes):
                not_discovered = not_discovered_nodes[i]
                not_discovered_texts.append(not_discovered)
            else:
                not_discovered_texts.append("")

        owned_text = "Owned: { " + ", ".join(owned_texts) + " }"
        discovered_text = "Discovered: { " + ", ".join(discovered_texts) + " }"
        not_discovered_text = "Not discovered: { " + ", ".join(not_discovered_texts) + " }"

        print(owned_text)
        print(discovered_text)
        print(not_discovered_text)
        if mode == 2:

            print()
            print("Source node:")
            print(self.__environment.network.nodes[self.source_node_index]['data'])
            print(self.environment.get_node(self.source_node_index))
            print(self.source_node_index)
            self.print_node_info(self.source_node_index, self.current_observation['source_node'])

            print("Target node:")
            print(self.target_node_index)
            self.print_node_info(self.target_node_index, self.current_observation['target_node'])
        os.system('clear')


    def seed(self, seed: Optional[int] = None) -> None:
            if seed is None:
                self._seed = seed
                # TODO: Do something with it
                return

            self.np_random, seed = seeding.np_random(seed)

    def close(self) -> None:
            return None
