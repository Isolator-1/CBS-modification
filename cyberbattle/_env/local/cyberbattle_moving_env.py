# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import time
import logging
from typing import NamedTuple, Optional, Tuple, List, Dict, TypedDict
import numpy
import random
import gym as gym
from gym import spaces
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import sys
import os
import copy
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)
from cyberbattle._env.defender import DefenderAgent
from cyberbattle.simulation.model import PortName, PrivilegeLevel, VulnerabilityType
from cyberbattle.simulation import commandcontrol, model, actions
from cyberbattle._env.local.cyberbattle_env_utils import flatten_space_dict_with_arrays, flatten_dict_with_arrays, concatenate_or_none, hide_features

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
        # Pick a random node as the agent entry node
        if self.random_starter_node:
            # While loop to be sure to select a starter node respecting the necessary condition (not isolated, if required)
            while True:
                self.__environment: model.Environment = copy.deepcopy(self.__initial_environment)
                entry_node_index = random.randrange(len(self.__environment.network.nodes))
                for node in self.__environment.network.nodes:
                    self.get_node(node).agent_installed = False
                entry_node_id, _ = list(self.__environment.network.nodes(data=True))[entry_node_index]
                self.starter_node = entry_node_id
                if self.env_type == "random_env":
                    # reset logic for the entry node random environment
                    self.__environment.network.nodes[entry_node_id]['data'].services = []
                    self.__environment.network.nodes[entry_node_id]['data'].value = 0
                    self.__environment.network.nodes[entry_node_id]['data'].properties = ["breach_node"]

                    # calculate the shortest path starting from the selected node

                    self.shortest_paths_starter = copy.deepcopy(model.calculate_average_shortest_path_length(self.__initial_environment.access_graph)[2][self.starter_node])

                    self.shortest_paths_starter.pop(int(self.starter_node))
                    # check if the number of nodes reachable is < isolation_filter_threshold, then isolated node
                    number_nodes = len(self.shortest_paths_starter)
                    self.reachable_count = sum(1 for value in self.shortest_paths_starter.values() if value is not None)
                    threshold_count = self.isolation_filter_threshold * number_nodes
                    if self.reachable_count < threshold_count:
                        # pick new starter node
                        continue
                    else:
                        # no isolation check for other environments games
                        break
                else:
                    break
            # property common to all games
            self.__environment.network.nodes[entry_node_id]['data'].agent_installed = True
        else:
            # not random starter node
            self.__environment: model.Environment = copy.deepcopy(self.__initial_environment)
            entry_node_id = None
            # use the starter node already set during random generation
            for node in self.__environment.network.nodes:
                if self.get_node(node).agent_installed == True:
                    entry_node_id = node
                    break

            # set properties according to the logic of the game (special requirements for random)
            if self.env_type == "random_env":
                if not entry_node_id:
                    entry_node_id = '0'
                self.__environment.network.nodes[entry_node_id]['data'].services = []
                self.__environment.network.nodes[entry_node_id]['data'].value = 0
                self.__environment.network.nodes[entry_node_id]['data'].properties = ["breach_node"]
        self.__environment.network.nodes[entry_node_id]['data'].agent_installed = True


        # Set it such that it is set in all cases
        self.starter_node = entry_node_id
        # Reset lists required to track episodes
        self.__discovered_nodes: List[model.NodeID] = []
        self.__owned_nodes: List[model.NodeID] = []
        self.__credential_cache: List[model.CachedCredential] = []
        self.__episode_rewards: List[float] = [] # used to track whether attacker goal is reached (if set based on episode rewards)

        # Set the actuator used to execute actions in the simulation environment
        self._actuator = actions.AgentActions(self.__environment, throws_on_invalid_actions=self.__throws_on_invalid_actions,
                                              value_coefficient=self.reward_coefficients["value_coefficient"], cost_coefficient=self.reward_coefficients["cost_coefficient"],
                                              property_discovered_coefficient=self.reward_coefficients["property_discovered_coefficient"],
                                              credential_discovered_coefficient=self.reward_coefficients["credential_discovered_coefficient"],
                                              node_discovered_coefficient=self.reward_coefficients["node_discovered_coefficient"],
                                              first_success_attack_coefficient=self.reward_coefficients["first_success_attack_coefficient"],
                                              penalty_dict=self.penalties,
                                              verbose = self.verbose
                                              )
        self._defender_actuator = actions.DefenderAgentActions(self.__environment)

        self.__stepcount = 0
        self.__start_time = time.time()
        self.__done = False

        # start with only the starting node as discovered/owned node, and set it as both source node and target node (equivalent to say no target node)
        self.__discovered_nodes.append(entry_node_id)
        self.__owned_nodes.append(entry_node_id)
        self.source_node_index = entry_node_id
        self.target_node_index = entry_node_id

    # Validation of the environment identifiers
    def validate_environment(self, environment: model.Environment):

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
                 episode_iterations = 100,
                 random_mode = "normal",
                 absolute_reward = False,
                 visible_node_features = None,
                 visible_global_features = None,
                 partial_observability=True,
                 stop_at_goal_reached=False,
                 reward_coefficients = None,
                 penalties = None,
                 verbose = False,
                 smart_winning_reward=False,
                 isolation_filter_threshold=0.1,
                 move_target_through_owned=True,
                 random_agent=False,
                 **kwargs
                 ):


        # Visualization variables
        self.fig = None
        self.render_index = 0
        self.viewer = None
        self.verbose = verbose
        self.__renderer = renderer

        self.validate_environment(initial_environment)

        self.__attacker_goal: Optional[AttackerGoal] = attacker_goal # Target goal for the attacker to win and stop the simulation.
        self.__defender_goal: DefenderGoal = defender_goal # Target goal for the defender to win and stop the simulation.
        self.__defender_constraint: DefenderConstraint = defender_constraint # Constraint to be mantain by the defender to keep the simulation running.
        self.__WINNING_REWARD = winning_reward # Reward granted to the attacker if the simulation ends because the attacker's goal is reached.
        self.__LOSING_REWARD = losing_reward # Reward granted to the attacker if the simulation ends because the Defender's goal is reached.
        self.__throws_on_invalid_actions = throws_on_invalid_actions
        self.__initial_environment: model.Environment = initial_environment
        self.__defender_agent = defender_agent

        self.env_type = env_type # identifier distinguishing the logic
        self.done = False

        # flag determining whether the simulation should stop when hacker owns all nodes
        self.stop_at_goal_reached = stop_at_goal_reached
        # cut off to truncate an episode
        self.episode_iterations = episode_iterations
        # whether at each episode there should be a new random starter node
        self.random_starter_node = random_starter_node
        # whether the reward should be absolute or not
        self.absolute_reward = absolute_reward
        # uniform random selection of actions (normal) or based on the category (probabilistic)
        self.random_mode = random_mode
        # flag indicating whether the target node should be partially observable (evolving view) or fully observable since the beginning
        self.partial_observability = partial_observability
        # whether to give winning reward when all possibel reachable nodes are reached (CyberBattleRandom)
        self.smart_winning_reward = smart_winning_reward
        # whether the target node can only be in the discovered_not_owned_list or also in the owned_list
        self.move_target_through_owned = move_target_through_owned
        # minimum percentage of nodes that should be accessible to candidate as starter node (CyberBattleRandom)
        self.isolation_filter_threshold = isolation_filter_threshold
        # whether the actions should be taken by a random agent only (debugging or benchmark generation)
        self.random_agent = random_agent

        if not reward_coefficients:
            # default reward coefficients
            reward_coefficients = {
                'value_coefficient': 1.0,
                'cost_coefficient': 1.0,
                'property_discovered_coefficient': 2.0,
                'credential_discovered_coefficient': 3.0,
                'node_discovered_coefficient': 5.0,
                'first_success_attack_coefficient': 7.0,
                'moved_source_node_unlock': 5.0
            }

        if not penalties:
            # default penalties
            penalties = {
                'suspiciousness': -5.0, # penalty for generic suspiciousness
                'scanning_unopen_port': -10.0, # penalty for attempting a connection to a port that was not open
                'repeat': -1, # penalty for repeating the same exploit attempt
                'local_exploit_failed': -20,
                'failed_remote_exploit': -50,
                'machine_not_running': 0, # penalty for attempting to connect or execute an action on a node that's not in running state
                'wrong_password': -10, # penalty for attempting a connection with an invalid password
                'blocked_by_local_firewall': -10, # traffic blocked by outgoing rule in a local firewall
                'blocked_by_remote_firewall': -10, # traffic blocked by incoming rule in a local firewall
                'invalid_action': -10,  # invalid action
                'invalid_movement': -50, # movement not possible (e.g. target movement when only one node is present)
                'movement': -10, # general movement penalty, reflecting the time spent that is lost
                'connection_to_same_node': -50 # trying to connect same node (when source == target, equivalent of no target node)
            }

        self.reward_coefficients = reward_coefficients
        self.penalties = penalties

        # lists to decide via YAML files what to change in the state space
        self.visible_node_features = visible_node_features # source and target node variables
        self.visible_global_features = visible_global_features # global variables

        # setting identifiers and counts
        self.local_vulnerabilities = self.identifiers.local_vulnerabilities
        self.remote_vulnerabilities = self.identifiers.remote_vulnerabilities
        self.ports = self.identifiers.ports
        self.properties = self.identifiers.properties

        self.local_vulnerabilities_count = len(self.local_vulnerabilities)
        self.remote_vulnerabilities_count = len(self.remote_vulnerabilities)
        self.property_count = len(self.properties)
        self.port_count = len(self.ports)

        self.__reset_environment()

        self.num_nodes = 0
        for _ in self.__environment.nodes():
            self.num_nodes += 1
        self.num_credentials = self.__initial_environment.num_credentials

        # ------- ACTION SPACE -------
        num_actions = self.local_vulnerabilities_count + self.remote_vulnerabilities_count + self.port_count
        num_actions += 2 # move forward or backward for the source node
        num_actions += 2 # move forward or backward for the target node
        self.action_space = spaces.Discrete(num_actions)

        # ------- OBSERVATION SPACE -------
        # --- NODE VARS (BOTH SOURCE AND TARGET) ---
        # We assume the agent can derive the value and sla weight of the node based on prior knowledge
        value_space = spaces.Box(low=0, high=100, shape=(1,), dtype=numpy.float32)
        privilege_level_space = spaces.Discrete(self.privilege_levels)
        status_space = spaces.Discrete(len(model.MachineStatus))
        reimageable_space = spaces.Discrete(2) # whether it can be reimaged from the defender
        sla_weight_space = spaces.Box(low=0, high=1, shape=(1,), dtype=numpy.float32)
        listening_service_space = spaces.Tuple([
            spaces.Discrete(2),  # running
            spaces.Discrete(2),  # accessible
            spaces.Discrete(2),  # previously used and found working
            spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=float), # sla weight
        ])
        listening_services_array_space = spaces.Tuple([listening_service_space] * self.port_count)
        # if partially observable, there are also the cases (0: not known, 1: known to be absent), not only local or remote
        if self.partial_observability:
            vulnerability_types = 4
            vulnerability_outcomes = 12 # outcome: 10 several possible types of outcomes may be discovered during simulation + 2 unknowns
        else:
            vulnerability_types = 2
            vulnerability_outcomes = 10

        vulnerability_space = spaces.Tuple([
            spaces.Discrete(vulnerability_types),
            spaces.Discrete(vulnerability_outcomes),
            spaces.Box(low=0.0, high=100, shape=(1,), dtype=float),  # cost
            spaces.Discrete(2) # already exploited or not
        ])
        vulnerabilities_array_space = spaces.Tuple([vulnerability_space] * (self.local_vulnerabilities_count + self.remote_vulnerabilities_count))
        property_array_space = spaces.Tuple([spaces.Discrete(2) for _ in range(self.property_count)])
        firewall_config_space = spaces.Discrete(2) # allow/block
        firewall_config_array_space = spaces.Tuple([firewall_config_space] * 2*self.port_count)

        # --- GLOBAL FEATURES ---
        number_discovered_nodes_space = spaces.Box(low=0, high=1000, shape=(1,), dtype=numpy.int32) # overall
        lateral_move_space = spaces.Discrete(2) # whether last lateral move was successful
        discovered_data_space = spaces.Discrete(2) # whether customer data were just discovered
        probe_result_space = spaces.Discrete(3) # whether last probe was successful and how
        escalation_result_space = spaces.Discrete(2) # whether last escalation was successful
        number_discovered_credentials_space = spaces.Box(low=0, high=1000, shape=(1,), dtype=numpy.int32) # overall
        owned_nodes_length_space = spaces.Discrete(1000) # how many nodes in the owned list
        discovered_nodes_not_owned_length_space = spaces.Discrete(1000) # how many nodes discovered but not owned yet
        credential_cache_empty_space = spaces.Discrete(2)
        owned_local_vulnerabilities_not_exploited_space = spaces.Discrete(1000 * self.local_vulnerabilities_count) # local vulns among the nodes in owned not yet exploited
        discovered_accessible_ports_value = spaces.Discrete(1000 * self.port_count) # ports accessible among the nodes in the discovered not owned list

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
            'global_features_number_discovered_credentials': number_discovered_credentials_space,
            'global_features_owned_nodes_length':owned_nodes_length_space,
            'global_features_discovered_not_owned_nodes_length': discovered_nodes_not_owned_length_space,
            'global_features_credential_cache_empty': credential_cache_empty_space,
            'global_features_average_discovered_value': value_space,
            'global_features_owned_local_vulnerabilities_not_exploited': owned_local_vulnerabilities_not_exploited_space,
            'global_features_discovered_accessible_ports': discovered_accessible_ports_value
        }

        # take the subset defined
        visible_features = concatenate_or_none(self.visible_node_features, self.visible_global_features)
        visible_dict = hide_features(dict_observation_space, visible_features)

        self.observation_space = spaces.Dict(visible_dict)
        self.observation_space = flatten_space_dict_with_arrays(self.observation_space)

        # ----- REWARD -----
        # reward_range: A tuple corresponding to the min and max possible rewards
        self.reward_range = (-float('inf'), float('inf'))

    # Use the actuator to execute action and gather the result
    def __execute_action(self, kind: int, action_index: int) -> actions.ActionResult:
        if kind == 0: # local_vulnerability
            if self.verbose:
                print("local vulnerability")
            return self._actuator.exploit_local_vulnerability(
                self.source_node_index,
                self.__index_to_local_vulnerabilityid(action_index))
        elif kind == 1: # remote_vulnerability
            if self.verbose:
                print("remote vulnerability")
            # no target node case
            if self.source_node_index == self.target_node_index:
                if self.verbose:
                    print("Penalty (remote vulnerability not possible on the same node):", self.penalties['invalid_action'])
                return actions.ActionResult(reward=self.penalties['invalid_action'], outcome=None)
            return self._actuator.exploit_remote_vulnerability(
                self.source_node_index,
                self.target_node_index,
                self.__index_to_remote_vulnerabilityid(action_index))
        elif kind == 2: # connect
            if self.verbose:
                print("Connection", self.source_node_index, self.target_node_index)
            # connection with empty cache
            if len(self.__credential_cache) == 0:
                if self.verbose:
                    print("Penalty (connection with empty cache):", -1) #TODO
                return actions.ActionResult(reward=-1, outcome=None)
            # connection with no target node
            if self.source_node_index == self.target_node_index:
                if self.verbose:
                    print("Penalty (connection not possible on the same node):", self.penalties['connection_to_same_node'])
                return actions.ActionResult(reward=self.penalties['connection_to_same_node'], outcome=None)
            # simplification: try all credentials
            for credential in self.__credential_cache:
                result = self._actuator.connect_to_remote_machine(
                    self.source_node_index,
                    self.target_node_index,
                    self.__index_to_port_name(action_index),
                    credential.credential)
                if result.outcome:
                    if self.target_node_index not in self.__owned_nodes:
                        self.__owned_nodes.append(self.target_node_index)
                    self.source_node_index = self.target_node_index
                    return result
        elif kind == 3: #movement
            if action_index == 0:
                if self.verbose:
                    print("movement -> source node forward")
                result = self.move_source_node(+1)
            elif action_index == 1:  # move source node
                if self.verbose:
                    print("movement -> source node backward")
                result = self.move_source_node(-1)
            elif action_index == 2:
                if self.verbose:
                    print("movement -> target node forward")
                result = self.move_target_node(+1)
            elif action_index == 3:
                if self.verbose:
                    print("movement -> target node backward")
                result = self.move_target_node(-1)
        return result

    def move_source_node(self, movement):
        if self.verbose:
            print("Owned nodes", self.__owned_nodes)
        if len(self.__owned_nodes) == 1:
            if self.verbose:
                print("Penalty (movement not possible - no other owned node):", self.penalties['invalid_movement'])
            return actions.ActionResult(reward=self.penalties['invalid_movement'], outcome=None)
        position = self.__owned_nodes.index(self.source_node_index)
        if position == len(self.__owned_nodes)-1 and movement == 1:
            new_position = 0
        elif position == 0 and movement == -1:
            new_position = len(self.__owned_nodes)-1
        else:
            new_position = position + movement
        # in case an owned node cannot be a target node, move also the target node if source node == target node, which means still no target node
        if not self.move_target_through_owned and self.target_node_index == self.source_node_index:
            self.target_node_index = self.__owned_nodes[new_position]
        self.source_node_index = self.__owned_nodes[new_position]
        if self.verbose:
            print("Reward (movement possible):", self.penalties['movement']) #TODO
        return actions.ActionResult(reward=self.penalties['movement'], outcome=model.Movement(source=1, forward=movement))

    def move_target_node(self, movement):
        # list to explore depends on whether also owned nodes can be target nodes
        if self.move_target_through_owned:
            # both owned and discovered not owned
            target_list = [node for node in self.get_owned_nodes()]
            target_list.extend([node for node in self.get_discovered_not_owned_nodes()])
        else:
            # discovered not owned only
            target_list = [node for node in self.get_discovered_not_owned_nodes()]
        if self.verbose:
            print("Discovered not owned nodes", target_list)
        if len(target_list) <= 1:
            if self.verbose:
                print("Penalty (movement not possible - no other possible movement):", self.penalties['invalid_movement'])
            return actions.ActionResult(reward=self.penalties['invalid_movement'], outcome=None)
        if self.target_node_index not in target_list: # possible only in the case discovered not owned only
            # if source node index == target node index go to the discovered nodes (first or last of the list)
            if movement == 1:
                self.target_node_index = target_list[0]
            elif movement == -1:
                self.target_node_index = target_list[-1]
            if self.verbose:
                print("Reward (movement possible - unlocked view):", self.penalties['movement'])
            return actions.ActionResult(reward=self.penalties['movement'], outcome=model.Movement(source=0, forward=movement)) # for now give small good reward because it has unlocked itself from the situation
        # node is in the target list
        position = target_list.index(self.target_node_index)
        if position == len(target_list)-1 and movement == 1:
            new_position = 0
        elif position == 0 and movement == -1:
            new_position = len(target_list)-1
        else:
            new_position = position + movement
        self.target_node_index = target_list[new_position]
        if self.verbose:
            print("Reward (movement possible):", self.penalties['movement'])
        return actions.ActionResult(reward=self.penalties['movement'], outcome=model.Movement(source=0, forward=movement))

    # Create a blank observation conforming to the observation space
    def __get_blank_observation(self):
        blank_observation = {
            'source_node': {
                'firewall_config_array': [0 for _ in range(2*self.port_count)],
                'listening_services_array': [(0, 0, 0, numpy.array([0.0], dtype=numpy.float64)) for _ in
                                             range(self.port_count)],
                'privilege_level': 0,
                'property_array': [0 for _ in range(self.property_count)],
                'reimageable': 0,
                'sla_weight': numpy.array([0.0], dtype=numpy.float32),
                'status': 0,
                'value': numpy.array([0.0], dtype=numpy.float32),
                'vulnerabilities_array': [(0, 0, 0, numpy.array([0.0], dtype=numpy.float64), 0) for _ in
                                                range(self.local_vulnerabilities_count + self.remote_vulnerabilities_count)],
            },
            'target_node': {
                'firewall_config_array': [0 for _ in range(2*self.port_count)],
                'listening_services_array': [(0, 0 ,0, numpy.array([0.0], dtype=numpy.float64)) for _ in
                                             range(self.port_count)],
                'privilege_level': 0,
                'property_array': [0 for _ in range(self.property_count)],
                'reimageable': 0,
                'sla_weight': numpy.array([0.0], dtype=numpy.float32),
                'status': 0,
                'value': numpy.array([0.0], dtype=numpy.float32),
                'vulnerabilities_array': [(0, 0, numpy.array([0.0], dtype=numpy.float64), 0) for _ in
                                                range(self.local_vulnerabilities_count + self.remote_vulnerabilities_count)],
            },
            'global_features': {
                'customer_data_found': 0,
                'escalation': 0,
                'lateral_move': 0,
                'number_discovered_credentials': numpy.array([0], dtype=numpy.int32),
                'number_discovered_nodes': numpy.array([0], dtype=numpy.int32),
                'probe_result': 0,
                'discovered_not_owned_nodes_length': 0,
                'owned_nodes_length': 1,
                'credential_cache_empty': 1,
                'average_discovered_value': numpy.array([0], dtype=numpy.float32),
                'owned_local_vulnerabilities_not_exploited': 0,
                'discovered_accessible_ports': 0
            }
        }

        return blank_observation


    # Convert node information to the proper feature vector used in the observation space
    def __convert_node_info_to_observation(self, node_info, node_id) -> Dict:
        # listening service always visible regarless if the node is owned or not
        listening_services_array = [
            (0, 0, 0, numpy.array([0.0], dtype=numpy.float32)) for _ in range(self.port_count)
        ]

        for service in node_info.services:
            if self.__portname_to_index(service.name) != -1:
                listening_services_array[self.__portname_to_index(service.name)] = (
                    int(service.running),
                    self.is_service_accessible(service, node_id),
                    int(service.name in self._actuator.get_ports_surely_working(node_id)),
                    numpy.array([service.sla_weight], dtype=numpy.float32)
                )

        # vulnerabilities info fully visible to the agent if node is owned, else have to be discovered
        if self.partial_observability:
            if self.__environment.get_node(node_id).agent_installed == True:
                # OWNED: Vulnerabilities not present by default
                vulnerabilities_array = [
                    (1, 1, numpy.array([0.0], dtype=numpy.float64),0) for _ in range(self.local_vulnerabilities_count + self.remote_vulnerabilities_count)
                ]
            else:
                # DISCOVERED: Vulnerabilities not known by default
                vulnerabilities_array = [
                    (0, 0, numpy.array([0.0], dtype=numpy.float64),0) for _ in range(self.local_vulnerabilities_count + self.remote_vulnerabilities_count)
                ]

            # write real values if discovered vulnerabilities or if the node is owned
            for vulnerability_id, vulnerability_info in node_info.vulnerabilities.items():
                if self.__vulnerabilityid_to_index(vulnerability_id) in self._actuator.get_discovered_vulnerabilities(node_id):
                    if self.__vulnerabilityid_to_index(vulnerability_id) != -1:
                        vulnerabilities_array[self.__vulnerabilityid_to_index(vulnerability_id)] = (
                            vulnerability_info.type.value + 2,
                            model.map_outcome_to_index(vulnerability_info.outcome) + 2,
                            numpy.array([vulnerability_info.cost], dtype=numpy.float64),
                            int(self.__vulnerabilityid_to_index(vulnerability_id) in self._actuator.get_vulnerabilities_used(node_id))
                        )

            # if the node is not owned, add the discovered absent vulnerabilities
            if self.__environment.get_node(node_id).agent_installed == False:
                for vulnerability_id in self._actuator.get_absence_discovered_vulnerabilities(node_id):
                    vulnerabilities_array[vulnerability_id] = (
                        1,
                        1,
                        numpy.array([0.0], dtype=numpy.float64),
                        0
                    )

        else:
            # fully observable, know the truth
            vulnerabilities_array = [
                (0, 0, numpy.array([0.0], dtype=numpy.float64),0) for _ in
                range(self.local_vulnerabilities_count + self.remote_vulnerabilities_count)
            ]
            for vulnerability_id, vulnerability_info in node_info.vulnerabilities.items():
                if self.__vulnerabilityid_to_index(vulnerability_id) != -1:
                    vulnerabilities_array[self.__vulnerabilityid_to_index(vulnerability_id)] = (
                        vulnerability_info.type.value,
                        model.map_outcome_to_index(vulnerability_info.outcome),
                        numpy.array([vulnerability_info.cost], dtype=numpy.float64),
                        int(self.__vulnerabilityid_to_index(vulnerability_id) in self._actuator.get_vulnerabilities_used(node_id))
                    )

        property_array = [
            0 for _ in range(self.property_count)
        ]

        # in case of partial observability only discovered properties are known or all if node is owned
        if self.partial_observability:
            for property in self._actuator.get_discovered_properties(node_id):
                property_array[property] = 1
        else:
            # fully visibility
            for property in node_info.properties:
                if self.__property_to_index(property) != -1:
                    property_array[self.__property_to_index(property)] = 1


        # Firewall information encoded always visible
        firewall_config_array = [
            0 for _ in range(2*self.port_count)
        ]
        for config in node_info.firewall.incoming:
            permission = config.permission.value
            if self.__portname_to_index(config.port) != -1:
                firewall_config_array[self.__portname_to_index(config.port)] = permission
        for config in node_info.firewall.outgoing:
            permission = config.permission.value
            if self.__portname_to_index(config.port) != -1:
                firewall_config_array[self.port_count + self.__portname_to_index(config.port)] = permission

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


    # Calculate reward based on outcome
    def __observation_reward_from_action_result(self, result: actions.ActionResult) -> Tuple[Observation, float]:
        self.current_observation = self.__get_blank_observation()
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


        self.current_observation['global_features']['owned_nodes_length'] = int(len(self.__owned_nodes))
        self.current_observation['global_features']['discovered_not_owned_nodes_length'] = int(len([node for node in self.get_discovered_not_owned_nodes()]))
        self.current_observation['global_features']['credential_cache_empty'] = numpy.int32(len(self.__credential_cache) == 0)
        discovered_not_owned = self.get_discovered_not_owned_nodes()
        other_discovered_not_owned = [node for node in discovered_not_owned if node != self.target_node_index]
        if len(other_discovered_not_owned) > 0:
            average_discovered_value = numpy.float32(numpy.mean([node_info.value for node_id, node_info in self.__environment.nodes() if node_id in discovered_not_owned and node_id != self.target_node_index]))
        else:
            average_discovered_value = numpy.float32(0)
        self.current_observation['global_features']['average_discovered_value'] = numpy.array([average_discovered_value], dtype=numpy.float32)

        self.current_observation['global_features']['owned_local_vulnerabilities_not_exploited'] = 0
        for node in self.__owned_nodes:
            # count number of local vulnerabilities not used yet for all owned nodes
            node_data = self.get_node(node)
            for vulnerability in node_data.vulnerabilities:
                vulnerability_data = node_data.vulnerabilities[vulnerability]
                if vulnerability_data.type == VulnerabilityType.LOCAL and self.__vulnerabilityid_to_index(vulnerability) not in self._actuator.get_vulnerabilities_used(node):
                    self.current_observation['global_features']['owned_local_vulnerabilities_not_exploited'] += 1

        self.current_observation['global_features']['discovered_accessible_ports'] = 0
        discovered_not_owned_nodes = self.get_discovered_not_owned_nodes()
        for node in discovered_not_owned_nodes:
            node_data = self.get_node(node)
            for service in node_data.services:
                 if self.__portname_to_index(service.name) != -1:
                    self.current_observation['global_features']['discovered_accessible_ports'] += self.is_service_accessible(service, node)

        self.current_observation['source_node'] = self.__convert_node_info_to_observation(
            self.__environment.get_node(self.source_node_index), self.source_node_index)
        if self.target_node_index == self.source_node_index:
            pass
            # keep it with 0s
        else:
            self.current_observation['target_node'] = self.__convert_node_info_to_observation(
                self.__environment.get_node(self.target_node_index), self.target_node_index)
        return self.current_observation, outcome, result.reward

    def step(self, action_index: int) -> Tuple[Observation, float, bool, StepInfo]:

        if self.__done:
            raise RuntimeError("new episode must be started with env.reset()")

        self.__stepcount += 1
        duration = time.time() - self.__start_time
        if self.random_agent:
            # replace the action_index with a random one
            action_index = random.randint(0, self.action_space.n - 1)
        kind, action_index = self.calculate_action(copy.deepcopy(action_index))
        try:
            result = self.__execute_action(kind, action_index)

            observation, outcome, reward = self.__observation_reward_from_action_result(result)

            # Execute the defender step if provided
            if self.__defender_agent:
                self._defender_actuator.on_attacker_step_taken()
                reimaged_nodes = self.__defender_agent.step(self.__environment, self._defender_actuator, self.__stepcount)
                self.last_reimaged = reimaged_nodes
                self.overall_reimaged.extend(reimaged_nodes)
                if reimaged_nodes != []:
                    for node in reimaged_nodes:
                        self.__owned_nodes.remove(node)
                if self.source_node_index in reimaged_nodes and len(self.__owned_nodes) > 0:
                    self.source_node_index = random.choice(self.__owned_nodes)
                    self.target_node_index = self.source_node_index
                if self.target_node_index in reimaged_nodes:
                    self.target_node_index = self.source_node_index

            if self.env_type == "random_env" and self.smart_winning_reward:
                # Check if all remaining paths are None ( == not reachable ) -> then finish episode
                for node_id in self.__owned_nodes:
                    self.shortest_paths_starter.pop(int(node_id), None)
                if all(value is None for value in self.shortest_paths_starter.values()):
                    print("Reached all the nodes accessible...")
                    reward = self.__WINNING_REWARD
                    self.__done = True

            # Check whether there has bee some ending conditions
            if not self.__done:
                if self.__attacker_goal_reached() or self.__defender_constraints_broken():
                    if self.env_type != "random_env" or self.stop_at_goal_reached:
                        self.__done = True
                    if self.__attacker_goal_reached():
                        reward = self.__WINNING_REWARD
                    if self.__defender_constraints_broken():
                        self.defender_constraints_broken = True
                elif self.__defender_goal_reached():
                    self.__done = True
                    self.evicted = True
                    reward = self.__LOSING_REWARD
                else:
                    if self.absolute_reward:
                        reward = max(0, reward)

        except OutOfBoundIndexError as error:
            logging.warning('Invalid entity index: ' + error.__str__())
            reward = 0.

        info = StepInfo(
            description='CyberBattle simulation',
            duration_in_ms=duration,
            step_count=self.__stepcount,
            network_availability=self._defender_actuator.network_availability,
            outcome=outcome)
        self.__episode_rewards.append(reward)
        self.num_iterations += 1
        self.truncated = (self.num_iterations >= self.episode_iterations)
        self.done = (self.__done == True or self.truncated == True)
        flattened_observation = flatten_dict_with_arrays(self.current_observation)
        visible_features = concatenate_or_none(self.visible_node_features, self.visible_global_features)
        visible_observation = hide_features(flattened_observation, visible_features)

        return visible_observation, reward, self.__done or self.truncated, info

    def reset(self) -> Observation:
        LOGGER.info("Resetting the CyberBattle environment")

        # Reset all elements needed
        self.__reset_environment()

        self.overall_reimaged = []
        self.defender_constraints_broken = False
        self.evicted = False

        self.current_observation = self.__get_blank_observation()
        self.current_observation['source_node'] = self.__convert_node_info_to_observation(
            self.__environment.get_node(self.source_node_index), self.source_node_index)
        # if target not set, leave the default values
        if self.target_node_index == self.source_node_index:
            pass
        else:
            self.current_observation['target_node'] = self.__convert_node_info_to_observation(
                self.__environment.get_node(self.target_node_index), self.target_node_index)

        # global default values
        self.current_observation['global_features']['owned_local_vulnerabilities_not_exploited'] = 0
        for node in self.__owned_nodes:
            # count number of local vulnerabilities not used yet for all owned nodes
            node_data = self.get_node(node)
            for vulnerability in node_data.vulnerabilities:
                vulnerability_data = node_data.vulnerabilities[vulnerability]
                if vulnerability_data.type == VulnerabilityType.LOCAL and self.__vulnerabilityid_to_index(
                    vulnerability) not in self._actuator.get_vulnerabilities_used(node):
                    self.current_observation['global_features']['owned_local_vulnerabilities_not_exploited'] += 1

        self.done = False
        self.num_iterations = 0
        flattened_observation = flatten_dict_with_arrays(self.current_observation)
        visible_features = concatenate_or_none(self.visible_node_features, self.visible_global_features)
        visible_observation = hide_features(flattened_observation, visible_features)
        return visible_observation

    # Check whether the attacker goal is reached
    def __attacker_goal_reached(self) -> bool:
        goal = self.__attacker_goal

        if not goal:
            return False

        if numpy.sum(self.__episode_rewards) < goal.reward:
            return False

        nodes_owned = self.__owned_nodes

        owned_count = len(nodes_owned)

        if owned_count < goal.own_atleast:
            return False

        if owned_count / self.num_nodes < goal.own_atleast_percent:
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

    # Check if defender's goal is reached (e.g. full eviction of attacker)
    def __defender_goal_reached(self) -> bool:
        goal = self.__defender_goal
        return goal.eviction and not (self.__owned_nodes)

    # Convert action index to action kind and sub-action index
    def calculate_action(self, action_index):
        if action_index < self.local_vulnerabilities_count:
            return 0, action_index
        elif action_index < self.remote_vulnerabilities_count + self.local_vulnerabilities_count:
            action_index -= self.local_vulnerabilities_count
            return 1, action_index
        elif action_index < self.remote_vulnerabilities_count + self.local_vulnerabilities_count + self.port_count:
            action_index -= (self.local_vulnerabilities_count + self.remote_vulnerabilities_count)
            return 2, action_index
        else:
            # movement of the source or target node
            return 3, action_index - (
                    self.remote_vulnerabilities_count + self.local_vulnerabilities_count + self.port_count)

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is None:
            self._seed = seed
            random.seed(seed)
            numpy.random.seed(seed)
            return

    def close(self) -> None:
        return None

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


    # Called at the end of the episode to gather statistics
    def get_statistics(self):
        owned_nodes = [node_id for node_id, node_data in self.__environment.nodes() if node_data.agent_installed]
        discovered_nodes = self.__discovered_nodes
        not_discovered_nodes = [node_id for node_id, node_data in self.__environment.nodes() if node_id not in self.__discovered_nodes and node_id != self.source_node_index]
        num_discovered_credentials = len(self.__credential_cache)
        return len(owned_nodes), len(discovered_nodes), len(not_discovered_nodes), self.num_nodes, num_discovered_credentials / self.num_credentials


    def get_owned_nodes_feature_vectors(self):
        owned_nodes = [node_id for node_id, node_data in self.__environment.nodes() if node_data.agent_installed]
        observation_vectors = []
        for node_id in owned_nodes:
            observation_vector = self.__convert_node_info_to_observation(
                self.__environment.get_node(node_id), node_id)
            observation_vectors.append((node_id,observation_vector))
        return observation_vectors

    def get_owned_nodes(self):
        return self.__owned_nodes

    def get_discovered_nodes(self):
        return self.__discovered_nodes


    def get_action_name(self, action_index):
        if action_index < self.local_vulnerabilities_count:
            return self.__index_to_local_vulnerabilityid(action_index)
        elif action_index < self.local_vulnerabilities_count + self.remote_vulnerabilities_count:
            return self.__index_to_remote_vulnerabilityid(action_index - self.local_vulnerabilities_count)
        elif action_index < self.local_vulnerabilities_count + self.remote_vulnerabilities_count + self.port_count:
            return self.__index_to_port_name(action_index - self.local_vulnerabilities_count - self.remote_vulnerabilities_count)
        elif action_index < self.local_vulnerabilities_count + self.remote_vulnerabilities_count + self.port_count + 1:
            return "source node selection forward"
        elif action_index < self.local_vulnerabilities_count + self.remote_vulnerabilities_count + self.port_count + 2:
            return "source node selection backward"
        elif action_index < self.local_vulnerabilities_count + self.remote_vulnerabilities_count + self.port_count + 3:
            return "target node selection forward"
        elif action_index < self.local_vulnerabilities_count + self.remote_vulnerabilities_count + self.port_count + 4:
            return "target node selection backward"

    def get_nodes(self):
        return self.__environment.nodes()

    def get_graph(self):
        return self.__environment

    def get_vulnerability_index(self, vulnerability_id: model.VulnerabilityID) -> int:
        return self.__vulnerabilityid_to_index(vulnerability_id)

    def get_actuator(self):
        return self._actuator

    def get_credential_cache(self):
        return self.__credential_cache

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

    def get_property_index(self, property_name: model.PropertyName) -> int:
        """Return the integer index from a PortName"""
        return self.__property_to_index(property_name)

    def get_discovered_credentials(self):
        credentials_list = []
        for credential in self.__credential_cache:
            credentials_list.append(credential.credential)
        return credentials_list

    def get_discovered_not_owned_nodes(self):
        discovered_not_owned_nodes = [node_id for node_id in self.__discovered_nodes if not node_id in self.__owned_nodes]
        return discovered_not_owned_nodes

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

    # determine whether a service is accessible with the current credentials
    def is_service_accessible(self, service, node_id):
        for credential in self.__credential_cache:
            if credential.port == service.name and credential.node == node_id:
                return 1
            else:
                continue
        return 0

    # determine whether a service is accessible with the current credentials and if they are valid (not discoverable to the agent)
    def is_service_accessible_by_valid_credentials(self, service, node_id):
        for credential in self.__credential_cache:
            if credential.port == service.name and credential.node == node_id and credential.valid:
                return 1
            else:
                continue
        return 0

    def set_cut_off(self, cut_off):
        self.episode_iterations = cut_off

    def set_isolation_filter_threshold(self, threshold):
        self.isolation_filter_threshold = threshold

    def sample_random_action(self):
        if self.random_mode == "probabilistic":
            return self.sample_random_probabilistic_action()
        else:
            return numpy.random.randint(0, self.action_space.n)

    # Sample action by giving same probability to each group (local, remote, connect ports)
    def sample_random_probabilistic_action(self):
        weights = [ 0.25 / self.local_vulnerabilities_count for _ in range(self.local_vulnerabilities_count) ]
        weights.extend([ 0.25 / self.remote_vulnerabilities_count for _ in range(self.remote_vulnerabilities_count) ] )
        weights.extend([0.25 / self.port_count for _ in
                                  range(self.port_count)])
        weights.extend([0.25 / 4 for _ in range(4)]) # movement
        chosen_index = random.choices(
            population=range(self.action_space.n),
            weights=weights,
            k=1
        )[0]
        return chosen_index

    def set_random_starter_node(self, random_starter_node):
        self.random_starter_node = random_starter_node

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
        if self.__environment.get_node(node_index).agent_installed == True:
            print("discovery status: owned")
        elif node_index in self.__discovered_nodes:
            print("discovery status: discovered")
        else:
            print("discovery status: not discovered")
        for key, value in node_info.items():
            print(f'{key}: {value}')
        print()

    def print_nodes_info(self, mode=1):

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
            print(self.source_node_index)
            self.print_node_info(self.source_node_index, self.current_observation['source_node'])
            print("Target node:")
            print(self.target_node_index)
            self.print_node_info(self.target_node_index, self.current_observation['target_node'])
            print("Global features:")
            for key, value in self.current_observation['global_features'].items():
                print(f'{key}: {value}')

            discovered_not_owned = self.get_discovered_not_owned_nodes()

            print("----------------")
            print("Discovered values: ", [self.__environment.get_node(node_id).value for node_id in discovered_not_owned])
            vulnerabilities_used = []
            for node in self.__owned_nodes:
                node_data = self.get_node(node)
                for vulnerability in node_data.vulnerabilities:
                    if node_data.vulnerabilities[vulnerability].type == VulnerabilityType.LOCAL:
                        vulnerabilities_used.append(int(self.__vulnerabilityid_to_index(vulnerability) in self._actuator.get_vulnerabilities_used(node)))

            services_accessible = []
            for node in discovered_not_owned:
                node_data = self.get_node(node)
                for service in node_data.services:
                    if self.__portname_to_index(service.name) != -1:
                        services_accessible.append(self.is_service_accessible(service, node))
            print("Local vulnerabilities used: ", vulnerabilities_used)
            print("Credential cache length: ", len(self.__credential_cache))
            print("Discovered accessible ports: ", services_accessible)



