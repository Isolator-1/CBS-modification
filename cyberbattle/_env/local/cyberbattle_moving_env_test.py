# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test the CyberBattle Gym environment"""

import pytest
import sys
import os
import copy
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)
import time
import numpy
from cyberbattle.simulation import model
import gym
import numpy as np
import argparse
from cyberbattle.agents.train_utils import dict_to_array
from cyberbattle._env.local.cyberbattle_env_switch import RandomSwitchEnv
from stable_baselines3.common.env_checker import check_env
import random


# List of environment names
env_names = [ 'CyberBattleRandomLocal-v0', 'CyberBattleChainLocal-v0', 'ActiveDirectoryLocal-v0','CyberBattleTinyLocal-v0', 'CyberBattleToyCtfLocal-v0']
observable_env_names = [ 'CyberBattleRandomLocalObservable-v0', 'CyberBattleChainLocalObservable-v0','ActiveDirectoryLocalObservable-v0', 'CyberBattleTinyLocalObservable-v0', 'CyberBattleToyCtfLocalObservable-v0']
random_starting_node_env_names = ['CyberBattleRandomLocal-v0', 'ActiveDirectoryLocal-v0'] # random selection to be tested for these
probabilistic_action_env_names = ['CyberBattleRandomLocalProbabilistic-v0','CyberBattleToyCtfLocalProbabilistic-v0', 'CyberBattleTinyLocalProbabilistic-v0', 'ActiveDirectoryLocalProbabilistic-v0',  'CyberBattleChainLocalProbabilistic-v0']


@pytest.mark.parametrize('env_name', env_names)
# test some normal gym iterations
def test_few_gym_iterations(env_name, num_episodes: int = 100, num_iterations: int = 10000, render_flag: bool = False, print_flag: bool = False) -> None:
    print("----- Test few gym iterations...")
    env = gym.make(env_name)
    print("Environment", env_name)
    """Run a few iterations of the gym environment"""
    print("Possible actions:", env.identifiers.local_vulnerabilities, env.identifiers.remote_vulnerabilities,env.identifiers.ports)
    for i in range(num_episodes):
        if print_flag:
            print(f"----- Episode {i} -----")
        env.reset()
        for t in range(num_iterations):
            if print_flag:
                print(f"----- Iteration {t} -----")
                env.print_nodes_info(mode=2)
                print()
            # Sample a valid action
            action = env.sample_random_action()
            if print_flag:
                print("Action:", env.get_action_name(action))
            observation, reward, done, info = env.step(action)
            if render_flag:
                env.render()
            if print_flag:
                print("Reward:", reward)
            if print_flag or render_flag:
                time.sleep(0.1)
            if done:
                if print_flag:
                    print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()
    print("VVVVV Test few gym iterations...")

@pytest.mark.parametrize('env_name', env_names)
# Test that the starter node has the correct properties set according to the game, and that there is just one at each episode
def test_starter_node(env_name, num_episodes: int = 100) -> None:
    print("----- Test starter node properties and checking it is the only one...")
    env = gym.make(env_name)
    for i in range(num_episodes):
        env.reset()
        starter_node = env.get_node(env.starter_node)
        assert starter_node.agent_installed
        if env_name == "CyberBattleRandomLocal-v0":
            assert starter_node.properties == ['breach_node']
            assert starter_node.value == 0
            assert starter_node.services == []
        for node in env.get_nodes():
            if node[0] != env.starter_node:
                assert not node[1].agent_installed
                if env_name == "CyberBattleRandomLocal-v0":
                    assert not node[1].properties == ['breach_node']

    if env_name == "CyberBattleRandomLocal-v0":
        env.set_random_starter_node(False)
        starter_node = None
        for i in range(num_episodes):
            env.reset()
            if i == 0:
                starter_node = env.starter_node
            else:
                assert starter_node == env.starter_node
    env.set_random_starter_node(True)
    env.close()
    print("VVVVV Test starter node properties...")

@pytest.mark.parametrize('env_name', probabilistic_action_env_names)
# Test the starting node is selected in average randomly
def test_probabilistic_action_selection(env_name):
    print("----- Testing distribution of action choices...")
    env = gym.make(env_name)
    num_episodes = 50
    num_iterations = 100

    # Track the distribution of started node IDs across episodes
    action_ids_distribution = []

    for i in range(num_episodes):
        _ = env.reset()
        for t in range(num_iterations):

            # Sample a valid action
            action = env.sample_random_action()
            action_ids_distribution.append(action)

            observation, reward, done, info = env.step(action)

            if done:
                break

    unique_action_choises, counts = np.unique(action_ids_distribution, return_counts=True)

    local_attacks_count = 0
    for action_id in range(env.get_local_attacks_count()):
        local_attacks_count += counts[action_id]

    remote_attacks_count = 0
    for action_id in range(env.get_local_attacks_count(), env.get_local_attacks_count()+env.get_remote_attacks_count()):
        remote_attacks_count += counts[action_id]

    ports_count = 0
    for action_id in range(env.get_local_attacks_count()+env.get_remote_attacks_count(), env.get_local_attacks_count()+env.get_remote_attacks_count()+env.get_port_count()):
        ports_count += counts[action_id]

    selections_count = 0
    for action_id in range(env.get_local_attacks_count()+env.get_remote_attacks_count()+env.get_port_count(), env.get_local_attacks_count()+env.get_remote_attacks_count()+env.get_port_count()+4):
        selections_count += counts[action_id]

    tolerance = 0.2  # double at maximum, tolerance for approximate randomness, depend also on the number of episodes
    expected_count = (num_episodes*num_iterations) / 4

    assert all(abs(count - expected_count) <= tolerance * (num_episodes*num_iterations) / 4
               for count in (local_attacks_count, remote_attacks_count, ports_count, selections_count))

    print("VVVVV Test passed: action IDs chosen based on their kind!")


@pytest.mark.parametrize('env_name', env_names)
# Test that actions are selected according to a uniform distribution
def test_normal_action_selection(env_name):
    print("----- Testing distribution of action choices...")
    env = gym.make(env_name)
    num_episodes = 50
    num_iterations = 100

    # Track the distribution of started node IDs across episodes
    action_ids_distribution = []

    for i in range(num_episodes):
        env.reset()
        for t in range(num_iterations):

            # Sample a valid action
            action = env.sample_random_action()
            action_ids_distribution.append(action)

            observation, reward, done, info = env.step(action)

            if done:
                break

    unique_action_choises, counts = np.unique(action_ids_distribution, return_counts=True)

    tolerance = 0.2  # double at maximum, tolerance for approximate randomness, depend also on the number of episodes
    expected_count = (num_episodes*num_iterations) / len(unique_action_choises)
    assert all(abs(count - expected_count) <= tolerance * (num_episodes*num_iterations) / 3
               for count in counts)
    time.sleep(5)
    print("VVVVV Test passed: action IDs chosen approximately uniformly!")


# Helper function to extract scalar values into an array
def extract_scalar_values(d):
    values = []
    if isinstance(d, np.ndarray):
        for value in d:
            values.append(value)
    else:
        for value in d.values():
            if isinstance(value, dict):
                values.extend(extract_scalar_values(value))
            elif isinstance(value, list):
                for elem in value:
                    if isinstance(elem, tuple):
                        for inner_value in elem:
                            if isinstance(inner_value, np.ndarray):
                                values.append(inner_value[0])
                            else:
                                values.append(inner_value)
                    elif isinstance(elem, np.ndarray) and elem.size == 1:
                        values.append(elem.item())
                    else:
                        values.append(elem)
            elif isinstance(value, (int, float)):
                values.append(value)
            elif isinstance(value, np.ndarray) and value.size == 1:
                values.append(value.item())
            else:
                values.append(value.item())
    return values

@pytest.mark.parametrize('env_name', env_names)
# test that the observation is correctly converted to a dictionary and into a list of scalar values
def test_observation_conversion(env_name):
    print("----- Test observation conversion...")
    env = gym.make(env_name)
    num_episodes = 100
    num_iterations = 100

    for i in range(num_episodes):
        observation = env.reset()
        for t in range(num_iterations):
            # Sample a valid action
            # Extract scalar values into arrays
            original_array = extract_scalar_values(env.current_observation)
            converted_array = extract_scalar_values(observation)
            final_array = extract_scalar_values(dict_to_array(observation))

            assert original_array == converted_array
            assert original_array == final_array

            action = env.sample_random_action()

            observation, reward, done, info = env.step(action)
            if done:
                break
    env.close()
    print("VVVVV Observation conversion correct...")

@pytest.mark.parametrize('env_name', env_names)
def test_wrap_spec(env_name):
    env = gym.make(env_name)
    class DummyWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            assert hasattr(self, 'spec')
            self.spec.dummy = 7

    assert hasattr(env.spec, 'properties')
    assert hasattr(env.spec, 'ports')
    assert hasattr(env.spec, 'local_vulnerabilities')
    assert hasattr(env.spec, 'remote_vulnerabilities')

    env = DummyWrapper(env)

    assert hasattr(env.spec, 'properties')
    assert hasattr(env.spec, 'ports')
    assert hasattr(env.spec, 'local_vulnerabilities')
    assert hasattr(env.spec, 'remote_vulnerabilities')

@pytest.mark.parametrize('env_name', env_names)
# Check correct update of global features
def test_global_features(env_name, num_episodes = 10, num_iterations = 100):
    print("----- Test global features update...")
    env = gym.make(env_name)
    blank_global_features = {
        'number_discovered_nodes': numpy.array([0], dtype=numpy.int32),
        'lateral_move': 0,
        'customer_data_found': 0,
        'probe_result': 0,
        'escalation': 0,
        'number_discovered_credentials': numpy.array([0], dtype=numpy.int32),
        'owned_nodes_length': 1,
        'discovered_not_owned_nodes_length': 0,
        'credential_cache_empty': 1,
        'average_discovered_value': numpy.array([0], dtype=numpy.float32),
        'owned_local_vulnerabilities_not_exploited': 0,
        'discovered_accessible_ports': 0
    }

    for i in range(num_episodes):
        env.reset()
        observation_dict = env.current_observation
        global_features = copy.deepcopy(blank_global_features)
        global_features['owned_local_vulnerabilities_not_exploited'] = 0
        for node in env.get_owned_nodes():
            # count number of local vulnerabilities not used yet for all owned nodes
            node_data = env.get_node(node)

            for vulnerability in node_data.vulnerabilities:
                vulnerability_data = node_data.vulnerabilities[vulnerability]

                if vulnerability_data.type == model.VulnerabilityType.LOCAL and env.get_vulnerability_index(
                    vulnerability) not in env.get_actuator().get_vulnerabilities_used(node):
                    global_features['owned_local_vulnerabilities_not_exploited'] += 1

        assert global_features == observation_dict['global_features']

        for t in range(num_iterations):
            action = env.sample_random_action()
            _, reward, done, info = env.step(action)
            observation_dict = env.current_observation
            outcome = info['outcome']

            if isinstance(outcome, model.LeakedNodesId):
                # Check that the number of discovered nodes is updated correctly
                # Not incremental for now: global_features['number_discovered_nodes'][0] + & global_features['number_discovered_nodes'] +
                assert observation_dict['global_features']['number_discovered_nodes'][0] == numpy.int32(len(outcome.new_nodes))
                global_features['number_discovered_nodes'][0] = numpy.int32(len(outcome.new_nodes))

            elif isinstance(outcome, model.LeakedCredentials):
                # Check that the number of discovered nodes and credentials is updated correctly
                # If done in an incremental way it should have been: global_features['number_discovered_nodes'][0] + & global_features['number_discovered_nodes'] +
                assert observation_dict['global_features']['number_discovered_nodes'][0] == numpy.int32(len(outcome.new_nodes))
                global_features['number_discovered_nodes'][0] = numpy.int32(len(outcome.new_nodes))
                # Same here: global_features['number_discovered_credentials'][0] + & global_features['number_discovered_credentials'] +
                assert observation_dict['global_features']['number_discovered_credentials'][0] == numpy.int32(len(outcome.new_credentials))
                global_features['number_discovered_credentials'][0] = numpy.int32(len(outcome.new_credentials))

            elif isinstance(outcome, model.LateralMove):
                # Check that lateral move is set to 1
                assert observation_dict['global_features']['lateral_move'] == numpy.int32(1)

            elif isinstance(outcome, model.CustomerData):
                # Check that customer data found is set to 1
                assert observation_dict['global_features']['customer_data_found'] == numpy.int32(1)

            elif isinstance(outcome, model.ProbeSucceeded):
                # Check that probe result is set to 2
                assert observation_dict['global_features']['probe_result'] == numpy.int32(2)

            elif isinstance(outcome, model.ProbeFailed):
                # Check that probe result is set to 1
                assert observation_dict['global_features']['probe_result'] == numpy.int32(1)

            elif isinstance(outcome, model.PrivilegeEscalation):
                # Check that escalation level is updated correctly
                assert observation_dict['global_features']['escalation'] == numpy.int32(outcome.level)

            owned_nodes = env.get_owned_nodes()
            assert observation_dict['global_features']['owned_nodes_length'] == len(owned_nodes)
            discovered_nodes_not_owned = env.get_discovered_not_owned_nodes()
            other_discovered_nodes_not_owned = [node for node in discovered_nodes_not_owned if node != env.target_node_index]
            assert observation_dict['global_features']['discovered_not_owned_nodes_length'] == len(discovered_nodes_not_owned)
            assert observation_dict['global_features']['credential_cache_empty'] == (len(env.get_credential_cache()) == 0)
            values = []
            for node in other_discovered_nodes_not_owned:
                values.append(env.get_node(node).value)

            if len(values) == 0:
                assert observation_dict['global_features']['average_discovered_value'] == 0
            else:
                assert observation_dict['global_features']['average_discovered_value'] == np.mean(values)

            vulnerabilities_used = []
            for node in env.get_owned_nodes():
                node_data = env.get_node(node)
                for vulnerability in node_data.vulnerabilities:
                    if node_data.vulnerabilities[vulnerability].type == model.VulnerabilityType.LOCAL:
                        vulnerabilities_used.append(int(env.get_vulnerability_index(
                            vulnerability) in env.get_actuator().get_vulnerabilities_used(node)))

            assert observation_dict['global_features']['owned_local_vulnerabilities_not_exploited'] == len(vulnerabilities_used) - np.count_nonzero(vulnerabilities_used)

            services_accessible = []
            for node in discovered_nodes_not_owned:
                node_data = env.get_node(node)
                for service in node_data.services:
                    if env.get_port_index(service.name) != -1:
                        services_accessible.append(env.is_service_accessible(service, node))

            assert observation_dict['global_features']['discovered_accessible_ports'] == np.count_nonzero(services_accessible)


            if done:
                break
    print("VVVVV Test passed: global features are updated correctly across episodes!")
    env.close()

@pytest.mark.parametrize('env_name', env_names)
# Test that the credential cache is correctly updated
def test_credential_addition(env_name, num_episodes = 10, num_iterations = 100):
    print("----- Test credential addition to the cache...")
    env = gym.make(env_name)
    for i in range(num_episodes):
        env.reset()
        cache = copy.deepcopy(env.get_credential_cache())
        for t in range(num_iterations):
            action = env.sample_random_action()
            _, reward, done, info = env.step(action)
            outcome = info['outcome']
            if isinstance(outcome, model.LeakedCredentials):
                new_cache = env.get_credential_cache()
                new_creds = [credential for credential in outcome.credentials if
                             credential not in cache]
                assert set(new_cache) == set(cache + new_creds)
                cache = new_cache
    print("VVVVV Test passed: credentials added!")


@pytest.mark.parametrize('env_name', env_names)
# Check correct update of global features
def test_lists_update(env_name, num_episodes = 10, num_iterations = 100):
    print("----- Test updates of lists (owned, discovered, etc.)...")
    env = gym.make(env_name)

    for i in range(num_episodes):
        env.reset()
        owned_nodes = copy.deepcopy(env.get_owned_nodes())
        discovered_nodes = copy.deepcopy(env.get_discovered_nodes())
        discovered_not_owned_nodes = copy.deepcopy(env.get_discovered_not_owned_nodes())
        assert len(owned_nodes) == len(discovered_nodes) == 1
        assert len(discovered_not_owned_nodes) == 0
        assert set(owned_nodes) == set(discovered_nodes) == set([str(env.source_node_index)])

        for t in range(num_iterations):
            action = env.sample_random_action()
            _, reward, done, info = env.step(action)
            outcome = info['outcome']
            if isinstance(outcome, model.LeakedNodesId):
                new_discovered_nodes = env.get_discovered_nodes()
                new_nodes = [node for node in outcome.nodes if node not in discovered_nodes]
                assert set(new_discovered_nodes) == set(discovered_nodes + new_nodes)
                assert len(new_discovered_nodes) == len(discovered_nodes) + len(new_nodes)
                discovered_nodes = copy.deepcopy(new_discovered_nodes)
            elif isinstance(outcome, model.LeakedCredentials):
                new_discovered_nodes = env.get_discovered_nodes()
                new_nodes = set([credential.node for credential in outcome.credentials if credential.node not in discovered_nodes])
                assert set(new_discovered_nodes) == set(list(set(discovered_nodes)) + list(set(new_nodes)))
                assert len(new_discovered_nodes) == len(discovered_nodes) + len(new_nodes)
                discovered_nodes = copy.deepcopy(new_discovered_nodes)
            elif isinstance(outcome, model.LateralMove):
                new_owned_nodes = env.get_owned_nodes()
                if env.source_node_index not in owned_nodes:
                    assert set(owned_nodes + [str(env.source_node_index)]) == set(new_owned_nodes)
                    assert len(new_owned_nodes) == len(owned_nodes) + 1
                else:
                    assert set(owned_nodes) == set(new_owned_nodes)
                    assert len(new_owned_nodes) == len(owned_nodes)
                owned_nodes = copy.deepcopy(new_owned_nodes)
            elif isinstance(outcome, model.Movement):
                new_discovered_nodes = env.get_discovered_nodes()
                new_owned_nodes = env.get_owned_nodes()
                assert set(new_discovered_nodes) == set(discovered_nodes)
                assert len(new_discovered_nodes) == len(discovered_nodes)
                assert set(new_owned_nodes) == set(owned_nodes)
                assert len(new_owned_nodes) == len(owned_nodes)
                discovered_nodes = copy.deepcopy(new_discovered_nodes)
                owned_nodes = copy.deepcopy(new_owned_nodes)
            if done:
                break
    print("VVVVV Test passed: updates of list correct!")
    env.close()

@pytest.mark.parametrize('env_name', random_starting_node_env_names)
# Test the starting node is selected in average randomly
def test_random_node_selection(env_name):
    print("----- Test random node selection...")
    env = gym.make(env_name)
    num_episodes = 1000

    # Track the distribution of started node IDs across episodes
    started_node_ids_distribution = []
    num_unique_nodes = 0
    for _, _ in env.get_nodes():
        num_unique_nodes += 1
    for episode in range(num_episodes):
        env.reset()
        started_node_id = None
        for node_id, node_data in env.get_nodes():
            # A node has both 'agent_installed' and that property or NONE of them is True
            if env_name == "CyberBattleRandomLocal-v0":
                assert (node_data.agent_installed == True) == (
                'breach_node' in node_data.properties)

            if node_data.agent_installed:
                started_node_id = node_id
        assert started_node_id is not None
        started_node_ids_distribution.append(started_node_id)

    unique_started_node_ids, counts = np.unique(started_node_ids_distribution, return_counts=True)
    tolerance = 1  # double at maximum, tolerance for approximate randomness, depend also on the number of episodes
    expected_count = num_episodes / num_unique_nodes
    assert all(abs(count - expected_count) <= tolerance * num_episodes / num_unique_nodes
               for count in counts)
    print("VVVVV Test passed: started node IDs chosen approximately randomly!")


@pytest.mark.parametrize('env_name', env_names)
# Check whether the source node is updated correctly by flipping the target node or remaining constant
def test_nodes_slip(env_name):
    env = gym.make(env_name)
    num_episodes = 100
    num_iterations = 1000
    print("----- Testing source and target nodes changes..")
    for i in range(num_episodes):
        env.reset()
        for t in range(num_iterations):
            action = env.sample_random_action()
            previous_source_node = env.source_node_index
            previous_target_node = env.target_node_index

            _, reward, done, info = env.step(action)
            outcome = info['outcome']

            if isinstance(outcome, model.Movement):
                if outcome.source == 1: # moved source node
                    source_list = copy.deepcopy(env.get_owned_nodes())
                    assert previous_source_node != env.source_node_index
                    if outcome.forward == 1:
                        previous_index = source_list.index(previous_source_node)
                        if previous_index != len(source_list)-1:
                            assert source_list[previous_index + 1] == env.source_node_index
                        else:
                            assert source_list[0] == env.source_node_index
                    else:
                        previous_index = source_list.index(previous_source_node)
                        assert source_list[previous_index - 1] == env.source_node_index
                else: # moved target node
                    assert previous_target_node != env.target_node_index
                    if env.move_target_through_owned:
                        target_list = copy.deepcopy(env.get_owned_nodes())
                        target_list.extend(env.get_discovered_not_owned_nodes())
                    else:
                        target_list = copy.deepcopy(env.get_discovered_not_owned_nodes())
                    if not previous_target_node in target_list: # case discovered not owned only and source == target
                        assert previous_target_node == env.source_node_index
                        if outcome.forward == 1:
                            assert target_list[0] == env.target_node_index
                        else:
                            assert target_list[-1] == env.target_node_index
                    else:
                        previous_index = target_list.index(previous_target_node)
                        if outcome.forward == 1:
                            if previous_index != len(target_list)-1:
                                assert target_list[previous_index+1] == env.target_node_index
                            else:
                                assert target_list[0] == env.target_node_index
                        else:
                            assert target_list[previous_index-1] == env.target_node_index

            elif isinstance(outcome, model.LateralMove):
                # Check that nodes are slipped correctly
                assert previous_target_node == env.source_node_index
            else:
                # Check that the source node remains as anchor
                assert previous_source_node == env.source_node_index
            if done:
                break

    print("VVVVV Test passed: source and target nodes are changed correctly according to the actions!")
    env.close()

@pytest.mark.parametrize('env_name', env_names)
# Check tht SB3 wraps the switcher with this environment
def test_sb3(env_name):
    print("----- StableBaselines3: Checking environment...")
    env = gym.make(env_name)
    keys_a = set(env.observation_space.keys())
    keys_b = set(env.reset().keys())
    missing_keys = keys_a - keys_b  # Keys in env.observation_space not in env.reset()
    extra_keys = keys_b - keys_a  # Keys in env.reset() not in env.observation_space

    print("Keys only in the observation space", missing_keys)
    print("Keys only in the returned dict:", extra_keys)
    check_env(RandomSwitchEnv([env]))
    print("VVVVV StableBaselines3: Checked Environment!")
    time.sleep(1)

@pytest.mark.parametrize('env_name', env_names)
# Check that features are visible at the right level of discovered or owned status
def test_invisibility_of_information(env_name):
    print("----- Testing information invisibility...")
    env = gym.make(env_name)
    num_episodes = 100
    num_iterations = 100

    for i in range(num_episodes):
        env.reset()
        for t in range(num_iterations):
            action = env.sample_random_action()
            _, reward, done, info = env.step(action)
            observation_dict = env.current_observation
            source_node_info = env.get_node(env.source_node_index)
            # The source node should be always owned
            assert source_node_info.agent_installed
            # Check that owned information is fully visible
            assert observation_dict['source_node']['privilege_level'] == int(source_node_info.privilege_level)
            assert observation_dict['source_node']['reimageable'] == int(source_node_info.reimagable)
            assert observation_dict['source_node']['value'] == source_node_info.value
            assert observation_dict['source_node']['status'] == source_node_info.status.value
            assert observation_dict['source_node']['sla_weight'] == source_node_info.sla_weight
            real_vulnerabilities_array = [
                (1, 1, 0.0, 0) for _ in range(env.get_local_attacks_count() + env.get_remote_attacks_count())
            ]
            for vulnerability_id, vulnerability_info in source_node_info.vulnerabilities.items():
                if env.get_vulnerability_index(vulnerability_id) in env.get_actuator().get_discovered_vulnerabilities(env.source_node_index):
                    real_vulnerabilities_array[env.get_vulnerability_index(vulnerability_id)] = (
                        vulnerability_info.type.value+2,
                        model.map_outcome_to_index(vulnerability_info.outcome)+2,
                        vulnerability_info.cost,
                        env.get_vulnerability_index(vulnerability_id) in env.get_actuator().get_vulnerabilities_used(env.source_node_index)
                    )

            observation_dict['source_node']['vulnerabilities_array'] = [(a, b, c[0], d) for a, b, c, d in observation_dict['source_node']['vulnerabilities_array']]
            assert observation_dict['source_node']['vulnerabilities_array'] == real_vulnerabilities_array
            real_property_array = [
                0 for _ in range(env.property_count)
            ]
            for property in source_node_info.properties:
                if env.get_property_index(property) != -1:
                    real_property_array[env.get_property_index(property)] = 1
            assert observation_dict['source_node']['property_array'] == real_property_array
            real_firewall_array = [0 for _ in range(env.get_port_count())]
            real_firewall_array.extend([0 for _ in range(env.get_port_count())])
            for config in source_node_info.firewall.incoming:
                # incoming
                permission = config.permission.value
                if env.get_port_index(config.port) != -1:
                    real_firewall_array[env.get_port_index(config.port)] = permission
            for config in source_node_info.firewall.outgoing:
                # outgoing
                permission = config.permission.value
                if env.get_port_index(config.port) != -1:
                    real_firewall_array[env.get_port_count() + env.get_port_index(config.port)] = permission
            assert observation_dict['source_node']['firewall_config_array'] == real_firewall_array
            real_listening_services_array = [
                (0, 0, 0, numpy.array([0.0], dtype=numpy.float64)) for _ in range(env.port_count)
            ]

            for service in source_node_info.services:
                if env.get_port_index(service.name) != -1:
                    real_listening_services_array[env.get_port_index(service.name)] = (
                        int(service.running),
                        env.is_service_accessible(service, env.source_node_index),
                        int(service.name in env.get_actuator().get_ports_surely_working(env.source_node_index)),
                        numpy.array([service.sla_weight], dtype=numpy.float64)
                    )
            assert observation_dict['source_node']['listening_services_array'] == real_listening_services_array
            # Target node may be just discovered or also owned
            target_node_info = env.get_node(env.target_node_index)
            if env.source_node_index == env.target_node_index:
                # If owned check that all information is visible
                assert observation_dict['target_node']['privilege_level'] == 0
                assert observation_dict['target_node']['reimageable'] == 0
                assert observation_dict['target_node']['value'] == 0
                assert observation_dict['target_node']['status'] == 0
                assert observation_dict['target_node']['sla_weight'] == 0
                real_vulnerabilities_array = [
                    (0, 0, 0.0, 0) for _ in range(env.get_local_attacks_count() + env.get_remote_attacks_count())
                ]
                observation_dict['target_node']['vulnerabilities_array'] = [(a, b, c[0], d) for a, b, c, d in
                                                                            observation_dict['target_node'][
                                                                                'vulnerabilities_array']]
                assert observation_dict['target_node']['vulnerabilities_array'] == real_vulnerabilities_array
                real_property_array = [
                    0 for _ in range(env.property_count)
                ]
                assert observation_dict['target_node']['property_array'] == real_property_array
                real_firewall_array = [0 for _ in range(env.get_port_count())]
                real_firewall_array.extend([0 for _ in range(env.get_port_count())])
                assert observation_dict['target_node']['firewall_config_array'] == real_firewall_array
                real_listening_services_array = [
                    (0, 0, 0, numpy.array([0.0], dtype=numpy.float64)) for _ in range(env.port_count)
                ]
                assert observation_dict['target_node']['listening_services_array'] == real_listening_services_array
            elif target_node_info.agent_installed:
                # If owned check that all information is visible
                assert observation_dict['target_node']['privilege_level'] == int(target_node_info.privilege_level)
                assert observation_dict['target_node']['reimageable'] == int(target_node_info.reimagable)
                assert observation_dict['target_node']['value'] == target_node_info.value
                assert observation_dict['target_node']['status'] == target_node_info.status.value
                assert observation_dict['target_node']['sla_weight'] == target_node_info.sla_weight
                real_vulnerabilities_array = [
                    (1, 1, 0.0, 0) for _ in range(env.get_local_attacks_count() + env.get_remote_attacks_count())
                ]
                for vulnerability_id, vulnerability_info in target_node_info.vulnerabilities.items():
                    if env.get_vulnerability_index(vulnerability_id) in env.get_actuator().get_discovered_vulnerabilities(env.target_node_index):
                        real_vulnerabilities_array[env.get_vulnerability_index(vulnerability_id)] = (
                            vulnerability_info.type.value+2,
                            model.map_outcome_to_index(vulnerability_info.outcome)+2,
                            vulnerability_info.cost,
                            env.get_vulnerability_index(vulnerability_id) in env.get_actuator().get_vulnerabilities_used(env.target_node_index)
                        )
                observation_dict['target_node']['vulnerabilities_array'] = [(a, b, c[0], d) for a, b, c, d in
                                                                            observation_dict['target_node'][
                                                                                'vulnerabilities_array']]
                assert observation_dict['target_node']['vulnerabilities_array'] == real_vulnerabilities_array
                real_property_array = [
                    0 for _ in range(env.property_count)
                ]
                for property in target_node_info.properties:
                    if env.get_property_index(property) != -1:
                        real_property_array[env.get_property_index(property)] = 1
                assert observation_dict['target_node']['property_array'] == real_property_array
                real_firewall_array = [0 for _ in range(env.get_port_count())]
                real_firewall_array.extend([0 for _ in range(env.get_port_count())])
                for config in target_node_info.firewall.incoming:
                    incoming_outgoing = 0
                    permission = config.permission.value
                    if env.get_port_index(config.port) != -1:
                        real_firewall_array[env.get_port_index(config.port)] = permission
                for config in target_node_info.firewall.outgoing:
                    incoming_outgoing = 1
                    permission = config.permission.value
                    if env.get_port_index(config.port) != -1:
                        real_firewall_array[env.get_port_count() + env.get_port_index(config.port)] = permission
                assert observation_dict['target_node']['firewall_config_array'] == real_firewall_array
                real_listening_services_array = [
                    (0, 0, 0, numpy.array([0.0], dtype=numpy.float32)) for _ in range(env.port_count)
                ]

                for service in target_node_info.services:
                    if env.get_port_index(service.name) != -1:
                        real_listening_services_array[env.get_port_index(service.name)] = (
                            int(service.running),
                            env.is_service_accessible(service, env.target_node_index),
                            int(service.name in env.get_actuator().get_ports_surely_working(env.target_node_index)),
                            numpy.array([service.sla_weight], dtype=numpy.float32)
                        )
                assert observation_dict['target_node']['listening_services_array'] == real_listening_services_array
            else:
                # If discovered only, check that only the correct subset of information is visible
                real_vulnerabilities_array = [
                    (0, 0, 0.0, 0) for _ in range(env.get_local_attacks_count() + env.get_remote_attacks_count())
                ]
                for vulnerability_id, vulnerability_info in target_node_info.vulnerabilities.items():
                    if env.get_vulnerability_index(
                        vulnerability_id) in env.get_actuator().get_discovered_vulnerabilities(env.target_node_index):
                        real_vulnerabilities_array[env.get_vulnerability_index(vulnerability_id)] = (
                            vulnerability_info.type.value+2,
                            model.map_outcome_to_index(vulnerability_info.outcome)+2,
                            vulnerability_info.cost,
                            env.get_vulnerability_index(vulnerability_id) in env.get_actuator().get_vulnerabilities_used(env.target_node_index)
                        )

                for vulnerability_index in env.get_actuator().get_absence_discovered_vulnerabilities(env.target_node_index):
                    real_vulnerabilities_array[vulnerability_index] = (
                        1,
                        1,
                        0.0,
                        env.get_vulnerability_index(vulnerability_index) in env.get_actuator().get_vulnerabilities_used(env.target_node_index)
                    )

                observation_dict['target_node']['vulnerabilities_array'] = [(a, b, c[0], d) for a, b, c, d in
                                                                            observation_dict['target_node'][
                                                                                'vulnerabilities_array']]
                assert observation_dict['target_node']['vulnerabilities_array'] == real_vulnerabilities_array
                actuator = env.get_actuator()
                for index, elem in enumerate(observation_dict['target_node']['property_array']):
                    if elem == 1:
                        assert index in actuator.get_discovered_properties(env.target_node_index)
                    if elem == 0:
                        assert index not in actuator.get_discovered_properties(env.target_node_index)
                real_firewall_array = [0 for _ in range(env.get_port_count())]
                real_firewall_array.extend([0 for _ in range(env.get_port_count())])
                for config in target_node_info.firewall.incoming:
                    permission = config.permission.value
                    if env.get_port_index(config.port) != -1:
                        real_firewall_array[env.get_port_index(config.port)] = permission
                for config in target_node_info.firewall.outgoing:
                    permission = config.permission.value
                    if env.get_port_index(config.port) != -1:
                        real_firewall_array[env.get_port_count() + env.get_port_index(config.port)] = permission

                assert observation_dict['target_node']['firewall_config_array'] == real_firewall_array

                real_listening_services_array = [
                    (0,0, 0, numpy.array([0.0], dtype=numpy.float64)) for _ in range(env.port_count)
                ]

                for service in target_node_info.services:
                    if env.get_port_index(service.name) != -1:
                        real_listening_services_array[env.get_port_index(service.name)] = (
                            int(service.running),
                            env.is_service_accessible(service, env.target_node_index),
                            int(service.name in env.get_actuator().get_ports_surely_working(env.target_node_index)),
                            numpy.array([service.sla_weight], dtype=numpy.float64)
                        )
                assert observation_dict['target_node']['listening_services_array'] == real_listening_services_array

            if done:
                break

    print("VVVVV Test passed: visibility of the feature vector is properly handled!")
    env.close()

@pytest.mark.parametrize('env_name', env_names)
# Check whether the max num iterations is corrected and same when the attacker goal is reached
def test_max_num_iterations(env_name, num_episodes = 100, num_iterations = 10000):
    print("----- Testing that the maximum number of iterations is respected...")
    env = gym.make(env_name)
    env.winning_reward = 1000
    env.set_isolation_filter_threshold(0)
    for i in range(num_episodes):
        cut_off = random.randint(1,1000)
        env.set_cut_off(cut_off)
        env.reset()
        if env.env_type == "CyberBattleRandom":
            shortest_paths_starter = copy.deepcopy(env.shortest_paths_starter)
            reachable_count = sum(1 for value in shortest_paths_starter.values() if value is not None)
        count_iterations = 0
        for t in range(num_iterations):
            action = env.sample_random_action()
            _, reward, done, info = env.step(action)
            count_iterations += 1
            if done:
                break
        if env.env_type == "CyberBattleRandom":
            for node_id in env.get_owned_nodes():
                shortest_paths_starter.pop(int(node_id), None)
                # Check if all remaining paths are None ( == not reachable )
            if all(value is None for value in shortest_paths_starter.values()):
                print("Conquered all!")
                assert reward >= env.winning_reward
                assert count_iterations <= cut_off
                owned = len(env.get_owned_nodes())
                assert owned == reachable_count
            else:
                assert reward < env.winning_reward
                assert count_iterations == cut_off
        else:
            assert count_iterations == cut_off
    print("VVVVV Tested that the maximum number of iterations is respected!")

# test remote vulnerabilities and port connection when it is known that they should not work
@pytest.mark.parametrize('env_name', observable_env_names)
def test_fatal_actions(env_name, num_episodes = 100, num_iterations = 100):
    print("----- Testing connection or remote actions when they are not valid...")
    env = gym.make(env_name)
    # connections
    for i in range(num_episodes):
        env.reset()
        for t in range(num_iterations):
            while True:
                action = env.sample_random_action()
                if env.get_action_name(action) in env.identifiers.ports:
                    break
            _, reward, done, info = env.step(action)
            if env.source_node_index == env.target_node_index:
                assert reward <= 0
                assert info['outcome'] == None
            elif len(env.get_credential_cache()) == 0:
                assert reward <= 0
                assert info['outcome'] == None
            elif not env.is_service_accessible(action, env.target_node_index):
                assert reward <= 0
                assert info['outcome'] == None
    # remote vulns
    for i in range(num_episodes):
        env.reset()
        for t in range(num_iterations):
            while True:
                action = env.sample_random_action()
                if env.get_action_name(action) in env.identifiers.remote_vulnerabilities:
                    break
            _, reward, done, info = env.step(action)
            if env.source_node_index == env.target_node_index:
                assert reward <= 0
                assert info['outcome'] == None
    print("VVVVV Tested connection or remote actions when they are not valid!")


@pytest.mark.parametrize('env_name', observable_env_names)
# test that when a movement invalid is selected, the outcome embeds this information
def test_invalid_movement(env_name, num_episodes=100, num_iterations=100):
    print("----- Testing movements in special cases...")
    env = gym.make(env_name)
    source_movements = ["source node selection forward", "source node selection backward"]
    target_movements = ["target node selection forward", "target node selection backward"]
    # connections
    for i in range(num_episodes):
        env.reset()
        for t in range(num_iterations):
            while True:
                action = env.sample_random_action()
                if env.get_action_name(action) in source_movements or env.get_action_name(action) in target_movements:
                    break
            previous_source_node = copy.deepcopy(env.source_node_index)
            _, reward, done, info = env.step(action)
            if env.get_action_name(action) in source_movements and len(env.get_owned_nodes()) == 1:
                assert info['outcome'] == None
            if len(env.get_discovered_not_owned_nodes()) == 0 and (env.get_action_name(action) in target_movements or env.get_action_name(action) in source_movements):
                assert env.source_node_index == env.target_node_index
                assert env.target_node_index in env.get_owned_nodes()
                assert info['outcome'] == None
                if len(env.get_owned_nodes()) > 1:
                    assert env.source_node_index != previous_source_node
            if env.get_action_name(action) in target_movements and len(env.get_discovered_not_owned_nodes()) > 0:
                assert env.target_node_index in env.get_discovered_not_owned_nodes()

    print("VVVVV Testing movements in special cases...")


@pytest.mark.parametrize('env_name', observable_env_names)
# Check that features are visible at all levels of discovered or owned status for the non partially observable solution
def test_visibility_of_information(env_name, num_episodes = 100, num_iterations = 100):
    print("----- Testing the visibility in a fully observable case...")
    env = gym.make(env_name)

    for i in range(num_episodes):
        env.reset()
        for t in range(num_iterations):
            action = env.sample_random_action()
            _, reward, done, info = env.step(action)
            observation_dict = env.current_observation
            source_node_info = env.get_node(env.source_node_index)
            # The source node should be always owned
            assert source_node_info.agent_installed
            # Check that owned information is fully visible
            assert observation_dict['source_node']['privilege_level'] == int(source_node_info.privilege_level)
            assert observation_dict['source_node']['reimageable'] == int(source_node_info.reimagable)
            assert observation_dict['source_node']['value'] == source_node_info.value
            assert observation_dict['source_node']['status'] == source_node_info.status.value
            assert observation_dict['source_node']['sla_weight'] == source_node_info.sla_weight
            real_vulnerabilities_array = [
                (0, 0, 0.0, 0) for _ in range(env.get_local_attacks_count() + env.get_remote_attacks_count())
            ]
            for vulnerability_id, vulnerability_info in source_node_info.vulnerabilities.items():
                real_vulnerabilities_array[env.get_vulnerability_index(vulnerability_id)] = (
                    vulnerability_info.type.value,
                    model.map_outcome_to_index(vulnerability_info.outcome),
                    vulnerability_info.cost,
                    env.get_vulnerability_index(vulnerability_id) in env.get_actuator().get_vulnerabilities_used(env.source_node_index)
                )
            observation_dict['source_node']['vulnerabilities_array'] = [(a, b, c[0], d) for a, b, c, d in observation_dict['source_node']['vulnerabilities_array']]
            assert observation_dict['source_node']['vulnerabilities_array'] == real_vulnerabilities_array
            real_property_array = [
                0 for _ in range(env.property_count)
            ]
            for property in source_node_info.properties:
                if env.get_property_index(property) != -1:
                    real_property_array[env.get_property_index(property)] = 1
            assert observation_dict['source_node']['property_array'] == real_property_array
            real_firewall_array = [0 for _ in range(env.get_port_count())]
            real_firewall_array.extend([0 for _ in range(env.get_port_count())])
            for config in source_node_info.firewall.incoming:
                permission = config.permission.value
                if env.get_port_index(config.port) != -1:
                    real_firewall_array[env.get_port_index(config.port)] = permission
            for config in source_node_info.firewall.outgoing:
                permission = config.permission.value
                if env.get_port_index(config.port) != -1:
                    real_firewall_array[env.get_port_count() + env.get_port_index(config.port)] = permission
            assert observation_dict['source_node']['firewall_config_array'] == real_firewall_array
            real_listening_services_array = [
                (0, 0, 0, numpy.array([0.0], dtype=numpy.float64)) for _ in range(env.port_count)
            ]

            for service in source_node_info.services:
                if env.get_port_index(service.name) != -1:
                    real_listening_services_array[env.get_port_index(service.name)] = (
                        int(service.running),
                        env.is_service_accessible(service, env.source_node_index),
                        int(service.name in env.get_actuator().get_ports_surely_working(env.source_node_index)),
                        numpy.array([service.sla_weight], dtype=numpy.float64)
                    )
            assert observation_dict['source_node']['listening_services_array'] == real_listening_services_array

            if env.source_node_index == env.target_node_index:
                # If owned check that all information is visible
                assert observation_dict['target_node']['privilege_level'] == 0
                assert observation_dict['target_node']['reimageable'] == 0
                assert observation_dict['target_node']['value'] == 0
                assert observation_dict['target_node']['status'] == 0
                assert observation_dict['target_node']['sla_weight'] == 0
                real_vulnerabilities_array = [
                    (0, 0, 0.0, 0) for _ in range(env.get_local_attacks_count() + env.get_remote_attacks_count())
                ]
                observation_dict['target_node']['vulnerabilities_array'] = [(a, b, c[0], d) for a, b, c, d in
                                                                            observation_dict['target_node'][
                                                                                'vulnerabilities_array']]
                assert observation_dict['target_node']['vulnerabilities_array'] == real_vulnerabilities_array
                real_property_array = [
                    0 for _ in range(env.property_count)
                ]
                assert observation_dict['target_node']['property_array'] == real_property_array
                real_firewall_array = [0 for _ in range(env.get_port_count())]
                real_firewall_array.extend([0 for _ in range(env.get_port_count())])
                assert observation_dict['target_node']['firewall_config_array'] == real_firewall_array
                real_listening_services_array = [
                    (0, 0, 0, numpy.array([0.0], dtype=numpy.float64)) for _ in range(env.port_count)
                ]
                assert observation_dict['target_node']['listening_services_array'] == real_listening_services_array
            else:
                # Target node may be just discovered or also owned
                target_node_info = env.get_node(env.target_node_index)
                # If owned check that all information is visible
                assert observation_dict['target_node']['privilege_level'] == int(target_node_info.privilege_level)
                assert observation_dict['target_node']['reimageable'] == int(target_node_info.reimagable)
                assert observation_dict['target_node']['value'] == target_node_info.value
                assert observation_dict['target_node']['status'] == target_node_info.status.value
                assert observation_dict['target_node']['sla_weight'] == target_node_info.sla_weight
                real_vulnerabilities_array = [
                    (0, 0, 0.0, 0) for _ in range(env.get_local_attacks_count() + env.get_remote_attacks_count())
                ]
                for vulnerability_id, vulnerability_info in target_node_info.vulnerabilities.items():
                    real_vulnerabilities_array[env.get_vulnerability_index(vulnerability_id)] = (
                        vulnerability_info.type.value,
                        model.map_outcome_to_index(vulnerability_info.outcome),
                        vulnerability_info.cost,
                        env.get_vulnerability_index(vulnerability_id) in env.get_actuator().get_vulnerabilities_used(env.target_node_index)
                    )
                observation_dict['target_node']['vulnerabilities_array'] = [(a, b, c[0], d) for a, b, c, d in
                                                                                observation_dict['target_node'][
                                                                                    'vulnerabilities_array']]
                assert observation_dict['target_node']['vulnerabilities_array'] == real_vulnerabilities_array
                real_property_array = [
                        0 for _ in range(env.property_count)
                ]
                for property in target_node_info.properties:
                    if env.get_property_index(property) != -1:
                           real_property_array[env.get_property_index(property)] = 1
                assert observation_dict['target_node']['property_array'] == real_property_array
                real_firewall_array = [0 for _ in range(env.get_port_count())]
                real_firewall_array.extend([0 for _ in range(env.get_port_count())])
                for config in target_node_info.firewall.incoming:
                    permission = config.permission.value
                    if env.get_port_index(config.port) != -1:
                        real_firewall_array[env.get_port_index(config.port)] = permission
                for config in target_node_info.firewall.outgoing:
                    permission = config.permission.value
                    if env.get_port_index(config.port) != -1:
                        real_firewall_array[env.get_port_count() + env.get_port_index(config.port)] = permission
                assert observation_dict['target_node']['firewall_config_array'] == real_firewall_array

                real_listening_services_array = [
                    (0, 0, 0, numpy.array([0.0], dtype=numpy.float64)) for _ in range(env.port_count)
                ]

                for service in target_node_info.services:
                    if env.get_port_index(service.name) != -1:
                        real_listening_services_array[env.get_port_index(service.name)] = (
                            int(service.running),
                            env.is_service_accessible(service, env.target_node_index),
                            int(service.name in env.get_actuator().get_ports_surely_working(env.target_node_index)),
                            numpy.array([service.sla_weight], dtype=numpy.float64)
                        )
                assert observation_dict['target_node']['listening_services_array'] == real_listening_services_array
            if done:
                break

    print("VVVVV Test passed: visibility of the feature vector in fully observable solution is properly handled!")
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test CyberBattleLocalEnv!")
    parser.add_argument('--render', default=False, help='Rendering option', action="store_true")
    parser.add_argument('--print', default=False,
                        help='Print dynamically the list of nodes owned, discovered, not discovered', action="store_true")
    args = parser.parse_args()

    for env_name in env_names:
        print("----- Testing environment", env_name, "...")
        test_wrap_spec(env_name)
        test_sb3(env_name)


        test_few_gym_iterations(env_name, 10, 1000, args.render, args.print)
        if env_name in random_starting_node_env_names:
            test_random_node_selection(env_name)
        test_invalid_movement(env_name)
        test_fatal_actions(env_name)
        test_max_num_iterations(env_name)
        test_global_features(env_name)
        test_observation_conversion(env_name)
        test_nodes_slip(env_name)
        test_normal_action_selection(env_name)
        test_starter_node(env_name)
        test_credential_addition(env_name)
        test_lists_update(env_name)

    partially_observable_env_names = env_names
    for env_name in partially_observable_env_names:
        print("----- Testing environment", env_name, "...")
        test_invisibility_of_information(env_name)

    for env_name in observable_env_names:
        print("----- Testing environment", env_name, "...")
        test_visibility_of_information(env_name)

    for env_name in probabilistic_action_env_names:
        print("----- Testing environment", env_name, "...")
        test_probabilistic_action_selection(env_name)
