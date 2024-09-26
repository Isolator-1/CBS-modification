# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

""" Generating random graphs"""
import copy

from cyberbattle.simulation.model import Identifiers, NodeID, CredentialID, PortName, FirewallConfiguration, FirewallRule, RulePermission
import numpy as np
import networkx as nx
from cyberbattle.simulation import model as m
import random
from typing import List, Optional, Tuple, DefaultDict

from collections import defaultdict

ENV_IDENTIFIERS = Identifiers(
    properties=[
        'breach_node'
    ],
    ports=['SMB', 'HTTP', 'RDP'],
    local_vulnerabilities=[
        'ScanWindowsCredentialManagerForRDP',
        'ScanWindowsExplorerRecentFiles',
        'ScanWindowsCredentialManagerForSMB'
    ],
    remote_vulnerabilities=[
        'Traceroute'
    ]
)

def generate_random_traffic_network(
    n_clients: int = 50,
    n_servers={
        "SMB": 15,
        "HTTP": 15,
        "RDP": 15,
    },
    seed: Optional[int] = None,
    tolerance: np.float32 = np.float32(1e-3),
    alpha=np.array([(0.1, 0.3), (0.18, 0.09)], dtype=float),
    beta=np.array([(100, 10), (10, 100)], dtype=float),
) -> nx.DiGraph:
    """
    Randomly generate a directed multi-edge network graph representing
    fictitious SMB, HTTP, and RDP traffic.

    Arguments:
        n_clients: number of workstation nodes that can initiate sessions with server nodes
        n_servers: dictionary indicatin the numbers of each nodes listening to each protocol
        seed: seed for the psuedo-random number generator
        tolerance: absolute tolerance for bounding the edge probabilities in [tolerance, 1-tolerance]
        alpha: beta distribution parameters alpha such that E(edge prob) = alpha / beta
        beta: beta distribution parameters beta such that E(edge prob) = alpha / beta

    Returns:
        (nx.classes.multidigraph.MultiDiGraph): the randomly generated network from the hierarchical block model
    """
    edges_labels = defaultdict(set)  # set backed multidict

    for protocol in list(n_servers.keys()):
        sizes = [n_clients, n_servers[protocol]]
        # sample edge probabilities from a beta distribution
        np.random.seed(seed)
        probs: np.ndarray = np.random.beta(a=alpha, b=beta, size=(2, 2))

        # scale by edge type
        if protocol == "SMB":
            probs = 3 * probs
        if protocol == "RDP":
            probs = 4 * probs

        # don't allow probs too close to zero or one
        probs = np.clip(probs, a_min=tolerance, a_max=np.float32(1.0 - tolerance))
        # sample edges using block models given edge probabilities
        di_graph_for_protocol = nx.stochastic_block_model(
            sizes=sizes, p=probs, directed=True, seed=seed)

        for edge in di_graph_for_protocol.edges:
            edges_labels[edge].add(protocol)

    digraph = nx.DiGraph()
    for (u, v), port in list(edges_labels.items()):
        digraph.add_edge(u, v, protocol=port)
    return digraph

def generate_random_traffic_network_per_protocol(
    n_clients,
    n_servers,
    seed,
    tolerance,
    alpha,
    beta
) -> nx.DiGraph:
    """
    Randomly generate a directed multi-edge network graph representing
    fictitious SMB, HTTP, and RDP traffic.

    Arguments:
        n_clients: number of workstation nodes that can initiate sessions with server nodes
        n_servers: dictionary indicatin the numbers of each nodes listening to each protocol
        seed: seed for the psuedo-random number generator
        tolerance: absolute tolerance for bounding the edge probabilities in [tolerance, 1-tolerance]
        alpha: beta distribution parameters alpha such that E(edge prob) = alpha / beta
        beta: beta distribution parameters beta such that E(edge prob) = alpha / beta

    Returns:
        (nx.classes.multidigraph.MultiDiGraph): the randomly generated network from the hierarchical block model
    """
    edges_labels = defaultdict(set)  # set backed multidict

    for protocol in list(n_servers.keys()):
        sizes = [n_clients, n_servers[protocol]]
        # sample edge probabilities from a beta distribution
        np.random.seed(seed)
        probs: np.ndarray = np.random.beta(a=alpha[protocol], b=beta[protocol], size=(2, 2))

        # don't allow probs too close to zero or one
        probs = np.clip(probs, a_min=tolerance, a_max=np.float32(1.0 - tolerance))
        # sample edges using block models given edge probabilities
        di_graph_for_protocol = nx.stochastic_block_model(
            sizes=sizes, p=probs, directed=True, seed=seed)

        for edge in di_graph_for_protocol.edges:
            edges_labels[edge].add(protocol)

    digraph = nx.DiGraph()

    # add all nodes before, otherwise only connected nodes will be added
    for node in range(n_clients + max(n_servers["SMB"], n_servers["HTTP"], n_servers["RDP"])):
        if not digraph.has_node(node):
            digraph.add_node(node)

    for (u, v), port in list(edges_labels.items()):
        digraph.add_edge(u, v, protocol=port)
    print(digraph)
    return digraph


def cyberbattle_model_from_traffic_graph(
    traffic_graph: nx.DiGraph,
    cached_smb_password_probability=0.75,
    cached_rdp_password_probability=0.8,
    cached_accessed_network_shares_probability=0.6,
    cached_password_has_changed_probability=0.1,
    traceroute_discovery_probability=0.5,
    probability_two_nodes_use_same_password_to_access_given_resource=0.8,
    firewall_rule_incoming_probability=0.2,
    firewall_rule_outgoing_probability=0.2
) -> [nx.DiGraph, int]:
    """Generate a random CyberBattle network model from a specified traffic (directed multi) graph.

    The input graph can for instance be generated with `generate_random_traffic_network`.
    Each edge of the input graph indicates that a communication took place
    between the two nodes with the protocol specified in the edge label.

    Returns a CyberBattle network with the same nodes and implanted vulnerabilities
    to be used to instantiate a CyverBattleSim gym.

    Arguments:

    cached_smb_password_probability, cached_rdp_password_probability:
        probability that a password used for authenticated traffic was cached by the OS for SMB and RDP
    cached_accessed_network_shares_probability:
        probability that a network share accessed by the system was cached by the OS
    cached_password_has_changed_probability:
        probability that a given password cached on a node has been rotated on the target node
        (typically low has people tend to change their password infrequently)
    probability_two_nodes_use_same_password_to_access_given_resource:
        as the variable name says
    traceroute_discovery_probability:
        probability that a target node of an SMB/RDP connection get exposed by a traceroute attack
    """
    # convert node IDs to string
    graph = nx.relabel_nodes(traffic_graph, {i: str(i) for i in traffic_graph.nodes})
    access_graph = copy.deepcopy(graph)
    access_graph.clear_edges()
    knows_graph = copy.deepcopy(graph)
    knows_graph.clear_edges()

    password_counter: int = 0

    def generate_password() -> CredentialID:
        nonlocal password_counter
        password_counter = password_counter + 1
        return f'unique_pwd{password_counter}'

    def traffic_targets(source_node: NodeID, protocol: str) -> List[NodeID]:
        neighbors = [t for (s, t) in graph.edges()
                     if s == source_node and protocol in graph.edges[(s, t)]['protocol']]
        return neighbors

    # Map (node, port name) -> assigned pwd
    assigned_passwords: DefaultDict[Tuple[NodeID, PortName],
                                    List[CredentialID]] = defaultdict(list)

    def assign_new_valid_password(node: NodeID, port: PortName) -> CredentialID:
        pwd = generate_password()
        assigned_passwords[node, port].append(pwd)
        return pwd

    def reuse_valid_password(node: NodeID, port: PortName) -> CredentialID:
        """Reuse a password already assigned to that node an port, if none is already
         assigned create and assign a new valid password"""
        if (node, port) not in assigned_passwords:
            return assign_new_valid_password(node, port)

        # reuse any of the existing assigned valid password for that node/port
        return random.choice(assigned_passwords[node, port])

    def create_cached_credential(node: NodeID, port: PortName) -> CredentialID:
        if random.random() < cached_password_has_changed_probability:
            # generate a new invalid password
            return generate_password(), False
        else:
            if random.random() < probability_two_nodes_use_same_password_to_access_given_resource:
                return reuse_valid_password(node, port), True
            else:
                return assign_new_valid_password(node, port), True

    def add_leak_neighbors_vulnerability(
            node_id: m.NodeID,
            library: Optional[m.VulnerabilityLibrary] = None) -> m.VulnerabilityLibrary:
        """Create random vulnerabilities that reveals immediate traffic neighbors from a given node"""

        if not library:
            library = {}

        rdp_neighbors = traffic_targets(node_id, 'RDP')

        # Generation of RDP vulnerability based on probability with update of knows graph edges
        if len(rdp_neighbors) > 0:
            leaked_credentials = []
            for target_node in rdp_neighbors:
                if random.random() < cached_rdp_password_probability:
                    credential, valid = create_cached_credential(target_node, 'RDP')
                    leaked_credentials.append(
                        m.CachedCredential(node=target_node, port='RDP',
                                           credential=credential, valid=valid)
                    )
                    knows_graph.add_edge(node_id, target_node,
                                         vulnerability='ScanWindowsCredentialManagerForRDP')
            library['ScanWindowsCredentialManagerForRDP'] = m.VulnerabilityInfo(
                description="Look for RDP credentials in the Windows Credential Manager",
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedCredentials(credentials=leaked_credentials),
                reward_string="Discovered creds in the Windows Credential Manager",
                cost=2.0
            )

        smb_neighbors = traffic_targets(node_id, 'SMB')

        # Generation of SMB vulnerability based on probability with update of knows graph edges
        if len(smb_neighbors) > 0:
            leaked_node_ids = []
            for target_node in smb_neighbors:
                if random.random() < cached_accessed_network_shares_probability:
                    leaked_node_ids.append(
                        target_node
                    )
                    knows_graph.add_edge(node_id, target_node,
                                                 vulnerability='ScanWindowsExplorerRecentFiles')
            leaked_credentials = []
            for target_node in smb_neighbors:
                if random.random() < cached_smb_password_probability:
                    credential, valid = create_cached_credential(target_node, 'SMB')
                    leaked_credentials.append(
                        m.CachedCredential(node=target_node, port='SMB',
                                       credential=credential, valid=valid)
                    )
                    knows_graph.add_edge(node_id, target_node,
                                        vulnerability='ScanWindowsCredentialManagerForSMB')
            library['ScanWindowsExplorerRecentFiles'] = m.VulnerabilityInfo(
                description="Look for network shares in the Windows Explorer Recent files",
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedNodesId(leaked_node_ids),
                reward_string="Windows Explorer Recent Files revealed network shares",
                cost=1.0
            )

            library['ScanWindowsCredentialManagerForSMB'] = m.VulnerabilityInfo(
                description="Look for network credentials in the Windows Credential Manager",
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedCredentials(credentials=leaked_credentials),
                reward_string="Discovered SMB creds in the Windows Credential Manager",
                cost=2.0
            )

        # Generation of RDP-SMB (Traceroute) vulnerability based on probability with update of knows graph edges
        if len(smb_neighbors) > 0 and len(rdp_neighbors) > 0:
            leaked_node_ids = []
            for target_node in smb_neighbors or rdp_neighbors:
                if random.random() < traceroute_discovery_probability:
                    leaked_node_ids.append(
                        target_node
                    )
                    knows_graph.add_edge(node_id, target_node,
                                         vulnerability='Traceroute')
            library['Traceroute'] = m.VulnerabilityInfo(
                description="Attempt to discover network nodes using Traceroute",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedNodesId(leaked_node_ids),
                reward_string="Discovered new network nodes via traceroute",
                cost=5.0
            )

        return library

    def create_vulnerabilities_from_traffic_data(node_id: m.NodeID):
        return add_leak_neighbors_vulnerability(node_id=node_id)

    # default firewall with no BLOCK rules
    firewall_conf = FirewallConfiguration(
        [FirewallRule("RDP", RulePermission.ALLOW), FirewallRule("SMB", RulePermission.ALLOW), FirewallRule("HTTP", RulePermission.ALLOW)],
        [FirewallRule("RDP", RulePermission.ALLOW), FirewallRule("SMB", RulePermission.ALLOW), FirewallRule("HTTP", RulePermission.ALLOW)])

    # default node features
    def create_node_data_without_vulnerabilities():
        return m.NodeInfo(
            services=[],  # will be filled later according to the overall credentials created
            value=random.randint(0, 100),
            agent_installed=False,
            firewall=copy.deepcopy(firewall_conf)
        )

    # Step 1: Create all the nodes with associated value and firewall configuration
    for node in list(graph.nodes):
        graph.nodes[node].clear()
        graph.nodes[node].update({'data': create_node_data_without_vulnerabilities()})

    # Step 2: Assign vulnerabilities to each node
    for node in list(graph.nodes):
        node_data = graph.nodes[node]['data']
        node_data.vulnerabilities = create_vulnerabilities_from_traffic_data(node)
        graph.nodes[node].update({'data': node_data})

    # Assign services aposteriori based on passwords generated (consequence of vulnerabilities assigned)
    for node in list(graph.nodes):
        graph.nodes[node]['data'].services = [m.ListeningService(name=port, allowedCredentials=assigned_passwords[(target_node, port)])
                                              for (target_node, port) in assigned_passwords.keys()
                                              if target_node == node
                                              ]

    # Assign firewall a-posteriori, generating probabilistically some firewall rules
    for node in list(graph.nodes):
        service_names = [service.name for service in graph.nodes[node]['data'].services]
        for index, rule in enumerate(graph.nodes[node]['data'].firewall.incoming):
            if rule.port in service_names:
                if random.random() < firewall_rule_incoming_probability:
                    graph.nodes[node]['data'].firewall.incoming[index].permission = RulePermission.BLOCK
                else:
                    graph.nodes[node]['data'].firewall.incoming[index].permission = RulePermission.ALLOW
        for index, rule in enumerate(graph.nodes[node]['data'].firewall.outgoing):
            if rule.port in ["RDP", "SMB", "HTTP"]:
                if random.random() < firewall_rule_outgoing_probability:
                    graph.nodes[node]['data'].firewall.outgoing[index].permission = RulePermission.BLOCK
                else:
                    graph.nodes[node]['data'].firewall.outgoing[index].permission = RulePermission.ALLOW

    # Determine access graph after firewall and credentials' generation (with possibility to be invalid)
    for node in list(graph.nodes):
        for vulnerability_id in graph.nodes[node]['data'].vulnerabilities:
            vulnerability = graph.nodes[node]['data'].vulnerabilities[vulnerability_id]
            if isinstance(vulnerability.outcome, m.LeakedCredentials):
                for credential in vulnerability.outcome.credentials:
                    # if the vulnerability gives a valid credential, update the access graph
                    if credential.valid:
                        firewall_outgoing = graph.nodes[node]['data'].firewall.outgoing
                        # by default 3 steps are needed at minimum to hack the node (local vuln + target node forward + connection)
                        num_steps = 3
                        for rule in firewall_outgoing:
                            if rule.port == credential.port and rule.permission == RulePermission.BLOCK:
                                # if there is also the outgoing block traffic you also need to change source node (minimum + 1)
                                num_steps = 4
                                break
                        access_graph.add_edge(node, credential.node, vulnerability=vulnerability_id, num_steps=num_steps)
                        firewall_incoming = graph.nodes[credential.node]['data'].firewall.incoming
                        # if instead there is the incoming rule, we cannot use that vulnerability -> remove edge
                        for rule in firewall_incoming:
                            if rule.port == credential.port and rule.permission == RulePermission.BLOCK:
                                access_graph.remove_edge(node, credential.node)
                                break

    traffic_graph = copy.deepcopy(graph)
    for node in traffic_graph.nodes:
        knows_graph.nodes[node]['data'] = traffic_graph.nodes[node]['data']
        access_graph.nodes[node]['data'] = traffic_graph.nodes[node]['data']

    graph.clear_edges() # starting graph
    evolving_visible_graph = nx.DiGraph() # GNN graph

    return graph, password_counter, traffic_graph, knows_graph, access_graph, evolving_visible_graph


def new_environment(n_servers_per_protocol: int, env_type: str = "random_env") -> m.Environment:
    """Create a new simulation environment based on
    a randomly generated network topology.

    NOTE: the probabilities and parameter values used
    here for the statistical generative model
    were arbirarily picked. We recommend exploring different values for those parameters.
    """
    traffic = generate_random_traffic_network(seed=None,
                                              n_clients=50,
                                              n_servers={
                                                  "SMB": n_servers_per_protocol,
                                                  "HTTP": n_servers_per_protocol,
                                                  "RDP": n_servers_per_protocol,
                                              },
                                              alpha=np.array([(1, 1), (0.2, 0.5)], dtype=float),
                                              beta=np.array([(1000, 10), (10, 100)], dtype=float))

    network, num_credentials, traffic_graph, knows_graph, access_graph, evolving_visible_graph = cyberbattle_model_from_traffic_graph(
        traffic,
        cached_rdp_password_probability=0.8,
        cached_smb_password_probability=0.7,
        cached_accessed_network_shares_probability=0.8,
        cached_password_has_changed_probability=0.01,
        probability_two_nodes_use_same_password_to_access_given_resource=0.9)
    return m.Environment(network=network,
                         vulnerability_library=dict([]),
                         identifiers=ENV_IDENTIFIERS,
                         num_credentials=num_credentials,
                         traffic_graph=traffic_graph,
                         knows_graph=knows_graph,
                         access_graph=access_graph,
                         evolving_visible_graph=evolving_visible_graph,
                         env_type=env_type
                         )
