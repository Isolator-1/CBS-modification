# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Data model for the simulation environment.

The simulation environment is given by the directed graph
formally defined by:

  Node := NodeID x ListeningService[] x Value x Vulnerability[] x FirewallConfig
  Edge := NodeID x NodeID x PortName

 where:
  - NodeID: string
  - ListeningService : Name x AllowedCredentials
  - AllowedCredentials : string[] # credential pair represented by just a
    string ID
  - Value : [0...100]     # Intrinsic value of reaching this node
  - Vulnerability : VulnerabilityID x Type x Precondition x Outcome x Rates
  - VulnerabilityID : string
  - Rates : ProbingDetectionRate x ExploitDetectionRate x SuccessRate
  - FirewallConfig: {
      outgoing :  FirwallRule[]
      incoming : FirwallRule [] }
  - FirewallRule: PortName x { ALLOW, BLOCK }
"""

from datetime import datetime
from typing import NamedTuple, List, Dict, Optional, Union, Tuple, Iterator
import dataclasses
from dataclasses import dataclass, field
import matplotlib.pyplot as plt  # type:ignore
from enum import Enum, IntEnum
from boolean import boolean
import networkx as nx
import yaml
import random

import matplotlib  # type: ignore
matplotlib.use('Agg')

VERSION_TAG = "0.1.0"

ALGEBRA = boolean.BooleanAlgebra()

# Type alias for identifiers
NodeID = str

# A unique identifier
ID = str

# a (login,password/token) credential pair is abstracted as just a unique
# string identifier
CredentialID = str

# Intrinsic value of a reaching a given node in [0,100]
NodeValue = int


PortName = str


@dataclass
class ListeningService:
    """A service port on a given node accepting connection initiated
    with the specified allowed credentials """
    # Name of the port the service is listening to
    name: PortName
    # credential allowed to authenticate with the service
    allowedCredentials: List[CredentialID] = dataclasses.field(default_factory=list)
    # whether the service is running or stopped
    running: bool = True
    # Weight used to evaluate the cost of not running the service
    sla_weight = 1.0


x = ListeningService(name='d')
VulnerabilityID = str

# Probability rate
Probability = float

# The name of a node property indicating the presence of a
# service, component, feature or vulnerability on a given node.
PropertyName = str


class Rates(NamedTuple):
    """Probabilities associated with a given vulnerability"""
    probingDetectionRate: Probability = 0.0
    exploitDetectionRate: Probability = 0.0
    successRate: Probability = 1.0


class VulnerabilityType(Enum):
    """Is the vulnerability exploitable locally or remotely?"""
    LOCAL = 0
    REMOTE = 1


class PrivilegeLevel(IntEnum):
    """Access privilege level on a given node"""
    NoAccess = 0
    LocalUser = 1
    Admin = 2
    System = 3
    MAXIMUM = 3


def escalate(current_level, escalation_level: PrivilegeLevel) -> PrivilegeLevel:
    return PrivilegeLevel(max(int(current_level), int(escalation_level)))


class VulnerabilityOutcome:
    """Outcome of exploiting a given vulnerability"""


class LateralMove(VulnerabilityOutcome):
    """Lateral movement to the target node"""
    success: bool


class CustomerData(VulnerabilityOutcome):
    """Access customer data on target node"""


class PrivilegeEscalation(VulnerabilityOutcome):
    """Privilege escalation outcome"""

    def __init__(self, level: PrivilegeLevel):
        self.level = level

    @property
    def tag(self):
        """Escalation tag that gets added to node properties when
        the escalation level is reached for that node"""
        return f"privilege_{self.level}"


class SystemEscalation(PrivilegeEscalation):
    """Escalation to SYSTEM privileges"""

    def __init__(self):
        super().__init__(PrivilegeLevel.System)


class AdminEscalation(PrivilegeEscalation):
    """Escalation to local administrator privileges"""

    def __init__(self):
        super().__init__(PrivilegeLevel.Admin)


class ProbeSucceeded(VulnerabilityOutcome):
    """Probing succeeded"""

    def __init__(self, discovered_properties: List[PropertyName]):
        self.discovered_properties = discovered_properties


class ProbeFailed(VulnerabilityOutcome):
    """Probing failed"""


class ExploitFailed(VulnerabilityOutcome):
    """This is for situations where the exploit fails """


class CachedCredential(NamedTuple):
    """Encodes a machine-port-credential triplet"""
    node: NodeID
    port: PortName
    credential: CredentialID
    valid: bool = True


class LeakedCredentials(VulnerabilityOutcome):
    """A set of credentials obtained by exploiting a vulnerability"""

    credentials: List[CachedCredential]

    def __init__(self, credentials: List[CachedCredential]):
        self.credentials = credentials


class LeakedNodesId(VulnerabilityOutcome):
    """A set of node IDs obtained by exploiting a vulnerability"""

    def __init__(self, nodes: List[NodeID]):
        self.nodes = nodes


class Movement(VulnerabilityOutcome):
    """A movement of the view"""

    def __init__(self, source=1, forward=1):
        self.source = source
        self.forward = forward
        pass

VulnerabilityOutcomes = Union[
    LeakedCredentials, LeakedNodesId, PrivilegeEscalation, AdminEscalation,
    SystemEscalation, CustomerData, LateralMove, ExploitFailed]


def map_outcome_to_index(outcome):
    outcomes = [
        LeakedCredentials, LeakedNodesId, PrivilegeEscalation, AdminEscalation,
        SystemEscalation, CustomerData, LateralMove, ExploitFailed, ProbeSucceeded, ProbeFailed]
    outcome_to_index = {cls: idx for idx, cls in enumerate(outcomes)}
    return outcome_to_index.get(outcome.__class__)


class AttackResult():
    """The result of attempting a specific attack (either local or remote)"""
    success: bool
    expected_outcome: Union[VulnerabilityOutcomes, None]


class Precondition:
    """ A predicate logic expression defining the condition under which a given
    feature or vulnerability is present or not.
    The symbols used in the expression refer to properties associated with
    the corresponding node.
    E.g. 'Win7', 'Server', 'IISInstalled', 'SQLServerInstalled',
    'AntivirusInstalled' ...
    """

    expression: boolean.Expression

    def __init__(self, expression: Union[boolean.Expression, str]):
        if isinstance(expression, boolean.Expression):
            self.expression = expression
        else:
            self.expression = ALGEBRA.parse(expression)


class VulnerabilityInfo(NamedTuple):
    """Definition of a known vulnerability"""
    # an optional description of what the vulnerability is
    description: str
    # type of vulnerability
    type: VulnerabilityType
    # what happens when successfully exploiting the vulnerability
    outcome: VulnerabilityOutcome
    # a boolean expression over a node's properties determining if the
    # vulnerability is present or not
    precondition: Precondition = Precondition("true")
    # rates of success/failure associated with this vulnerability
    rates: Rates = Rates()
    # points to information about the vulnerability
    URL: str = ""
    # some cost associated with exploiting this vulnerability (e.g.
    # brute force more costly than dumping credentials)
    cost: float = 1.0
    # a string displayed when the vulnerability is successfully exploited
    reward_string: str = ""


# A dictionary storing information about all supported vulnerabilities
# or features supported by the simulation.
# This is to be used as a global dictionary pre-populated before
# starting the simulation and estimated from real-world data.
VulnerabilityLibrary = Dict[VulnerabilityID, VulnerabilityInfo]


class RulePermission(Enum):
    """Determine if a rule is blocks or allows traffic"""
    NON_EXISTING = -1
    ALLOW = 0
    BLOCK = 1


@dataclass
class FirewallRule:
    """A firewall rule"""
    # A port name
    port: PortName
    # permission on this port
    permission: RulePermission
    # An optional reason for the block/allow rule
    reason: str = ""

    def __eq__(self, other):
        if not isinstance(other, FirewallRule):
            return NotImplemented
        return (self.port, self.permission, self.reason) == (other.port, other.permission, other.reason)

    def __hash__(self):
        return hash((self.port, self.permission, self.reason))


@dataclass
class FirewallConfiguration:
    """Firewall configuration on a given node.
    Determine if traffic should be allowed or specifically blocked
    on a given port for outgoing and incoming traffic.
    The rules are process in order: the first rule matching a given
    port is applied and the rest are ignored.

    Port that are not listed in the configuration
    are assumed to be blocked. (Adding an explicit block rule
    can still be useful to give a reason for the block.)
    """
    outgoing: List[FirewallRule] = field(repr=True, default_factory=lambda: [
        FirewallRule("RDP", RulePermission.ALLOW),
        FirewallRule("SSH", RulePermission.ALLOW),
        FirewallRule("HTTPS", RulePermission.ALLOW),
        FirewallRule("HTTP", RulePermission.ALLOW)])
    incoming: List[FirewallRule] = field(repr=True, default_factory=lambda: [
        FirewallRule("RDP", RulePermission.ALLOW),
        FirewallRule("SSH", RulePermission.ALLOW),
        FirewallRule("HTTPS", RulePermission.ALLOW),
        FirewallRule("HTTP", RulePermission.ALLOW)])


class MachineStatus(Enum):
    """Machine running status"""
    Stopped = 0
    Running = 1
    Imaging = 2


@dataclass
class NodeDiscoveredInfo:
    """A computer node in the enterprise network"""
    # List of port/protocol the node is listening to
    services: List[ListeningService]
    # Intrinsic value of the node (translates into a reward if the node gets owned)
    value: NodeValue = 0
    # Properties of the nodes, some of which can imply further vulnerabilities
    properties: List[PropertyName] = dataclasses.field(default_factory=list)
    # List of known vulnerabilities for the node
    vulnerabilities: VulnerabilityLibrary = dataclasses.field(default_factory=dict)
    # Fireall configuration of the node
    firewall: FirewallConfiguration = FirewallConfiguration()
    # Escalation level
    privilege_level: PrivilegeLevel = PrivilegeLevel.NoAccess
    # Can the node be re-imaged by a defender agent?
    reimagable: bool = True
    # Machine status: running or stopped
    status = MachineStatus.Running
    # Relative node weight used to calculate the cost of stopping this machine
    # or its services
    sla_weight: float = 1.0


@dataclass
class NodeOwnedInfo(NodeDiscoveredInfo):
    # Same properties but some are not masked
    pass


@dataclass
class NodeInfo(NodeOwnedInfo):
    # Node ID
    node_id: NodeID = ""
    # Attacker agent installed on the node? (aka the node is 'pwned')
    agent_installed: bool = False
    # Last time the node was reimaged
    last_reimaging: Optional[datetime] = None
    # String displayed when the node gets owned
    owned_string: str = ""


class Identifiers(NamedTuple):
    """Define the global set of identifiers used
    in the definition of a given environment.
    Such set defines a common vocabulary possibly
    shared across multiple environments, thus
    ensuring a consistent numbering convention
    that a machine learniong model can learn from."""
    # Array of all possible node property identifiers
    properties: List[PropertyName] = []
    # Array of all possible port names
    ports: List[PortName] = []
    # Array of all possible local vulnerabilities names
    local_vulnerabilities: List[VulnerabilityID] = []
    # Array of all possible remote vulnerabilities names
    remote_vulnerabilities: List[VulnerabilityID] = []


def iterate_network_nodes(network: nx.graph.Graph) -> Iterator[Tuple[NodeID, NodeInfo]]:
    """Iterates over the nodes in the network"""
    for nodeid, nodevalue in network.nodes.items():
        node_data: NodeInfo = nodevalue['data']
        yield nodeid, node_data

def calculate_average_shortest_path_length(digraph):
    """
    Calculate the average shortest path length in a directed graph.

    Parameters:
    - digraph: networkx.DiGraph
        The directed graph.

    Returns:
    - float
        Average shortest path length.
    """
    try:
        all_pairs_shortest_lengths = dict(nx.all_pairs_shortest_path_length(digraph))
        max_possible_value = len(all_pairs_shortest_lengths) - 1  # The maximum number of steps between any two nodes = number of nodes - 1

        total_shortest_path_length = 0
        num_pairs = 0
        not_reachable = 0
        for source, target_lengths in all_pairs_shortest_lengths.items():
            target_lengths = {int(key): value for key, value in target_lengths.items()}
            for node in range(max_possible_value+1):
                if node not in target_lengths:
                    not_reachable += 1
                    # If the node is not reachable from the source, use double the maximum possible value
                    target_lengths[node] = max_possible_value*2
            for target, length in target_lengths.items():
                if target != int(source):
                    total_shortest_path_length += length
                    num_pairs += 1


        reachability_metric = 1 - (not_reachable / (len(all_pairs_shortest_lengths) * len(all_pairs_shortest_lengths)))
        average_shortest_path_length = total_shortest_path_length / num_pairs
        connectivity_metric = 1 - (average_shortest_path_length / (2*max_possible_value))
        # filtering the dict to include None if a node is not reachable, used later
        for source, target_lengths in all_pairs_shortest_lengths.items():
            target_lengths = {int(key): value for key, value in target_lengths.items()}
            all_pairs_shortest_lengths[source] = target_lengths

        for source, target_lengths in all_pairs_shortest_lengths.items():
            for node in range(len(all_pairs_shortest_lengths)):
                if node not in target_lengths.keys():
                    all_pairs_shortest_lengths[source][node] = None
        return reachability_metric, connectivity_metric, all_pairs_shortest_lengths
    except nx.NetworkXError:
        # If the graph is not weakly connected, handle the exception
        return float('inf')

# NOTE: Using `NameTuple` instead of `dataclass` breaks deserialization
# with PyYaml 2.8.1 due to a new recrusive references to the networkx graph in the field
#   edges: !!python/object:networkx.classes.reportviews.EdgeView
#     _adjdict: *id018
#     _graph: *id019
@dataclass
class Environment:

    """ The static graph defining the network of computers """
    vulnerability_library: VulnerabilityLibrary
    identifiers: Identifiers
    network: nx.DiGraph = None
    network_parameters: dict = None
    creationTime: datetime = datetime.utcnow()
    lastModified: datetime = datetime.utcnow()
    # a version tag indicating the environment schema version
    version: str = VERSION_TAG
    cached_rdp_password_probability: float = 0.5
    cached_smb_password_probability: float = 0.5
    cached_accessed_network_shares_probability: float = 0.5
    cached_password_has_changed_probability: float = 0.01
    traceroute_discovery_probability: float = 0.5
    probability_two_nodes_use_same_password_to_access_given_resource: float = 0.5
    firewall_rule_incoming_probability: float = 0.2
    firewall_rule_outgoing_probability: float = 0.2
    num_credentials: int = 1
    env_type: str = "general_env"
    tolerance: float = 1e-3
    traffic_graph = None
    knows_graph = None
    access_graph = None
    evolving_visible_graph = None

    def __init__(self,
                 vulnerability_library,
                 identifiers,
                 network=None,
                 network_parameters=None,
                 creationTime=None,
                 lastModified=None,
                 version="VERSION_TAG",
                 cached_rdp_password_probability=0.5,
                 cached_smb_password_probability=0.5,
                 cached_accessed_network_shares_probability=0.5,
                 cached_password_has_changed_probability=0.01,
                 traceroute_discovery_probability=0.5,
                 probability_two_nodes_use_same_password_to_access_given_resource=0.5,
                 firewall_rule_incoming_probability=0.2,
                 firewall_rule_outgoing_probability=0.2,
                 num_credentials=1,
                 env_type="general_env",
                 tolerance=1e-3,
                 traffic_graph=None,
                 knows_graph=None,
                 access_graph=None,
                 evolving_visible_graph=None,
                 **kwargs):

        # Set the fields from arguments
        self.vulnerability_library = vulnerability_library
        self.identifiers = identifiers
        self.network = network
        self.network_parameters = network_parameters
        self.creationTime = creationTime if creationTime is not None else datetime.utcnow()
        self.lastModified = lastModified if lastModified is not None else datetime.utcnow()
        self.version = version
        self.cached_rdp_password_probability = cached_rdp_password_probability
        self.cached_smb_password_probability = cached_smb_password_probability
        self.cached_accessed_network_shares_probability = cached_accessed_network_shares_probability
        self.cached_password_has_changed_probability = cached_password_has_changed_probability
        self.traceroute_discovery_probability = traceroute_discovery_probability
        self.probability_two_nodes_use_same_password_to_access_given_resource = probability_two_nodes_use_same_password_to_access_given_resource
        self.firewall_rule_incoming_probability = firewall_rule_incoming_probability
        self.firewall_rule_outgoing_probability = firewall_rule_outgoing_probability
        self.evolving_visible_graph = evolving_visible_graph
        self.num_credentials = num_credentials
        self.env_type = env_type
        self.tolerance = tolerance
        self.traffic_graph = traffic_graph
        self.knows_graph = knows_graph
        self.access_graph = access_graph

        # imported here due to a circular import issue
        import cyberbattle.simulation.generate_network as g

        # case without network passed in input, but only network parameters
        if self.network_parameters is not None and self.network is None:
            seed = self.network_parameters['seed']
            n_clients = self.network_parameters['n_clients']
            n_servers = self.network_parameters['n_servers']
            alpha = self.network_parameters['alpha']
            beta = self.network_parameters['beta']
            # generating new network
            traffic = g.generate_random_traffic_network_per_protocol(
                seed=seed, n_clients=n_clients,
                n_servers={
                    "SMB": n_servers["SMB"],
                    "HTTP": n_servers["HTTP"],
                    "RDP": n_servers["RDP"],
                },
                alpha=alpha,
                beta=beta,
                tolerance=self.tolerance
            )
            self.network, self.num_credentials, self.traffic_graph, self.knows_graph, self.access_graph, self.evolving_visible_graph = g.cyberbattle_model_from_traffic_graph(
                traffic,
                cached_rdp_password_probability=self.cached_rdp_password_probability,
                cached_smb_password_probability=self.cached_smb_password_probability,
                cached_accessed_network_shares_probability=self.cached_accessed_network_shares_probability,
                cached_password_has_changed_probability=self.cached_password_has_changed_probability,
                traceroute_discovery_probability=self.traceroute_discovery_probability,
                probability_two_nodes_use_same_password_to_access_given_resource=self.probability_two_nodes_use_same_password_to_access_given_resource,
                firewall_rule_incoming_probability=self.firewall_rule_incoming_probability,
                firewall_rule_outgoing_probability=self.firewall_rule_outgoing_probability)
        # generate metrics for the environment
        if self.env_type == "random_env":
            self.knows_reachability, self.knows_connectivity, self.knows_shortest_paths = calculate_average_shortest_path_length(self.knows_graph)
            self.access_reachability, self.access_connectivity, self.access_shortest_paths = calculate_average_shortest_path_length(self.access_graph)
            self.traffic_reachability, self.traffic_connectivity, self.traffic_shortest_paths = calculate_average_shortest_path_length(self.traffic_graph)


    def nodes(self) -> Iterator[Tuple[NodeID, NodeInfo]]:
        """Iterates over the nodes in the network"""
        return iterate_network_nodes(self.network)

    def get_node(self, node_id: NodeID) -> NodeInfo:
        """Retrieve info for the node with the specified ID"""
        node_info: NodeInfo = self.network.nodes[node_id]['data']
        return node_info

    def plot_environment_graph(self) -> None:
        """Plot the full environment graph"""
        nx.draw(self.network,
                with_labels=True,
                node_color=[n['data'].value
                            for i, n in
                            self.network.nodes.items()],
                cmap=plt.cm.Oranges)  # type:ignore


def create_network(nodes: Dict[NodeID, NodeInfo]) -> nx.DiGraph:
    """Create a network with a set of nodes and no edges"""
    graph = nx.DiGraph()
    graph.add_nodes_from([(k, {'data': v}) for (k, v) in list(nodes.items())])
    return graph

# Helpers to infer constants from an environment


def collect_ports_from_vuln(vuln: VulnerabilityInfo) -> List[PortName]:
    """Returns all the port named referenced in a given vulnerability"""
    if isinstance(vuln.outcome, LeakedCredentials):
        return [c.port for c in vuln.outcome.credentials]
    else:
        return []


def collect_vulnerability_ids_from_nodes_bytype(
        nodes: Iterator[Tuple[NodeID, NodeInfo]],
        global_vulnerabilities: VulnerabilityLibrary,
        type: VulnerabilityType) -> List[VulnerabilityID]:
    """Collect and return all IDs of all vulnerability of the specified type
    that are referenced in a given set of nodes and vulnerability library
    """
    return sorted(list({
        id
        for _, node_info in nodes
        for id, v in node_info.vulnerabilities.items()
        if v.type == type
    }.union(
        id
        for id, v in global_vulnerabilities.items()
        if v.type == type
    )))


def collect_properties_from_nodes(nodes: Iterator[Tuple[NodeID, NodeInfo]]) -> List[PropertyName]:
    """Collect and return sorted list of all property names used in a given set of nodes"""
    return sorted({
        p
        for _, node_info in nodes
        for p in node_info.properties
    })


def collect_ports_from_nodes(
        nodes: Iterator[Tuple[NodeID, NodeInfo]],
        vulnerability_library: VulnerabilityLibrary) -> List[PortName]:
    """Collect and return all port names used in a given set of nodes
    and global vulnerability library"""
    return sorted(list({
        port
        for _, v in vulnerability_library.items()
        for port in collect_ports_from_vuln(v)
    }.union({
        port
        for _, node_info in nodes
        for _, v in node_info.vulnerabilities.items()
        for port in collect_ports_from_vuln(v)
    }.union(
        {service.name
         for _, node_info in nodes
         for service in node_info.services}))))


def collect_ports_from_environment(environment: Environment) -> List[PortName]:
    """Collect and return all port names used in a given environment"""
    return collect_ports_from_nodes(environment.nodes(), environment.vulnerability_library)


def infer_constants_from_nodes(
        nodes: Iterator[Tuple[NodeID, NodeInfo]],
        vulnerabilities: Dict[VulnerabilityID, VulnerabilityInfo]) -> Identifiers:
    """Infer global environment constants from a given network"""
    return Identifiers(
        properties=collect_properties_from_nodes(nodes),
        ports=collect_ports_from_nodes(nodes, vulnerabilities),
        local_vulnerabilities=collect_vulnerability_ids_from_nodes_bytype(
            nodes, vulnerabilities, VulnerabilityType.LOCAL),
        remote_vulnerabilities=collect_vulnerability_ids_from_nodes_bytype(
            nodes, vulnerabilities, VulnerabilityType.REMOTE)
    )


def infer_constants_from_network(
        network: nx.Graph,
        vulnerabilities: Dict[VulnerabilityID, VulnerabilityInfo]) -> Identifiers:
    """Infer global environment constants from a given network"""
    return infer_constants_from_nodes(iterate_network_nodes(network), vulnerabilities)


# Network creation

# A sample set of envrionment constants
SAMPLE_IDENTIFIERS = Identifiers(
    ports=['RDP', 'SSH', 'SMB', 'HTTP', 'HTTPS', 'WMI', 'SQL'],
    properties=[
        'Windows', 'Linux', 'HyperV-VM', 'Azure-VM', 'Win7', 'Win10',
        'PortRDPOpen', 'GuestAccountEnabled']
)


def assign_random_labels(
        graph: nx.DiGraph,
        vulnerabilities: VulnerabilityLibrary = dict([]),
        identifiers: Identifiers = SAMPLE_IDENTIFIERS) -> nx.DiGraph:
    """Create an envrionment network by randomly assigning node information
    (properties, firewall configuration, vulnerabilities)
    to the nodes of a given graph structure"""

    # convert node IDs to string
    graph = nx.relabel_nodes(graph, {i: str(i) for i in graph.nodes})

    def create_random_firewall_configuration() -> FirewallConfiguration:
        return FirewallConfiguration(
            outgoing=[
                FirewallRule(port=p, permission=RulePermission.ALLOW)
                for p in
                random.sample(
                    identifiers.ports,
                    k=random.randint(0, len(identifiers.ports)))],
            incoming=[
                FirewallRule(port=p, permission=RulePermission.ALLOW)
                for p in random.sample(
                    identifiers.ports,
                    k=random.randint(0, len(identifiers.ports)))])

    def create_random_properties() -> List[PropertyName]:
        return list(random.sample(
            identifiers.properties,
            k=random.randint(0, len(identifiers.properties))))

    def pick_random_global_vulnerabilities() -> VulnerabilityLibrary:
        count = random.random()
        return {k: v for (k, v) in vulnerabilities.items() if random.random() > count}

    def add_leak_neighbors_vulnerability(library: VulnerabilityLibrary, node_id: NodeID) -> None:
        """Create a vulnerability for each node that reveals its immediate neighbors"""
        neighbors = {t for (s, t) in graph.edges() if s == node_id}
        if len(neighbors) > 0:
            library['RecentlyAccessedMachines'] = VulnerabilityInfo(
                description="AzureVM info, including public IP address",
                type=VulnerabilityType.LOCAL,
                outcome=LeakedNodesId(list(neighbors)))

    def create_random_vulnerabilities(node_id: NodeID) -> VulnerabilityLibrary:
        library = pick_random_global_vulnerabilities()
        add_leak_neighbors_vulnerability(library, node_id)
        return library

    # Pick a random node as the agent entry node
    entry_node_index = random.randrange(len(graph.nodes))
    entry_node_id, entry_node_data = list(graph.nodes(data=True))[entry_node_index]
    graph.nodes[entry_node_id].clear()
    node_data = NodeInfo(services=[],
                         value=0,
                         properties=create_random_properties(),
                         vulnerabilities=create_random_vulnerabilities(entry_node_id),
                         firewall=create_random_firewall_configuration(),
                         agent_installed=True,
                         reimagable=False,
                         privilege_level=PrivilegeLevel.Admin)
    graph.nodes[entry_node_id].update({'data': node_data})

    def create_random_node_data(node_id: NodeID) -> NodeInfo:
        return NodeInfo(
            services=[],
            value=random.randint(0, 100),
            properties=create_random_properties(),
            vulnerabilities=create_random_vulnerabilities(node_id),
            firewall=create_random_firewall_configuration(),
            agent_installed=False,
            privilege_level=PrivilegeLevel.NoAccess)

    for node in list(graph.nodes):
        if node != entry_node_id:
            graph.nodes[node].clear()
            graph.nodes[node].update({'data': create_random_node_data(node)})

    return graph


# Serialization

def setup_yaml_serializer() -> None:
    """Setup a clean YAML formatter for object of type Environment.
    """
    yaml.add_representer(Precondition,
                         lambda dumper, data: dumper.represent_scalar('!BooleanExpression',
                                                                      str(data.expression)))  # type: ignore
    yaml.SafeLoader.add_constructor('!BooleanExpression',
                                    lambda loader, expression: Precondition(
                                        loader.construct_scalar(expression)))  # type: ignore
    yaml.add_constructor('!BooleanExpression',
                         lambda loader, expression:
                         Precondition(loader.construct_scalar(expression)))  # type: ignore

    yaml.add_representer(VulnerabilityType,
                         lambda dumper, data: dumper.represent_scalar('!VulnerabilityType',
                                                                      str(data.name)))  # type: ignore

    yaml.SafeLoader.add_constructor('!VulnerabilityType',
                                    lambda loader, expression: VulnerabilityType[
                                        loader.construct_scalar(expression)])  # type: ignore
    yaml.add_constructor('!VulnerabilityType',
                         lambda loader, expression: VulnerabilityType[
                             loader.construct_scalar(expression)])  # type: ignore
