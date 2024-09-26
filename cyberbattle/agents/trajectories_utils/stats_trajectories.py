import argparse
import pandas as pd
import pickle
import networkx as nx
from pyvis.network import Network
import sys
import matplotlib
matplotlib.use('TkAgg')
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

# Adapted only for CyberBattleRandomEnv

def calculate_path_length(vulnerabilities_df, vulnerabilities_movements_df, initial_episode_df):
    unique_source_nodes = vulnerabilities_df['Source node'].unique()
    num_unique_source_nodes = len(unique_source_nodes)
    margin = 0.25
    num_valid_edges = vulnerabilities_df[['Source node', 'Target node', 'Action']].drop_duplicates().shape[0]
    vulnerability_ratio = num_unique_source_nodes*(2-margin) / num_valid_edges
    num_valid_edges_with_movements = vulnerabilities_movements_df[['Source node', 'Target node', 'Action']].drop_duplicates().shape[0]
    vulnerability_movements_ratio = num_unique_source_nodes*(3-margin) / num_valid_edges_with_movements
    num_overall_edges = initial_episode_df[['Source node', 'Target node', 'Action']].drop_duplicates().shape[0]
    overall_ratio = num_unique_source_nodes*(3-margin) / num_overall_edges
    #print(num_unique_source_nodes, num_valid_edges, num_valid_edges_with_movements, num_overall_edges)
    print("Redundancy factor: ", num_unique_source_nodes*(3-margin) / num_overall_edges)
    return vulnerability_ratio, vulnerability_movements_ratio, overall_ratio


def calculate_revisiting_ratio(df):
    # Extract sequences of source and target nodes
    source_sequence = df['Source node'].tolist()
    target_sequence = df['Target node'].tolist()

    def revisits(sequence):
        seen = []
        last_node = None
        revisited = set()
        for index, node in enumerate(sequence):
            # Check if this node was seen before and at least one different node has appeared since then
            if node in seen and last_node != None and last_node != node:
                revisited.add(node)
            if not node in seen:
                seen.append(node)
            last_node = node
        return len(revisited), len(set(sequence))

    revisited_sources, total_unique_sources = revisits(source_sequence)
    revisited_targets, total_unique_targets = revisits(target_sequence)

    source_revisiting_ratio = revisited_sources / total_unique_sources if total_unique_sources else 0
    target_revisiting_ratio = revisited_targets / total_unique_targets if total_unique_targets else 0

    print(f"Source revisiting ratio: {source_revisiting_ratio}")
    print(f"Target revisiting ratio: {target_revisiting_ratio}")

    return source_revisiting_ratio, target_revisiting_ratio

def calculate_attack_path(df):
    print("Calculating attack path...")
    # Filter actions with reward > 0 = vulnerabilities or connections
    vulnerabilities_df = df[(df['Reward'] > 0)]
    vulnerabilities_movements_df = df[(df['Reward'] > 0) | df['Action'].str.contains("selection")]
    return vulnerabilities_df, vulnerabilities_movements_df


def calculate_discovered_attack_paths(df):
    test_path = os.path.join(original_folder, str(df.iloc[0]["Run"]), "test_envs.pkl")

    with open(test_path, 'rb') as file:
        test_envs = pickle.load(file)

    #print("Calculating shortest path...")
    initial_source_node = initial_episode_df.iloc[0]['Source node']
    environment_id = initial_episode_df.iloc[0]['Environment']
    test_env = test_envs[environment_id]
    environment = test_env.get_graph()
    #print("Calculating discovered attack paths...")

    reachable_nodes = nx.descendants(environment.access_graph, str(initial_source_node))
    reachable_nodes.add(initial_source_node)  # Include the source node itself

    reachable_access_graph = environment.access_graph.subgraph(reachable_nodes).copy()
    num_modified = 0
    target_nodes_accessible = []
    source_nodes = []
    for _, row in df.iterrows():
        source_node = row['Source node']
        target_node = row['Target node']
        action = row['Action']
        if source_node != initial_source_node and not source_node in source_nodes:
            source_nodes.append(source_node)
        # Traceroute does not spot credentials
        if action.startswith('Scan'):
            #print(f"Checking edge {source_node} with action {action}")
            for u, v, data in reachable_access_graph.edges(data=True):
                if u == str(source_node):
                    if data.get('vulnerability') == action:
                        #print("Found exact edge")
                        # for every target node
                        reachable_access_graph[str(source_node)][v]["discovered"] = True
                        if v not in target_nodes_accessible:
                            target_nodes_accessible.append(v)
                        num_modified += 1

    total_edges = reachable_access_graph.number_of_edges()
    discovered_edges_count = sum(1 for _, _, data in reachable_access_graph.edges(data=True) if data.get('discovered') == True)
    if num_modified == 0:
        print("Isolated run")
        return None, None
    #print(f"Total edges: {total_edges}")
    #print(f"Modified edges: {num_modified}")
    #print(f"Discovered edges: {discovered_edges_count}")
    print(f"Percentage of discovered edges: {discovered_edges_count / total_edges}")
    #print(f"Percentage of nodes accessible: {len(target_nodes_accessible)/reachable_access_graph.number_of_nodes()}")

    #print(f"Accessed nodes among the accessible: {len(source_nodes)/len(target_nodes_accessible)}")
    return discovered_edges_count / total_edges, len(source_nodes)/len(target_nodes_accessible)


def plot_and_save(data, title, xlabel, ylabel, filename, args, figsize=(10, 6), bins=30, kde=True, stat="density"):
    df = pd.DataFrame(data, columns=['Value'])

    # Calculate mean and 95% CI using bootstrapping
    res = bootstrap((df['Value'].values,), np.mean, confidence_level=0.95, n_resamples=10000, method='percentile')
    mean_value = np.mean(df['Value'])
    ci_lower, ci_upper = res.confidence_interval

    plt.figure(figsize=figsize)
    sns.histplot(df, x='Value', bins=bins, kde=kde, stat=stat)
    plt.axvline(mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.3f}')
    plt.axvspan(ci_lower, ci_upper, color='r', alpha=0.3, label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    file_base_name = os.path.basename(args.file).split(".")[0]
    save_path = os.path.join("..", "logs", os.path.dirname(args.file), f"{file_base_name}_{filename}.png")
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a CSV file to calculate vulnerabilities found.")
    parser.add_argument("-f", "--file", help="Path to the input CSV file")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot attack tree")
    args = parser.parse_args()

    args.file = os.path.join("..", "logs", args.file)
    original_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(args.file)))) # 4 levels up
    run_folder = os.path.dirname(args.file).split("/")[-1]

    initial_df = pd.read_csv(args.file)
    initial_df['Reward'] = initial_df['Reward'].apply(lambda x: float(x.strip('[]')))
    initial_df['Episode_RunID'] = initial_df['Episode'].astype(str) + "_" + initial_df['Run'].astype(str)
    vulnerabilities_df, vulnerabilities_movements_df = calculate_attack_path(initial_df)

    vulnerabilities_df['Episode_RunID'] = vulnerabilities_df['Episode'].astype(str) + "_" + vulnerabilities_df['Run'].astype(str)
    unique_episode_run_ids = vulnerabilities_df['Episode_RunID'].unique()
    vulnerabilities_movements_df['Episode_RunID'] = vulnerabilities_movements_df['Episode'].astype(str) + "_" + vulnerabilities_movements_df['Run'].astype(str)

    vuln_list = []
    vuln_mov_list = []
    all_list = []
    der_list = []
    nar_list = []
    source_revisiting_ratio_list = []
    target_revisiting_ratio_list = []


    for episode_run_id in unique_episode_run_ids:
        vulnerabilities_episode_df = vulnerabilities_df[vulnerabilities_df['Episode_RunID'] == episode_run_id]
        vulnerabilities_movements_episode_df = vulnerabilities_movements_df[vulnerabilities_movements_df['Episode_RunID'] == episode_run_id]
        initial_episode_df = initial_df[initial_df['Episode_RunID'] == episode_run_id]
        print(f"--- Episode & Run ID {episode_run_id} ---")
        vuln, vuln_mov, all = calculate_path_length(vulnerabilities_episode_df, vulnerabilities_movements_episode_df, initial_episode_df)
        source_revisiting_ratio, target_revisiting_ratio = calculate_revisiting_ratio(vulnerabilities_movements_episode_df)
        der, nar = calculate_discovered_attack_paths(vulnerabilities_episode_df)
        source_revisiting_ratio_list.append(source_revisiting_ratio)
        target_revisiting_ratio_list.append(target_revisiting_ratio)
        vuln_list.append(vuln)
        vuln_mov_list.append(vuln_mov)
        all_list.append(all)
        if nar is not None:
            nar_list.append(nar)
        if der is not None:
            der_list.append(der)
        if args.plot:
            plot_attack_tree(vulnerabilities_episode_df)

    data_lists = [
       # (vuln_list, 'Vulnerability Ratio Probability Distribution', 'vd'),
        #(nar_list, 'Vulnerability Movements Ratio Probability Distribution', 'nar'),
        (der_list, 'Discovered Attack Paths Ratio Probability Distribution', 'der'),
        #(vuln_mov_list, 'Vulnerability Movements Ratio Probability Distribution', 'vmd'),
        (all_list, 'Redundancy factor distribution', 'redundancy_factor'),
        (source_revisiting_ratio_list, 'Source Revisiting Ratio Probability Distribution', 'source_revisiting_ratio'),
        (target_revisiting_ratio_list, 'Target Revisiting Ratio Probability Distribution', 'target_revisiting_ratio'),
    ]

    for data_list, title, filename in data_lists:
        plot_and_save(data_list, title, 'Value', 'Density', filename, args)
