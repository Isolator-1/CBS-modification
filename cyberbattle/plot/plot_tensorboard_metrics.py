import copy
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for plotting without a display
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Use offscreen platform to avoid Qt issues
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from scipy.stats import sem, t
import seaborn as sns

mapping_dict = {
    'actions/train/Rate of local attacks': 'rate_local_attacks',
    'actions/train/Rate of port connections': 'rate_port_connections',
    'actions/train/Rate of remote attacks': 'rate_remote_attacks',
    'actions/train/Rate of movements': 'rate_movements',
    'actions/train/Success rate for actions': 'success_rate_actions',
    'actions/train/Success rate for local attackswhere we have data': 'success_rate_local_attacks',
    'actions/train/Success rate for port connections': 'success_rate_port_connections',
    'actions/train/Success rate for remote attacks': 'success_rate_remote_attacks',
    'rollout/ep_len_mean': 'ep_length_mean',
    'rollout/ep_rew_mean': 'ep_reward_mean',
    'rollout/exploration_rate': 'exploration_rate',
    'time/fps': 'fps',
    'train/Discovered credentials percentage': 'discovered_credentials_percentage',
    'train/Discovered nodes percentage': 'discovered_nodes_percentage',
    'train/Network availability': 'network_availability',
    'train/Owned nodes percentage': 'owned_nodes_percentage',
    'train/Owned-discovered ratio': 'owned_discovered_ratio',
    'train/entropy_loss': 'entropy_loss',
    'train/explained_variance': 'explained_variance',
    'train/learning_rate': 'learning_rate',
    'train/policy_loss': 'policy_loss',
    'train/value_loss': 'value_loss',
    'actions/validation/Rate of local attacks': 'val_rate_local_attacks',
    'actions/validation/Rate of port connections': 'val_rate_port_connections',
    'actions/validation/Rate of remote attacks': 'val_rate_remote_attacks',
    'actions/validation/Rate of movements': 'val_rate_movements',
    'actions/validation/Success rate for local attacks': 'val_success_rate_local_attacks',
    'actions/validation/Success rate for port connections': 'val_success_rate_port_connections',
    'actions/validation/Success rate for remote attacks': 'val_success_rate_remote_attacks',
    'actions/validation/Success rate for actions': 'val_success_rate_actions',
    'validation/Average discovered credentials percentage': 'val_avg_discovered_credentials_percentage',
    'validation/Average discovered percentage': 'val_avg_discovered_percentage',
    'validation/Average owned percentage': 'val_avg_owned_percentage',
    "validation/Average owned percentage among reachable nodes": 'val_avg_owned_percentage_reachable',
    'validation/Average reward': 'val_avg_reward',
    'validation/Network availability': 'val_network_availability',
    'validation/Owned-discovered ratio': 'val_owned_discovered_ratio'
}

legend_label_mapping = {
    'rate_local_attacks': 'Average rate of local attacks (train)',
    'rate_port_connections': 'Average rate of port connections (train)',
    'rate_remote_attacks': 'Average rate of remote attacks (train)',
    'rate_movements': 'Average rate of switches (train)',
    'success_rate_actions': 'Success rate for actions (train)',
    'success_rate_local_attacks': 'Success rate for local attacks (train)',
    'success_rate_port_connections': 'Success rate for port connections (train)',
    'success_rate_remote_attacks': 'Success rate for remote attacks (train)',
    'ep_length_mean': 'Episode length mean',
    'ep_reward_mean': 'Average episode reward mean (train)',
    'exploration_rate': 'Exploration rate',
    'fps': 'Frames per second',
    'discovered_credentials_percentage': 'Discovered credentials percentage (train)',
    'discovered_nodes_percentage': 'Discovered nodes percentage (train)',
    'network_availability': 'Network availability (train)',
    'owned_nodes_percentage': 'Owned nodes percentage (train)',
    'owned_discovered_ratio': 'Owned-discovered ratio (train)',
    'entropy_loss': 'Entropy loss',
    'explained_variance': 'Explained variance',
    'learning_rate': 'Learning rate',
    'policy_loss': 'Policy loss',
    'value_loss': 'Value loss',
    'val_rate_local_attacks': 'Average rate of local attacks',
    'val_rate_port_connections': 'Average rate of port connections',
    'val_rate_remote_attacks': 'Average rate of remote attacks',
    'val_rate_movements': "Average rate of nodes' switches",
    'val_success_rate_local_attacks': 'Success rate for local attacks (validation)',
    'val_success_rate_port_connections': 'Success rate for port connections (validation)',
    'val_success_rate_remote_attacks': 'Success rate for remote attacks (validation)',
    'val_success_rate_actions': 'Success rate for actions',
    'val_avg_discovered_credentials_percentage': 'Average discovered credentials percentage',
    'val_avg_discovered_percentage': 'Average discovered nodes percentage',
    'val_avg_owned_percentage': 'Average owned nodes percentage',
    'val_avg_owned_percentage_reachable': 'Average owned nodes percentage among reachable nodes',
    'val_avg_reward': 'Average episode reward mean (validation)',
    'val_network_availability': 'Network availability (validation)',
    'val_owned_discovered_ratio': 'Owned-discovered ratio (validation)'
}


def load_tensorboard_logs(logs_folder, metric):
    runs = [run for run in os.listdir(logs_folder) if
            os.path.isdir(os.path.join(logs_folder, run)) and ("A2C" in run or "DQN" in run or "PPO" in run)]
    data = []
    for run in runs:
        print("Run", run)
        event_file = os.path.join(logs_folder, run,
                                  os.listdir(os.path.join(logs_folder, run))[0])  # Assuming only one event file per run
        scalar_data = load_scalar_data(event_file, metric)
        data.append(scalar_data)

    return data

def load_scalar_data(event_file, metric):
    data = {'steps': [], 'values': []}
    for event in tf.compat.v1.train.summary_iterator(event_file):
        for value in event.summary.value:
            if value.tag == metric:
                data['steps'].append(event.step)
                data['values'].append(value.simple_value)
    return data

def plot_mean_ci(data, metric, color, linestyle='-', legend_label=None):
    # Dictionary to accumulate values at each step
    step_values = {}

    for run in data:
        for step, value in zip(run['steps'], run['values']):
            if step not in step_values:
                step_values[step] = []
            step_values[step].append(value)

    # Now, step_values contains all values for each step across runs
    # Calculate mean and confidence interval where we have data
    steps = sorted(step_values.keys())
    mean_values = []
    ci_lower = []
    ci_upper = []

    for step in steps:
        values = step_values[step]
        mean_val = np.mean(values)
        std_err = sem(values)
        n = len(values)
        confidence_level = 0.95
        degrees_freedom = n - 1
        critical_value = t.ppf((1 + confidence_level) / 2, degrees_freedom)
        margin_of_error = critical_value * std_err

        mean_values.append(mean_val)
        ci_lower.append(mean_val - margin_of_error)
        ci_upper.append(mean_val + margin_of_error)

    # Plotting
    plt.gcf().set_size_inches(11, 10)
    plt.fill_between(steps, ci_lower, ci_upper, color=color, alpha=0.2)
    plt.plot(steps, mean_values, color=color, linestyle=linestyle, label=legend_label)
    #plt.legend(loc='upper center', ncol=1, fontsize=20)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.52), ncol=1, fontsize=22)

    plt.subplots_adjust(top=0.7, wspace=0.1, hspace=0.05)


    #plt.legend(loc='upper left', ncol=1, fontsize=16)


def save_and_show_plot(y_axis, metrics):
    """
    Save and show the plot for the given metrics.
    """
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title("Mean and CI of action rates' metrics")
    modified_metrics = "_".join(metrics).replace('/', '_')
    filename = f'output/{modified_metrics}_plot.pdf'
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    plt.savefig(filename)
    plt.close()


def get_key_by_value(dictionary, search_value):
    """
    Get the key from a dictionary by its value.
    """
    for key, value in dictionary.items():
        if value == search_value:
            return key
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and plot mean and std of metrics from TensorBoard logs.')
    parser.add_argument('--logs_folder', type=str, help='Path to the folder containing TensorBoard logs')
    parser.add_argument('--metrics', nargs="+", type=str, help='Metrics to plot (e.g., episode_reward)')
    parser.add_argument('--y_axis', type=str, help='Y-axis label for the plot')
    args = parser.parse_args()

    args.original_metrics = copy.deepcopy(args.metrics)

    # Validate and map metrics
    for index, metric in enumerate(args.metrics):
        if metric not in mapping_dict.values():
            raise ValueError(f"Metric not found: {metric}")
        args.metrics[index] = get_key_by_value(mapping_dict, metric)

    colors = sns.color_palette(n_colors=7)
    linestyle = '-'  # Default line style for training metrics

    print("Reading metrics:", args.metrics)
    logs_folder = os.path.join('..', 'agents', 'logs', 'final-comparison', args.logs_folder)
    print("Logs folder:", logs_folder)

    plt.rcParams.update({'font.size': 22})

    for i, metric in enumerate(args.metrics):
        data = load_tensorboard_logs(logs_folder, metric)
        color = colors[i % len(colors)]  # Cycle through colors

        if "val" in metric:  # Use dotted line for validation metrics
            linestyle = '-'
        else:
            linestyle = '-'
        legend_label = legend_label_mapping[args.original_metrics[i]]
        plot_mean_ci(data, metric, color, linestyle, legend_label)

    save_and_show_plot(args.y_axis, args.metrics)
