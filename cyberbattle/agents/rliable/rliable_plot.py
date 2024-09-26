import argparse
import os
from rliable_metrics import load_auc, plot_score_hist, plot_interval_estimates, plot_performance_profile, plot_probability_improvement, plot_ranks, determine_algorithm_name, determine_task_name, plot_alternative_metrics
import numpy as np
from metrics_utils import normalize_score, RANDOM_SCORE
from datetime import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load and analyze scores from a folder.')
    parser.add_argument('-f', '--folders', nargs='+', type=str,
                        help='List of folder names containing algorithm results.')
    parser.add_argument('-m', '--metric', type=str, default='validation/Average owned percentage among reachable nodes',
                        help='Name of the metric to be used for analysis.')
    parser.add_argument('-o', '--option', type=str, default='interval_estimate', help='Type of analysis to be performed.', choices=['interval_estimate', 'rank', 'histogram', 'performance_profile', 'probability_improvement', 'alternative',
                                                                                                                                    'superhuman_probability', 'difficulty_progress'])

    args = parser.parse_args()

    folders = [os.path.join("../logs/final-comparison/", folder_name) for folder_name in args.folders]

    scores_dict = {}

    logs_folder = os.path.join('./logs', "metrics_" + datetime.now().strftime('%Y-%m-%d_%H'))
    os.makedirs(logs_folder, exist_ok=True)

    all_tasks = sorted(set(determine_task_name(algorithm_folder) for algorithm_folder in folders))
    all_algorithms = sorted(set(determine_algorithm_name(algorithm_folder) for algorithm_folder in folders))

    for algorithm in all_algorithms:
        scores_dict[algorithm] = {}

    for algorithm_folder in folders:
        scores = []
        algorithm_name = determine_algorithm_name(algorithm_folder)
        task_name = determine_task_name(algorithm_folder)
        if algorithm_name != "Unknown":
            for run_folder in os.listdir(algorithm_folder):
                if os.path.isdir(os.path.join(algorithm_folder, run_folder)) and run_folder.startswith(algorithm_name):
                    print(f"Loading scores from {os.path.join(algorithm_folder, run_folder)}")
                    auc, max_auc = load_auc(os.path.join(algorithm_folder, run_folder), args.metric)
                    scores.append(normalize_score(auc, RANDOM_SCORE[task_name], max_auc)) # normalization assumes random_score as minimum and 1 as maximum
        scores_dict[algorithm_name][task_name] = np.array(scores)
    for algorithm in scores_dict:
        to_order = []
        for task in scores_dict[algorithm]:
            to_order.append((task, scores_dict[algorithm][task]))
        ordered_scores = [scores for task, scores in sorted(to_order, key=lambda x: x[0])]
        scores_dict[algorithm] = np.array(ordered_scores)

    if args.option == 'interval_estimate':
        plot_interval_estimates(scores_dict, logs_folder=logs_folder)
    elif args.option == 'histogram':
        for algorithm_name in all_algorithms:
            plot_score_hist(scores_dict[algorithm_name], bins=10, N=min(5,len(all_tasks)), figsize=(26, 11), names=all_tasks, algorithm_name=algorithm_name, logs_folder=logs_folder)
    elif args.option == 'performance_profile':
        plot_performance_profile(scores_dict, logs_folder=logs_folder)
    elif args.option == 'probability_improvement':
        plot_probability_improvement(scores_dict, all_algorithms, logs_folder=logs_folder)
    elif args.option == 'rank':
        plot_ranks(scores_dict, all_algorithms, all_tasks, logs_folder=logs_folder)
    elif args.option == 'difficulty_progress' or args.option == 'superhuman_probability' or args.option == 'alternative':
        plot_alternative_metrics(scores_dict, logs_folder=logs_folder)

