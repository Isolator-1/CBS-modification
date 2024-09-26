# coding=utf-8
# Copyright 2021 The Rliable Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
from plot_utils import decorate_axis, plot_interval_estimates_shifted
from metrics_utils import IQM, OG, MEAN, MEDIAN, get_rank_matrix
import numpy as np
import os
import warnings
import logging
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib import rcParams
from matplotlib import rc
from itertools import combinations

warnings.filterwarnings('default')
RAND_STATE = np.random.RandomState(42)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
sns.set_style("white")
# Matplotlib params
rcParams['legend.loc'] = 'best'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rc('text', usetex=False)

def plot_score_hist(score_matrix, bins=20, figsize=(28, 14),
                    fontsize='xx-large', N=6, extra_row=1,
                    names=None, algorithm_name="unknown", logs_folder="./logs"):
  num_tasks = score_matrix.shape[0]
  N1 = (num_tasks // N) + extra_row
  fig, ax = plt.subplots(nrows=N1, ncols=N, figsize=figsize)
  for i in range(N):
    for j in range(N1):
      idx = j * N + i
      if idx < num_tasks:
        ax[j, i].set_title(names[idx], fontsize=fontsize)
        print(score_matrix[idx, :])
        sns.histplot(score_matrix[idx, :], bins=bins, ax=ax[j,i], kde=True)
      else:
        ax[j, i].axis('off')
      decorate_axis(ax[j, i], wrect=5, hrect=5, labelsize='xx-large')
      ax[j, i].xaxis.set_major_locator(plt.MaxNLocator(4))
      if idx % N == 0:
        ax[j, i].set_ylabel('Count', size=fontsize)
      else:
        ax[j, i].yaxis.label.set_visible(False)
      ax[j, i].grid(axis='y', alpha=0.1)
  fig.subplots_adjust(hspace=0.85, wspace=0.17)
  plt.savefig(os.path.join(logs_folder,f"histogram_{algorithm_name}.pdf"))
  plt.show()

def load_auc(folder_path, metric_name):
    file_path = os.path.join(folder_path,os.listdir(folder_path)[0])
    print(file_path)
    event_acc = EventAccumulator(file_path)
    event_acc.Reload()

    scalar_events = event_acc.Scalars(metric_name)

    steps = [event.step for event in scalar_events]
    values = [event.value for event in scalar_events]
    # calculate area ander the curve
    auc = np.trapz(values, steps)
    print(f'AUC: {auc}')
    max_auc = 1 * steps[-1]
    return auc, max_auc


def determine_algorithm_name(folder_name):
    # Determine the algorithm name based on the folder structure
    runs = [run_folder for run_folder in os.listdir(folder_name) if "_" in run_folder]
    for run in runs:
        if "RecurrentPPO" in run:
            return "RecurrentPPO"
        elif "TRPO" in run:
            return "TRPO"
        elif "QRDQN" in run:
            return "QRDQN"
        elif "DQN" in run:
            return "DQN"
        elif "PPO" in run:
            return "PPO"
        elif "A2C" in run:
            return "A2C"
        else:
            continue
    return "Unknown"

def determine_task_name(folder_name):
    # Determine the task name based on the folder structure
    return folder_name.split("_")[0].split("/")[-1]

def plot_interval_estimates(scores_dict, reps=10000, logs_folder="./logs"):
    plt.rcParams.update({'font.size': 9})
    DP_25 = lambda scores: difficulty_progress(scores, 0.25)
    aggregate_func = lambda x: np.array([MEDIAN(x), IQM(x), MEAN(x), DP_25(x)])
    aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
        scores_dict, aggregate_func, reps=reps)
    print(aggregate_scores)
    print(aggregate_interval_estimates)
    algorithms = list(scores_dict.keys())
    colors = sns.color_palette('colorblind')
    xlabels = algorithms
    color_idxs = [i for i in range(len(algorithms))]
    color_dict = dict(zip(xlabels, [colors[idx] for idx in color_idxs]))
    plot_interval_estimates_shifted(
        aggregate_scores,
        aggregate_interval_estimates,
        metric_names=['Median', 'IQM', 'Mean', 'Difficulty Progress (25%)'],
        algorithms=algorithms,
        colors=color_dict,
        xlabel_y_coordinate=-1.6,
        xlabel='Human Normalized Score',
        names=algorithms,
    )
    plt.gcf().set_size_inches(12, 10)
    plt.tight_layout()
    plt.savefig(os.path.join(logs_folder,'interval_estimates.pdf'))
    plt.show()


def plot_performance_profile(scores_dict, reps=10000, logs_folder="./logs"):
    plt.rcParams.update({'font.size': 16})
    algorithms = list(scores_dict.keys())
    score_dict = {key: scores_dict[key] for key in algorithms}
    taus = np.linspace(0.0, 0.75, 101)

    score_distributions, score_distributions_cis = rly.create_performance_profile(
        score_dict, taus, reps=reps)

    #color_dict = dict(zip(algorithms, sns.color_palette('colorblind')))
    color_dict = {'A2C': (0.03784313725490196, 0.6496078431372549, 0.40098039215686275),
                  'DQN': (0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
                  'PPO': (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
                  'QRDQN': (0.00392156862745098, 0.25098039215686275, 0.4980392156862745),
                  'RecurrentPPO': (0.8352941176470589, 0.3686274509803922, 0.0),
                  'TRPO': (0.792156862745098, 0.5686274509803921, 0.3803921568627451)}

    fig, axes = plt.subplots(ncols=1, figsize=(12, 10))
    xticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    plot_utils.plot_performance_profiles(
        score_distributions, taus,
        performance_profile_cis=score_distributions_cis,
        colors=color_dict,
        xlabel=r'Normalized Score $(\tau)$', # Human
        labelsize='x-large',
        xticks=xticks,
        ax=axes)

    desired_order = ['PPO', 'TRPO', 'RecurrentPPO', 'QRDQN', 'DQN', 'A2C']

    fake_patches = [mpatches.Patch(color=color_dict[alg],
                                   alpha=0.75) for alg in desired_order]
    fig.legend(fake_patches, desired_order, loc='upper center',
                        fancybox=True, ncol=len(algorithms)/2,
                        fontsize='large')
    fig.subplots_adjust(top=0.9, wspace=0.1, hspace=0.05)
    #plt.tight_layout(pad=2)
    plt.savefig(os.path.join(logs_folder,"performance_profile.pdf"))
    plt.show()


def plot_probability_improvement(scores_dict, algorithms, reps=10000, logs_folder="./logs"):
    # probability of improvement, one algorithm with respect to another
    plt.rcParams.update({'font.size': 16})
    possible_pairs = list(combinations(algorithms, 2))

    # just focus on a subset of pairs
    required_pairs = [("PPO", "TRPO"),("TRPO", "PPO"),("QRDQN", "DQN"),("DQN", "QRDQN"),("QRDQN","RecurrentPPO"),("RecurrentPPO","QRDQN"), ("A2C", "DQN"), ("DQN", "A2C"),
                      ("TRPO", "RecurrentPPO"), ("RecurrentPPO", "TRPO"), ("QRDQN", "A2C"), ("A2C", "QRDQN")]

    all_pairs = {}
    for pair in possible_pairs:
        if pair in required_pairs:
            pair_name = f'{pair[0]}_{pair[1]}'
            all_pairs[pair_name] = (
                scores_dict[pair[0]], scores_dict[pair[1]])

    probabilities, probability_cis = rly.get_interval_estimates(
        all_pairs, metrics.probability_of_improvement, reps=reps)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
    h = 0.6

    ax2 = ax.twinx()

    for i, (pair, p) in enumerate(probabilities.items()):
        (l, u), p = probability_cis[pair], p

        ax.barh(y=i, width=u - l, height=h,
                left=l, color="orange",
                alpha=0.75, label=pair[0])
        ax2.barh(y=i, width=u - l, height=h,
                 left=l, color="orange",
                 alpha=0.0, label=pair[1])
        ax.vlines(x=p, ymin=i - 7.5 * h / 16, ymax=i + (6 * h / 16),
                  color='k', alpha=0.85)

    ax.set_yticks(list(range(len(all_pairs))))
    ax2.set_yticks(range(len(all_pairs)))
    pairs = [x.split('_') for x in probabilities.keys()]
    ax2.set_yticklabels([pair[1] for pair in pairs], fontsize='large')
    ax.set_yticklabels([pair[0] for pair in pairs], fontsize='large')
    ax2.set_ylabel('Algorithm Y', fontweight='bold', rotation='horizontal', fontsize='x-large')
    ax.set_ylabel('Algorithm X', fontweight='bold', rotation='horizontal', fontsize='x-large')
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.yaxis.set_label_coords(0, 1.0)
    ax2.yaxis.set_label_coords(1.0, 1.0375)
    decorate_axis(ax, wrect=5)
    decorate_axis(ax2, wrect=5)

    ax.tick_params(axis='both', which='major', labelsize='x-large')
    ax2.tick_params(axis='both', which='major', labelsize='x-large')
    ax.set_xlabel('P(X > Y)', fontsize='xx-large')
    ax.grid(axis='x', alpha=0.2)
    plt.subplots_adjust(wspace=0.05)
    ax.spines['left'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(logs_folder,"probability_improvement.pdf"))
    plt.show()

def plot_ranks(scores_dict, algorithms, tasks, logs_folder="./logs"):
    plt.rcParams.update({'font.size': 16})
    #color_dict = dict(zip(algorithms, sns.color_palette('colorblind')))
    color_dict = {'A2C': (0.03784313725490196, 0.6496078431372549, 0.40098039215686275), 'DQN': (0.00392156862745098, 0.45098039215686275, 0.6980392156862745), 'PPO':   (0.8705882352941177, 0.5607843137254902, 0.0196078431372549), 'QRDQN':  (0.00392156862745098, 0.25098039215686275, 0.4980392156862745), 'RecurrentPPO': (0.8352941176470589, 0.3686274509803922, 0.0), 'TRPO': (0.792156862745098, 0.5686274509803921, 0.3803921568627451)}

    all_ranks = get_rank_matrix(
        scores_dict, 10000, algorithms=algorithms)
    mean_ranks = np.mean(all_ranks, axis=0)

    keys = algorithms
    labels = list(range(1, len(keys) + 1))
    width = 1.0  # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
    bottom = np.zeros_like(mean_ranks[0])
    for i, key in enumerate(keys):
        ranks = mean_ranks
        ax.bar(labels, ranks[i], width, color=color_dict[key],
                   bottom=bottom, alpha=0.9)
        bottom += mean_ranks[i]

        if i == 0:
            ax.set_ylabel('Probability Distribution', size='x-large')
    ax.set_xlabel('Ranking', size='x-large')
    ax.set_title(f'Mean Ranking', size='x-large')
    ax.set_xticks(labels)
    ax.set_ylim(0, 1)
    ax.set_xticklabels(labels, size='x-large')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='both', bottom=False, top=False,
                       left=False, right=False, labeltop=False,
                       labelbottom=True, labelleft=False, labelright=False)

    desired_order = ['PPO', 'TRPO', 'RecurrentPPO', 'QRDQN', 'DQN', 'A2C']

    fake_patches = [mpatches.Patch(color=color_dict[m], alpha=0.75)
                    for m in desired_order]
    fig.legend(fake_patches, desired_order, loc='upper center',
                        fancybox=True, ncols=3, fontsize='large')
    fig.subplots_adjust(top=0.85, wspace=0.1, hspace=0.05)
    plt.savefig(os.path.join(logs_folder,"ranks.pdf"))
    plt.show()

def difficulty_progress(scores, proportion=0.25):
    sorted_scores = sorted(scores.flatten())
    max_index = int(proportion * len(sorted_scores))
    return np.mean(sorted_scores[:max_index])

def plot_alternative_metrics(scores_dict, logs_folder="./logs"):
    # Compute Difficulty Progress/ SuperHuman Probability
    DP_25 = lambda scores: difficulty_progress(scores, 0.25)
    DP_50 = lambda scores: difficulty_progress(scores, 0.5)
    improvement_prob = lambda scores: np.mean(scores > 1)

    aggregate_func = lambda x: np.array([DP_25(x), DP_50(x), improvement_prob(x)])
    aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
        scores_dict, aggregate_func, reps=10000)

    algorithms = list(scores_dict.keys())
    metric_names = ['Difficulty Progress (25%)', 'Difficulty Progress (50%)',
                    'Superhuman Probability']
    plot_interval_estimates_shifted(
        aggregate_scores,
        aggregate_interval_estimates,
        algorithms=algorithms,
        metric_names=metric_names,
        xlabel='Human Normalized Score')
    plt.savefig(os.path.join(logs_folder,"alternative.pdf"))
    plt.show()
