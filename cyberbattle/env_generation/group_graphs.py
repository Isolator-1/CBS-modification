import re
import numpy as np
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

def group_folders(base_path, num_groups):
    folders = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])
    group_size = len(folders) // num_groups
    groups = [folders[i:i + group_size] for i in range(0, len(folders), group_size)]

    if len(groups) > num_groups:
        groups[-2].extend(groups[-1])
        groups.pop()

    return groups

def sample_files_from_group(base_path, groups, num_samples=100):
    sampled_files = {}
    score_pattern = re.compile(r'_([0-9]+\.[0-9]+)(?:_|\.)')

    for group_idx, group in enumerate(groups):
        all_files = []
        for folder in group:
            folder_path = os.path.join(base_path, folder)
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pkl')]
            scored_files = [(f, float(score_pattern.search(f).group(1))) for f in files if score_pattern.search(f)]
            all_files.extend(scored_files)

        # Sort files by score
        all_files.sort(key=lambda x: x[1])

        # Sample files to achieve an even distribution across the range
        sampled = np.linspace(0, len(all_files) - 1, num_samples, dtype=int)
        sampled_files[group_idx] = [all_files[i][0] for i in sampled]

    return sampled_files

import pickle

def save_sampled_files(base_path, sampled_files):
    grouped_folder_path = os.path.join(base_path, "grouped")
    os.makedirs(grouped_folder_path, exist_ok=True)
    for group_idx, files in sampled_files.items():
        group_folder_path = os.path.join(grouped_folder_path, f"group_{group_idx}")
        os.makedirs(group_folder_path, exist_ok=True)
        file_list = []
        for file_path in files:
            file_list.append(pickle.load(open(file_path, 'rb')))
        pickle.dump(file_list, open(os.path.join(group_folder_path, "networks.pkl"), "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Interpret and monitor metrics of a neural network in a given environment')
    parser.add_argument('-f','--folder', required=True, help='Path to the run folder')
    parser.add_argument('-n','--num_groups', type=int, default=5, help='Number of groups to divide the folders into')
    parser.add_argument('-s','--num_samples', type=int, default=100, help='Number of files to sample from each group')
    args = parser.parse_args()
    path = os.path.join("logs", "splitting_graphs", args.folder)
    groups = group_folders(path, args.num_groups)
    sampled_files = sample_files_from_group(path, groups, args.num_samples)

    for group_idx, files in sampled_files.items():
        print(f"Group {group_idx}:")
        print(f"  Number of files: {len(files)}")

    save_sampled_files(path, sampled_files)

