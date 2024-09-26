import pickle
import argparse
import os
import sys
import shutil
import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

def compute_score(dict):
    return 0.25* dict["knows_connectivity"] + 0.75 * dict["access_connectivity"]
    # + 0.25 * dict["knows_rechability"] + 0.25 * dict["access_reachability"]

def load_and_copy_pkls():
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join("logs", "splitting_graphs", current_date)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    networks = {}
    stats = {}

    for i in range(10):
        networks[i] = []
        stats[i] = []
        lower_bound = i / 10.0
        upper_bound = (i + 1) / 10.0
        subfolder_name = f"{lower_bound:.1f}-{upper_bound:.1f}"
        subfolder_path = os.path.join(output_folder, subfolder_name)

        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)


    target_dir = os.path.join("logs")
    folders = [folder for folder in os.listdir(target_dir) if folder.startswith("graphs_generation")]
    print("Folders:", folders)


    for folder in folders:
        print(f"Loading .pkl files from folder: {folder}")
        folder = os.path.join(target_dir, folder)
        files = [f for f in os.listdir(folder) if f.endswith(".pkl")]

        for file in files:

            file_path = os.path.join(folder, file)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    score = compute_score(data[1])

                    net = data[0]
                    stat = data[1]
                    # Copy the file to the appropriate subfolder based on the score
                    for i in range(10):
                        lower_bound = i / 10.0
                        upper_bound = (i + 1) / 10.0
                        if lower_bound <= score < upper_bound:
                            subfolder_name = f"{lower_bound:.1f}-{upper_bound:.1f}"
                            subfolder_path = os.path.join(output_folder, subfolder_name)
                            new_file_name = f"network_{score:.3f}.pkl"
                            new_file_path = os.path.join(subfolder_path, new_file_name)
                            # Handle existing files with the same name by appending a postfix
                            count = 1
                            while os.path.exists(new_file_path):
                                new_file_name = f"network_{score:.3f}_{count}.pkl"
                                new_file_path = os.path.join(subfolder_path, new_file_name)
                                count += 1
                            shutil.copy(file_path, new_file_path)
                            break
                    networks[i].append(net)
                    stats[i].append(stat)
            except Exception as e:
                print(f"Error loading file {file}: {e}")
    for i in range(10):
        print("Score:", i)
        lower_bound = i / 10.0
        upper_bound = (i + 1) / 10.0
        subfolder_name = f"{lower_bound:.1f}-{upper_bound:.1f}"
        subfolder_path = os.path.join(output_folder, subfolder_name)
        with open(os.path.join(subfolder_path, "networks.pkl"), "wb") as f:
            pickle.dump(networks[i], f)
        print("Number of elements in split:", len(networks[i]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and print contents of .pkl files from specified folders.')
    args = parser.parse_args()
    load_and_copy_pkls()
