# CyberBattleSim with a Local View MDP
## Modification for the paper "Leveraging Deep Reinforcement Learning for Cyber-Attack Paths Prediction: Formulation, Generalization, and Evaluation"
### Franco Terranova, Abdelkader Lahmadi, Isabelle Chrisment
### Université de Lorraine, CNRS, Inria, LORIA, 54000 Nancy, France
### Accepted at The 27th International Symposium on Research in Attacks, Intrusions and Defenses (RAID 2024), September 30-October 02, 2024, Padua, Italy
This repository contains a modification of CyberBattleSim, an experimentation research platform to investigate the interaction of automated agents operating in a simulated abstract enterprise network environment.
The description of the original framework is contained in the file **README_ORIGINAL.md**.

This project relies on the utilization of the following main libraries:
- CyberBattleSim
- RLiable (https://github.com/google-research/rliable/tree/master)
- StableBaselines3
- Optuna

## Setting up the environment
Install the required packages by running the following command:
```
conda create --name cyberbattlesim
conda activate cyberbattlesim
```

```
pip install -r local.requirements.txt
```

The following guide contains general instructions about how to use the CyberBattleSim modification.
The exact commands used in the paper are instead contained in the file **paper_commands.md**.

## Run the agents (Train/Test)

### Train the agent

```
cd cyberbattle/agents/stable_baselines/
```
To use the same groups of seeds and networks used in the paper, you can use the following commands:

```
python3 train_agent.py --environment random --switch --load_nets group_X --load_seeds seeds_20 --algorithm ALGORITHM --name randomX_ALGORITHM
```
Where X is the group of networks you want to use [1 (connectivity in [0.2, 0.4]), ..., 4 (connectivity in 0.8-1.0)], and ALGORITHM is the algorithm you want to use to train the agent [ppo, a2c, trpo, dqn, qr_dqn, recurrent_ppo].
Loading a fixed number of seeds already set the number of runs.
The number of runs can be set as well with the command '--num_runs'.
The train_config.yaml file contains the hyperparameters for the algorithms and the simulation.

### Hyperparameters optimization
To run an hyperparameters optimization of the algorithms, you can use the following commands:

```
python3 train_agent_hyperopt.py --algorithm ALGORITHM --environment random --switch --num_runs 3 --load_nets single_group --load_seeds seeds_3 --num_trials 50 --name hyperopt_ALGORITHM
```
Where ALGORITHM is the algorithm you want to use to train the agent [ppo, a2c, trpo, dqn, qr_dqn, recurrent_ppo].
The hyper-parameters optimization of the paper works on a single group of 200 networks in the range [0.2, 1.0] for 50 trials of 3 runs each.
These parameters can be modified properly.
The train_config.yaml file contains the hyperparameters for the algorithms and the simulation, while the hyperparams.yaml file contains the hyperparameters' ranges for the hyperparameters optimization.

### Random agent run
To train the random agent, you can use the following command:

```
python3 train_agent.py --environment random --switch --load_nets group_X --name randomX_random --random_agent
```

### Plot of metrics using the RLiable library

To plot the metrics of the agents, you can use the following command:
```
cd cyberbattle/agents/rliable/
```
```
python3 rliable_plot.py -f LIST -o OPTION
```
Where LIST is the list of folders for each algorithm and each game. 
E.g. random2-4_a2c_2024-02-29_10-10-47 random2-4_trpo_2024-02-29_10-48-10 ....
The OPTION is the type of rliable plot desidered [interval_estimate, rank, histogram, performance_profile, probability_improvement, alternative, difficulty_progress]

### Plot of tensorboard metrics

To plot the metrics of the agents using tensorboard, you can use the following command:
```
cd cyberbattle/plot/
```
```
python3 plot_tensorboard_metrics.py --logs_folder FOLDER --metric METRICS
```
Where FOLDER is the logs folder targeted (e.g. random8-10_ppo_2024-02-29_10-49-38) and METRICS is the list of metrics to be plot together (e.g. val_avg_owned_percentage val_avg_discovered_percentage val_avg_discovered_credentials_percentage val_avg_owned_percentage_reachable).

### Run the agent on a test set and generate trajectories
To run the agent on a test set and generate the trajectories, you can use the following command:
```
cd cyberbattle/agents/
```
```
python3 test_agent.py --environment random --switch --load_test_envs --last_checkpoint --val_checkpoint --run_folder FOLDER --algorithm ALGORITHM --option trajectories
```
The agent can also be run in sets different than the test set. In this default command the best validation checkpoint is used. 
The run to be used can also be set with the command '--run_id'.
Other metrics can be generated (e.g. average_performances).
The test_config.yaml file contains the hyperparameters for the test run and the simulation.

### Generate statistics on the attack paths
To generate statistics on the attack paths, you can use the following command:
```
cd cyberbattle/agents/trajectories_utils/
```
```
python3 stats_trajectories.py -f CLEAN_TRAJ_PATH
```
Where CLEAN_TRAJ_PATH is the path of the clean trajectories to be analyzed.

### Generate new groups
To generate new topologies and divide them into groups, use the following commands:
```
cd cyberbattle/env_generation/
python3 generate_graphs.py
```
The parameters of the generation should be set with the graphs_generation.yaml file.
The graphs can then be organized by connectivity with the following command:
```
python3 split_graphs.py
```
The graphs can then be grouped with the following command:
```
python3 group_graphs.py -f TIMESTAMP_FOLDER
```
During the execution of the commands, the graphs will be saved in the logs folder of graphs_generation and TIMESTAMP_FOLDER will be the folder containing the graphs organized to be grouped.

## LICENSE
The original MIT License will continue to apply also to the modification of the codebase.

## Note on privacy
This project does not include any customer data.
The provided models and network topologies are purely fictitious.
Users of the provided code provide all the input to the simulation
and must have the necessary permissions to use any provided data.

