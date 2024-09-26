# Commands reproducibility

Follow this set of commands to reproduce the results of the paper. They should be run from the cyberbattle/agents/stable_baselines/ folder.

P.S. these commands rely on the presence of the topologies and seeds in the logs folder, present by default.

## Hyperparameters optimization
For each algorithm, the hyperparameters optimization is done on a single group of 200 networks in the range [0.2, 1.0] for 50 trials of 3 runs each.
Here the commands are shown for the PPO algorithm, but the same can be done for the other algorithms (change algorithm option and name option).
```
cd cyberbattle/agents/stable_baselines/
```
```
python3 train_agent_hyperopt.py --algorithm ppo --environment random --switch --num_runs 3 --load_nets single_group --load_seeds seeds_3 --num_trials 50 --name hyperopt_ppo
```
The optimal hyper-parameters should be used to update the train_config.yaml file.

## Train on each group of networks
For each algorithm, the training is done on each group for 20 runs.
Here the commands are shown for the DQN algorithm, but the same can be done for the other algorithms (change algorithm option and name option).
```
python3 train_agent.py --environment random --switch --load_nets group_4 --load_seeds seeds_20 --algorithm dqn --name random8-10_dqn

python3 train_agent.py --environment random --switch --load_nets group_3 --load_seeds seeds_20 --algorithm dqn --name random6-8_dqn

python3 train_agent.py --environment random --switch --load_nets group_2 --load_seeds seeds_20 --algorithm dqn --name random4-6_dqn

python3 train_agent.py --environment random --switch --load_nets group_1 --load_seeds seeds_20 --algorithm dqn --name random2-4_dqn
```

## Run a random agent to gather the baseline performances
Train random agent for one run per simplicity using the proper option:
```
python3 train_agent.py --environment random --switch --load_nets group_1 --algorithm ppo --name random2-4_random --random_agent

python3 train_agent.py --environment random --switch --load_nets group_2 --algorithm ppo --name random4-6_random --random_agent

python3 train_agent.py --environment random --switch --load_nets group_3 --algorithm ppo --name random6-8_random --random_agent

python3 train_agent.py --environment random --switch --load_nets group_4 --algorithm ppo --name random8-10_random --random_agent
```
The AUC of the validation curve of each random agent should be stored in the proper dict of the cyberbattle/agents/rliable/metrics_utils file.

## Use RLiable to plot the metrics

To plot the metrics of the agents, you can use the following command:
```
cd cyberbattle/agents/rliable/
```
```
python3 rliable_plot.py -f random2-4_a2c_2024-02-29_10-10-47  	random2-4_trpo_2024-02-29_10-48-10 	random4-6_qrdqn_2024-02-29_10-44-26	random6-8_ppo_2024-02-29_10-48-45	random8-10_lstmppo_2024-02-29_10-22-02 random2-4_dqn_2024-02-29_10-41-11  	random4-6_a2c_2024-02-29_10-12-23  	random4-6_trpo_2024-02-29_10-48-01 	random6-8_qrdqn_2024-02-29_10-45-22  random8-10_ppo_2024-02-29_10-49-38 random2-4_lstmppo_2024-02-29_10-15-47  random4-6_dqn_2024-02-29_10-41-10  	random6-8_a2c_2024-02-29_10-13-13  	random6-8_trpo_2024-02-29_10-49-04   random8-10_qrdqn_2024-02-29_10-46-24 random2-4_ppo_2024-02-29_10-47-38  	random4-6_lstmppo_2024-02-29_10-16-52  random6-8_dqn_2024-02-29_10-41-27  	random8-10_a2c_2024-02-29_10-14-32   random8-10_trpo_2024-02-29_10-47-47 random2-4_qrdqn_2024-02-29_10-42-36	random4-6_ppo_2024-02-29_10-47-18  	random6-8_lstmppo_2024-02-29_10-20-05  random8-10_dqn_2024-02-29_10-41-00 -o OPTION
```
Where OPTION should be replaced with the type of metric that you want: [rank, interval_estimate, histogram, performance_profile, probability_improvement].

## Test the best agent on the test environment of each group of networks

To test the best agent on the test environment of each group of networks, you can use the following command:
```
cd cyberbattle/agents/
```
```
python3 test_agent.py --environment random --switch --load_test_envs --last_checkpoint --val_checkpoint --run_folder random8-10_ppo_2024-02-29_10-49-38 --algorithm ppo --option average_performances --test_folder random6-8_ppo_2024-02-29_10-48-45   
```
Iterate the run_folder and test_folder across all combination. 
A stats folder will be created inside each folder with the summary metrics.

## Generate trajectories and stats

To generate the trajectories on the test set of the best group and best algorithm, you can use the following command:
```
python3 test_agent.py --environment random --switch --load_test_envs --last_checkpoint --val_checkpoint --run_folder random6-8_ppo_2024-02-29_10-48-45 --algorithm dqn --option trajectories
```
The clean_trajectories file generated in the stats folder can then be used to generate statistics and plots.
```
python3 stats_trajectories.py -f random6-8_ppo_2024-02-29_10-48-45/stats/validation/merged/clean_trajectories_merged_100_1000.csv
```
