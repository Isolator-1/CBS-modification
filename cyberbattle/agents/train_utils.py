import numpy as np
from typing import Callable
import gymnasium
import torch.nn as nn
import torch.optim as optim

def replace_with_classes(policy_kwargs):
    # Handle activation functions
    activation_functions = {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "Tanh": nn.Tanh,
        "Sigmoid": nn.Sigmoid,
        "ELU": nn.ELU
    }
    if 'activation_fn' in policy_kwargs:
        policy_kwargs['activation_fn'] = activation_functions[policy_kwargs['activation_fn']]

    # Handle optimizers
    optimizers = {
        "Adam": optim.Adam,
        "SGD": optim.SGD
    }
    if 'optimizer_class' in policy_kwargs:
        policy_kwargs['optimizer_class'] = optimizers[policy_kwargs['optimizer_class']]

    return policy_kwargs

# convert dictionary representing the state into an array of values
def dict_to_array(input_dict):
    keys = input_dict.keys()
    values = [input_dict[key] for key in keys]
    for i in range(len(values)):
        if isinstance(values[i], list):
            values[i] = values[i][0]
        if isinstance(values[i], np.ndarray):
            values[i] = values[i][0]
    return np.array(values, dtype=np.float32)


def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return initial_value + (final_value - initial_value) * (1.0 - progress_remaining)
    return func


def get_box_variables(observation_space_var):
    box_variable_names = []
    if isinstance(observation_space_var, gymnasium.spaces.Box):
        # if box -> add
        return True

    elif isinstance(observation_space_var, gymnasium.spaces.Dict):
        # if dict -> recursively check each variable
        for key, sub_space in observation_space_var.spaces.items():
            if get_box_variables(sub_space):
                box_variable_names.append(key)

    elif isinstance(observation_space_var, gymnasium.spaces.Tuple):
        # if tuple -> recursively check each variable
        for key, sub_space in enumerate(observation_space_var):
            if get_box_variables(sub_space):
                box_variable_names.append(key)

    return box_variable_names
