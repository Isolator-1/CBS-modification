from gym import spaces
import gym.spaces as gym_spaces
import gymnasium.spaces as gymnasium_spaces

# Flatten observation into Discrete or Box supported types as requested by StableBaselines3
def flatten_dict_with_arrays(input_dict):
    flattened_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):
            # If the value is a dictionary, flatten it recursively
            flattened_subdict = flatten_dict_with_arrays(value)
            flattened_dict.update(
                {f"{key}_{sub_key}": sub_value for sub_key, sub_value in flattened_subdict.items()})
        elif isinstance(value, list):
            # If the value is a list, flatten it to individual components
            for i, sub_value in enumerate(value):
                if isinstance(sub_value, tuple):
                    for j, inner_sub_value in enumerate(sub_value):
                        flattened_dict[f"{key}_{i}_{j}"] = inner_sub_value
                else:
                    flattened_dict[f"{key}_{i}"] = sub_value
        else:
            # If the value is not a dictionary or list, include it as is
            flattened_dict[key] = value
    return flattened_dict

# Flatten Discrete and Box Spaces in hierarchies in pure single hierarchy of Discrete and Box Spaces
def flatten_space_dict_with_arrays(input_dict):
    flattened_dict = {}

    for key, value in input_dict.items():
        if isinstance(value, spaces.Dict):
            # If the value is a dictionary, flatten it recursively
            flattened_subdict = flatten_space_dict_with_arrays(value)
            flattened_dict.update(
                    {f"{key}_{sub_key}": sub_value for sub_key, sub_value in flattened_subdict.items()})
        elif isinstance(value, spaces.Tuple):
            # If the value is a tuple, flatten it to individual components
            for i, sub_space in enumerate(value):
                if isinstance(sub_space, spaces.Tuple):
                    # If the element of the tuple is another tuple, flatten it again
                    for j, nested_sub_space in enumerate(sub_space):
                        if isinstance(nested_sub_space, spaces.Dict):
                            # Recursively flatten nested dictionary in the tuple
                            nested_flattened_dict = flatten_space_dict_with_arrays({f"{key}_{i}_{j}_{sub_key}": sub_value for sub_key, sub_value in nested_sub_space.items()})
                            flattened_dict.update(nested_flattened_dict)
                        elif isinstance(nested_sub_space, spaces.Box):
                            # Flatten box components
                            flattened_dict[f"{key}_{i}_{j}"] = nested_sub_space
                        elif isinstance(nested_sub_space, spaces.Discrete):
                            # Flatten discrete components
                            flattened_dict[f"{key}_{i}_{j}"] = nested_sub_space
                elif isinstance(sub_space, spaces.Dict):
                    # Recursively flatten nested dictionary in the tuple
                    nested_flattened_dict = flatten_space_dict_with_arrays(
                        {f"{key}_{i}_{sub_key}": sub_value for sub_key, sub_value in sub_space.items()})
                    flattened_dict.update(nested_flattened_dict)
                elif isinstance(sub_space, spaces.Box):
                    # Flatten box components
                    flattened_dict.update(
                        {f"{key}_{i}_{sub_key}": sub_value for sub_key, sub_value in
                            zip(sub_space.spaces.keys(), sub_space)}
                    )
                elif isinstance(sub_space, spaces.Discrete):
                    # Flatten discrete components
                    flattened_dict[f"{key}_{i}"] = sub_space
        else:
            # If the value is not a dictionary or tuple, include it with indices
            flattened_dict[key] = value

    return spaces.Dict(flattened_dict)

def hide_features(input_dict, visible_prefixes=None):
    if visible_prefixes == None:
        return input_dict
    else:
        return {key: value for key, value in input_dict.items() if
                    any(key.__contains__(prefix) for prefix in visible_prefixes)}

def concatenate_or_none(list1, list2):
    if list1 is not None and list2 is not None:
        # Concatenate if both lists are not None
        return list1 + list2
    elif list1 is not None:
        # Use list1 if only list2 is None
        return list1
    elif list2 is not None:
        # Use list2 if only list1 is None
        return list2
    else:
        # Both lists are None, return None
        return None

 # This class must be gymnasium for the stable baselines 3 implementation check environment
def convert_to_gymnasium(observation_space):
    if not isinstance(observation_space, gym_spaces.Dict):
        raise ValueError("Input must be a gym.spaces.Dict")

    gymnasium_observation_space_dict = {}

    for key, space in observation_space.spaces.items():
        if isinstance(space, gym_spaces.Discrete):
            gymnasium_observation_space_dict[key] = gymnasium_spaces.Discrete(space.n)
        elif isinstance(space, gym_spaces.Box):
            gymnasium_observation_space_dict[key] = gymnasium_spaces.Box(low=space.low, high=space.high, dtype=space.dtype)
        elif isinstance(space, gym_spaces.Dict):
            gymnasium_observation_space_dict[key] = convert_to_gymnasium(space)
        else:
            raise ValueError(f"Unsupported space type for key='{key}': {type(space)}")

    return gymnasium_spaces.Dict(gymnasium_observation_space_dict)
