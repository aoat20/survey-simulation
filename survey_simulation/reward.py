'''can be used to define custom reward functions
should be able to take a log file and calculate the reward based on the log file
'''

import numpy as np

# Dictionary to store registered reward functions
reward_functions_registry = {}


def register_reward_function(name):
    def decorator(func):
        reward_functions_registry[name] = func
        return func
    return decorator


# make into reward function class


class RewardFunction():
    """
    RewardFunction class to calculate the rewards in the survey simulation grid class.
    Attributes:
        type (str): The type of reward function, either 'live' or 'log_file'.
        survey_simulation (SurveySimulationGrid): The survey simulation object, 
        required if type is 'live'.
        log_file (str): The path to the log file, required if type is 'log_file'.
        reward_function_id (str): The identifier for the specific reward function to use.
    Methods:
        __init__(**kwargs):
            Initializes the RewardFunction with the given parameters.
        get_reward(**kwargs):
            Returns the reward based on the type of reward function.
        _get_live_reward(**kwargs):
            Calculates and returns the reward for the 'live' type reward function.
    """

    def __init__(self, **kwargs):

        self.reward_function_id = kwargs.get('reward_id', 'incremental')
        self.rewards = [0]
        self.reward = 0

    def get_reward(self,
                   obs_dict,
                   **kwargs):
        reward_function = reward_functions_registry.get(
            self.reward_function_id)
        if reward_function is None:
            raise ValueError(
                f'Reward function {self.reward_function_id} not found')
        self.reward = reward_function(obs_dict, **kwargs)
        self.rewards.append(self.rewards[-1]+self.reward)

    def reset(self):
        self.rewards = [0]


"""
Reward functions can be registered using the `register_reward_function` decorator. 
This allows you to define custom reward functions and register them with a unique name. 
These registered reward functions can then be used by specifying their name 
in the `reward_id` keyword argument 
when initializing the `RewardFunction` class.

Example of registering a reward function:

@register_reward_function('custom_reward')
def custom_reward_function(survey_simulation: SurveySimulationGrid, **kwargs):
    # Implement custom reward logic here
    reward = ...
    return reward

Example of using a registered reward function:

# Initialize the RewardFunction with the registered reward function name
reward_function = RewardFunction(type='live', 
                                survey_simulation=survey_simulation_instance, 
                                reward_id='custom_reward')

# Get the reward
reward = reward_function.get_reward()
"""


# implement reward functions here

@register_reward_function('default')
def default_reward_function(obs_dict: dict, step_scale=100):
    '''
    Default reward function for the RL environment
    step scale is the scale of the reward for current path reward
    # '''
    # cov_map_non_zero = np.count_nonzero(~np.isnan(survey_simulation.covmap.map_stack),
    #                                     axis=0)
    # reward = np.sum(cov_map_non_zero) / np.prod(cov_map_non_zero.shape)

    cov_map = obs_dict.get('cov_map')
    path_l = obs_dict.get('path_length')

    cov_map_non_zero = np.count_nonzero(np.array(cov_map),
                                        axis=0)
    reward = np.sum(cov_map_non_zero) / np.prod(cov_map_non_zero.shape)
    # print('cov reward', reward)
    step_reward = path_l / step_scale
    reward += step_reward
    # print('step reward', step_reward)
    return reward


@register_reward_function('singlevalue')
def custom_reward_function1(survey_simulation, value=1):
    '''
    Custom reward function 1 for the RL environment
    '''
    # Implement custom reward logic here
    reward = value
    # Example: reward based on some custom criteria
    return reward



@register_reward_function('incremental')
def incremental_reward_function(obs_dict: dict, step_scale=10000):
    '''
    Default reward function for the RL environment
    step scale is the scale of the reward for current path reward
    '''
    cov_map = obs_dict.get('cov_map')
    path_l = obs_dict.get('path_length')

    cov_map_non_zero = np.count_nonzero(np.array(cov_map), axis=0)
    current_cov = np.sum(cov_map_non_zero) / np.prod(cov_map_non_zero.shape)

    # Calculate the difference with the previous value
    reward_difference = current_cov - incremental_reward_function.previous_cov

    # Update the previous reward value
    incremental_reward_function.previous_cov =  current_cov

    # Calculate the step reward
    step_reward = path_l / step_scale

    # step_reward = (path_l-1) / step_scale
    
    # step_reward = 0

    # Total reward
    reward = reward_difference + step_reward

    return reward

# Initialize the previous_reward attribute
incremental_reward_function.previous_cov = 0    