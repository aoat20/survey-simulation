'''can be used to define custom reward functions
should be able to take a log file and calculate the reward based on the log file
'''

import numpy as np
from survey_simulation import SurveySimulationGrid

# Dictionary to store registered reward functions
reward_functions_registry = {}

def register_reward_function(name):
    def decorator(func):
        reward_functions_registry[name] = func
        return func
    return decorator


### make into reward function class 


class RewardFunction():
    """
    RewardFunction class to calculate the rewards in the survey simulation grid class.
    Attributes:
        type (str): The type of reward function, either 'live' or 'log_file'.
        survey_simulation (SurveySimulationGrid): The survey simulation object, required if type is 'live'.
        log_file (str): The path to the log file, required if type is 'log_file'.
        reward_function_id (str): The identifier for the specific reward function to use.
    Methods:
        __init__(**kwargs):
            Initializes the RewardFunction with the given parameters.
        get_reward(**kwargs):
            Returns the reward based on the type of reward function.
        _get_live_reward(**kwargs):
            Calculates and returns the reward for the 'live' type reward function.
        _get_reward_from_log_file(**kwargs):
            Calculates and returns the reward for the 'log_file' type reward function.
    """



    def __init__(self, **kwargs):
        self.type = kwargs.get('type', 'live')
        #if live make sure survey simulation is provided in kwargs
        if self.type == 'live':
            self.survey_simulation = kwargs.get('survey_simulation', None)
            if self.survey_simulation is None:
                raise ValueError('Survey simulation must be provided for live reward function')
            #check if survey simulation is of the correct type
            if not isinstance(self.survey_simulation, SurveySimulationGrid):
                raise TypeError('Survey simulation must be of type SurveySimulationGrid')
        #if log file make sure log file is provided in kwargs
        elif self.type == 'log_file':
            self.log_file = kwargs.get('log_file', None)
            if self.log_file is None:
                raise ValueError('Log file must be provided for log file reward function')
            else:
                #create playback survey simulation with path to log file
                self.survey_simulation = SurveySimulationGrid('playback', agent_viz=0,plotter=0,log_file=self.log_file)

        self.reward_function_id = kwargs.get('reward_id', 'default')


    def get_reward(self, **kwargs):
        if self.type == 'live':
            return self._get_live_reward(**kwargs)
        elif self.type == 'log_file':
            return self._get_reward_from_log_file(**kwargs)
        else:
            raise ValueError('Invalid reward function type')

    def _get_live_reward(self, **kwargs):
        reward_function = reward_functions_registry.get(self.reward_function_id)
        if reward_function is None:
            raise ValueError(f'Reward function {self.reward_function_id} not found')
        return reward_function(self.survey_simulation,**kwargs)

    def _get_reward_from_log_file(self, **kwargs):
        #create playback survey simulation with path to log file and loop through the log file to get reward
        survey_end = kwargs.get('survey_end', None) 
        reward = 0

        #step through the simulation to get the reward
        if survey_end is None:
            running = True
            while running :
                running = self.survey_simulation.playback_step()
                reward += self._get_live_reward(**kwargs)
            return reward

        else:
            i = 0
            while  i < survey_end:
                running = self.survey_simulation.playback_step()
                reward += self._get_live_reward(**kwargs)
                i += 1
                if not running:
                    print ('Survey ended before requested step')
                    return reward
            return reward
                
        #calculate the reward up to requested step or end of survey


"""
Reward functions can be registered using the `register_reward_function` decorator. 
This allows you to define custom reward functions and register them with a unique name. 
These registered reward functions can then be used by specifying their name in the `reward_id` keyword argument 
when initializing the `RewardFunction` class.

Example of registering a reward function:

@register_reward_function('custom_reward')
def custom_reward_function(survey_simulation: SurveySimulationGrid, **kwargs):
    # Implement custom reward logic here
    reward = ...
    return reward

Example of using a registered reward function:

# Initialize the RewardFunction with the registered reward function name
reward_function = RewardFunction(type='live', survey_simulation=survey_simulation_instance, reward_id='custom_reward')

# Get the reward
reward = reward_function.get_reward()
"""



#implement reward functions here 

@register_reward_function('default')
def default_reward_function(survey_simulation: SurveySimulationGrid, step_scale=100):
    '''
    Default reward function for the RL environment
    step scale is the scale of the reward for current path reward
    # '''
    # cov_map_non_zero = np.count_nonzero(~np.isnan(survey_simulation.covmap.map_stack),
    #                                     axis=0)
    # reward = np.sum(cov_map_non_zero) / np.prod(cov_map_non_zero.shape)
    
    # print (survey_simulation.griddata.cov_map)
<<<<<<< HEAD
    cov_map_non_zero = np.count_nonzero(np.array(survey_simulation.griddata.cov_map),
                                        axis=0)
    reward = np.sum(cov_map_non_zero) / np.prod(cov_map_non_zero.shape)
    # print ('cov reward',reward)
    step_reward =survey_simulation.agent.get_current_path_len() / step_scale
    reward += step_reward
    # print ('step reward',step_reward)
=======
    cov_map_non_zero = np.count_nonzero(survey_simulation.griddata.cov_map[0])
    reward = cov_map_non_zero / survey_simulation.griddata.cov_map[0].size
    step_reward =survey_simulation.agent.get_current_path_len() / step_scale
    reward += step_reward
>>>>>>> 567a1849f7aa5bca1ffd43e0007c676beba7ee9a
    return reward


@register_reward_function('singlevalue')
def custom_reward_function1(survey_simulation: SurveySimulationGrid,value=1):
    '''
    Custom reward function 1 for the RL environment
    '''
    # Implement custom reward logic here
    reward = value
    # Example: reward based on some custom criteria
    return reward



@register_reward_function('rawpathreward')
def custom_reward_function2(survey_simulation: SurveySimulationGrid):
    '''
    Custom reward function 2 for the RL environment
    '''
    # Implement custom reward logic here

    reward  = survey_simulation.agent.get_current_path_len() 
    # Example: reward based on some custom criteria
    return reward



@register_reward_function('edge')
def custom_reward_function3(survey_simulation: SurveySimulationGrid,const_value=1, edge_pen = 100):
    '''
    Custom reward function 2 for the RL environment
    '''
    # Implement custom reward logic here
    const_reward = custom_reward_function1(survey_simulation, value=const_value)
    survey_simulation.check_termination()
    if survey_simulation.termination_reason == 'grounded':
        return -edge_pen
    else:
        return const_reward


