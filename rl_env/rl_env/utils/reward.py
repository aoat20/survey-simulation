'''
reward functions for the RL environment
can be used to define custom reward functions
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
                reward += self._get_live_reward()
            return reward

        else:
            i = 0
            while  i < survey_end:
                running = self.survey_simulation.playback_step()
                reward += self._get_live_reward()
                i += 1
                if not running:
                    print ('Survey ended before requested step')
                    return reward
            return reward
                
        #calculate the reward up to requested step or end of survey

        return NotImplementedError('Implement this method to get reward from log file')



@register_reward_function('default')
def default_reward_function(survey_simulation: SurveySimulationGrid, step_scale=100):
    '''
    Default reward function for the RL environment
    step scale is the scale of the reward for current path reward
    '''
    cov_map_non_zero = np.count_nonzero(~np.isnan(survey_simulation.covmap.map_stack),
                                        axis=0)
    reward = np.sum(cov_map_non_zero) / np.prod(cov_map_non_zero.shape)
    step_reward =survey_simulation.agent.get_current_path_len() / step_scale
    reward += step_reward
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






