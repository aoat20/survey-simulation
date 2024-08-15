import gymnasium as gym
from gymnasium import spaces
print (gym.__version__)
import numpy as np
from survey_simulation import SurveySimulation
from survey_simulation import SurveySimulationGrid
from survey_simulation import SEASSimulation

class BasicEnv(gym.Env):

    '''
    Basic environment for the survey simulation
    observation space - time step
    action space - move in 360 / n_actions degree increments
    

    plan:
    - build out the observation space to include agent position, occupancy map, coverage map, contacts]
    - build out the action space to include move, group, ungroup, etc
    '''



    def __init__(self,params_filepath, **kwargs) -> gym.Env:

        default_config = {}
        self.config = default_config

        self.set_action_space()
        self.set_observation_space()

        self.survey_simulation = SurveySimulationGrid('test',
                                                      save_dir='data',
                                                      params_filepath = params_filepath) #initialize the survey simulation in manual mode
        
        self.save_logs = kwargs.get('save_logs', False)



    def set_action_space(self,n_actions=60):
        #for now have 360 degree movement in 360 / 60 = 6 degree increments
        self.action_space = spaces.Discrete(n_actions)
        self.actions = np.linspace(0,360,n_actions)
    
    def set_observation_space(self):

        #this should match the observation space of the simulation
        #next step returns a tuple (t, agent_pos, occ_map, cov_map, cts)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,)) #for now just return the time step


    def step(self, action):


        #for now just have move action
        self.survey_simulation.new_action('move',self.actions[action])

        observation = self._get_observation()
        reward = self.get_reward()


        terminated = False
        truncated = False
        info = {}
        done = False

        if self.survey_simulation.end_episode:
            info['termination_reason'] = self.survey_simulation.termination_reason
            terminated = True




        return observation, reward, terminated, truncated, info , done



    def _get_observation(self):
        '''
        Get the current observation from the simulation
        t - time step
        agent_pos - agent position
        occ_map - occupancy map
        cov_map - coverage map
        cts - contacts
        '''
        t, agent_pos, occ_map, cov_map, cts = self.survey_simulation.next_step()



        #for now just return the time step we will add more later
        observation = t / self.survey_simulation.timer.time_lim
        return observation
    
    def get_reward(self):

        reward = -1 #default reward is -1 for each step to minimize the number of steps
        return reward

    def reset(self, *, seed, options) :
        super().reset(seed=seed)


        if self.save_logs:
            self.survey_simulation.save_episode()

        info = {}
        self.survey_simulation.reset()

        return self._get_observation() , info

    def render(self):
        pass





#test the environment


kwargs = {
    'save_logs': True
}

env = BasicEnv('/Users/edwardclark/Documents/SURREY/survey-simulation/params.txt',**kwargs)
print(env.observation_space)
print(env.action_space)

obs = env.reset(seed=0,options={})
print(obs)

for i in range(300):
    # action = env.action_space.sample()
    action = 0

    print (env.actions[action])
    obs, reward, terminated, truncated, info, done = env.step(action)
    print(obs, reward, terminated, truncated, info, done)
    if terminated:
        env.reset(seed=0,options={})
        print('resetting')

