import gymnasium as gym
from gymnasium import spaces
print (gym.__version__)
import numpy as np
from survey_simulation import SurveySimulation
from survey_simulation import SurveySimulationGrid
from survey_simulation import SEASSimulation



from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


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
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64) #for now just return the time step


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




        return observation, reward, terminated, truncated, info 



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
        observation = np.array([t / self.survey_simulation.timer.time_lim])
        return observation
    
    def get_reward(self):

        # reward = -1 #default reward is -1 for each step to minimize the number of steps

        #updated reward function here (base it on the coverage map)


        cov_map_non_zero =  np.count_nonzero(~np.isnan(self.survey_simulation.covmap.map_stack),
                                            axis=0)
        reward = np.sum(cov_map_non_zero) / np.prod(cov_map_non_zero.shape)

        return reward

    def reset(self, *, seed = None, options = None) :
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
    'save_logs':True
}

env = BasicEnv('/Users/edwardclark/Documents/SURREY/survey-simulation/params.txt',**kwargs)
# print(env.observation_space)
# print(env.action_space)

# obs = env.reset(seed=0,options={})
# print(obs)


# action = 1

# for i in range(100):
#     # action = env.action_space.sample()

#     obs, reward, terminated, truncated, info, done = env.step(action)
#     #every 5 steps increment action by 1
#     if i % 15 == 0:
#         action += 5
#     if reward > 0:
#         print('reward:', reward)

#     if terminated:
#         action = 1
#         env.reset(seed=0,options={})
#         print('resetting')


#check the environment with sb3

# check_env(env)


#train the environment with sb3



# model = PPO("MlpPolicy", env, verbose = 1, n_steps=5000, n_epochs=2)
# model.learn(total_timesteps=10e6)
# model.save("ppo_survey_simulation_test_2")



model_path = 'ppo_survey_simulation_test_2.zip'
model = PPO.load(model_path)

obs , info = env.reset(seed=0,options={})
print   (obs)


for i in range(1000):
    action  = model.predict(obs)
    print (action)
    obs, reward, terminated, truncated, info = env.step(action[0])
    env.render()

    if terminated:
        obs, info = env.reset(seed=0,options={})
        print('resetting')

