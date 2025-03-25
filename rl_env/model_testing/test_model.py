import gymnasium as gym
import rl_env
from stable_baselines3 import PPO

# MODEL_PATH = '/Users/edwardclark/Documents/SURREY/models/tmjaxm1s/model.zip'
# MODEL_PATH = '/Users/edwardclark/Documents/SURREY/survey-simulation/models/5hixnng4/model.zip'
# MODEL_PATH = '/Users/edwardclark/Desktop/ppo_model_server_86500000_steps.zip'
# MODEL_PATH ='/Volumes/eprc20/ppo_model_server_9500000_steps.zip'
MODEL_PATH = '/Users/edward/Downloads/model (2).zip'
# MODEL_PATH = '/Users/edward/Documents/university/coding/survey-simulation/wandb/run-20241118_205309-l8pi0a9i/files/model.zip'

# MODEL_PATH ='/Users/edward/Downloads/latest.zip'
VIS_STEPS = 1000
#create env 

# env_kwargs = {
#         'params_filepath':  '/Users/edward/Documents/university/coding/survey-simulation/rl_env/params.txt',
#         'save_logs': True,
#         'obs_type': 'coverage_occupancy'
#     }



env_kwargs = {
            'params_filepath': '/Users/edward/Documents/university/coding/survey-simulation/rl_env/training/initial_params.txt',
            'save_logs': True,
            'obs_type': 'coverage_occupancy'
        }
# # env_kwargs = {# #         'params_filepath': '/Users/edward/Documents/university/coding/survey-simulation/rl_env/training/initial_params.txt',
# #         'save_logs': False,
# #         'obs_type': 'coverage_occupancy',
# #         'reward_kwargs':{
# #             'reward_id':'rawpathreward',
# #         }

#     }

env = gym.make('BasicEnv-v0', **env_kwargs)



#load the model
model = PPO.load(MODEL_PATH, env=env, verbose = 1)

#run the model for N steps

obs,_ = env.reset()
reward = 0
for i in range(VIS_STEPS):
    action, _states = model.predict(obs)
    obs, rewards, terminated ,  truncated, info = env.step(action)
    if rewards < 0:
        print ('step reward', rewards)
    # print ('reward', rewards)
    reward += rewards

    # env.render()
    if terminated or truncated:
        # print ('terminated')
        if reward > 0:
            print ('reward', reward)
        reward = 0
        obs , _ = env.reset()

    