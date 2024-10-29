import gymnasium as gym
import rl_env
from stable_baselines3 import PPO

# MODEL_PATH = '/Users/edwardclark/Documents/SURREY/models/tmjaxm1s/model.zip'
# MODEL_PATH = '/Users/edwardclark/Documents/SURREY/survey-simulation/models/5hixnng4/model.zip'
MODEL_PATH = '/Users/edwardclark/Desktop/ppo_model_server_86500000_steps.zip'
# MODEL_PATH ='/Volumes/eprc20/ppo_model_server_9500000_steps.zip'
VIS_STEPS = 3000

#create env 

env_kwargs = {
        'params_filepath': '/Users/edwardclark/Documents/SURREY/survey-simulation/rl_env/params.txt',
        'save_logs': True,
        'obs_type': 'coverage_occupancy'
    }

env = gym.make('BasicEnv-v0', **env_kwargs)



#load the model
model = PPO.load(MODEL_PATH, env=env, verbose = 1)

#run the model for N steps

obs,_ = env.reset()
reward = 0
for i in range(VIS_STEPS):
    action, _states = model.predict(obs)
    obs, rewards, terminated ,  truncated, info = env.step(action)
    # print ('reward', rewards)
    reward += rewards

    # env.render()
    if terminated or truncated:
        print ('terminated')
        print ('reward', reward)
        reward = 0
        obs , _ = env.reset()

    