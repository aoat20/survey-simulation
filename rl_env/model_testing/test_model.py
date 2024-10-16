import gymnasium as gym
import rl_env
from stable_baselines3 import PPO

MODEL_PATH = '/Users/edwardclark/Downloads/ppo_model_server_30000000_steps.zip'
VIS_STEPS = 100

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
for i in range(VIS_STEPS):
    action, _states = model.predict(obs)
    obs, rewards, terminated ,  truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs , _ = env.reset()

    