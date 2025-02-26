import rl_env  # This will automatically register your custom environment
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import SubprocVecEnv, make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from wandb.integration.sb3 import WandbCallback

import wandb

if __name__ == "__main__":


    N_ENVS = 4
    MODEL_PATH = '/Users/edward/Downloads/model.zip'

    retrain = False

    config = {
        "policy_type": "CnnPolicy",
        "total_timesteps": 1e6,
        "env_name": "BasicEnv-v0",
    }
    run = wandb.init(
        project="sb3",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )





    env_kwargs = {
            # 'params_filepath': '/Users/edwardclark/Documents/SURREY/survey-simulation/rl_env/params.txt',
            'params_filepath': '/Users/edward/Documents/university/coding/survey-simulation/rl_env/training/initial_params.txt',
            'save_logs': False,
            'obs_type': 'coverage_occupancy'
        }
    env = make_vec_env(config['env_name'], n_envs=N_ENVS, env_kwargs=env_kwargs)
    env = VecMonitor(env)

    if retrain:
        model = PPO.load(MODEL_PATH, env, verbose = 1,tensorboard_log=f"runs/{run.id}", device= 'mps')
    else:
        
    model.learn(total_timesteps=config['total_timesteps'],
            callback=WandbCallback(
            gradient_save_freq=10000,
            model_save_freq=int(100000/N_ENVS),
            model_save_path=f"models/{run.id}",
            verbose=2,
        ))