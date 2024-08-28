import rl_env  # This will automatically register your custom environment
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import SubprocVecEnv, make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from wandb.integration.sb3 import WandbCallback

import wandb

if __name__ == "__main__":

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
            'params_filepath': '/Users/edwardclark/Documents/SURREY/survey-simulation/rl_env/params.txt',
            'save_logs': False,
            'obs_type': 'coverage_occupancy'
        }
    env = make_vec_env(config['env_name'], n_envs=8, env_kwargs=env_kwargs)
    env = VecMonitor(env)


    kwargs = {
        'n_steps': 500,
        'n_epochs': 4, 
        'batch_size': 500,
    }

    policy_kwargs = {
        'activation_fn': 'tanh',
        'normalize_images':False,
    }

    model = PPO(config['policy_type'], env, verbose = 1,tensorboard_log=f"runs/{run.id}", policy_kwargs=policy_kwargs,device= 'mps',  **kwargs)

    model.learn(total_timesteps=config['total_timesteps'],
            callback=WandbCallback(
            gradient_save_freq=10000,
            model_save_freq=100000,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ))