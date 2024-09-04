import rl_env  # This will automatically register your custom environment
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import SubprocVecEnv, make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from wandb.integration.sb3 import WandbCallback

import wandb

if __name__ == "__main__":


    N_ENVS = 4

    config = {
        "policy_type": "CnnPolicy",
        "total_timesteps": 3e6,
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
    env = make_vec_env(config['env_name'], n_envs=N_ENVS, env_kwargs=env_kwargs)
    env = VecMonitor(env)


    kwargs = {
        'n_steps': 1000,
        'n_epochs': 2, 
        'batch_size': 1000,
        'gae_lambda': 0.95,
        'gamma': 0.99,
        'clip_range': 0.2,
        'ent_coef': 0,
        'learning_rate': 1e-3,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
    }

    policy_kwargs = {
        'activation_fn': 'tanh',
        'normalize_images':False,
    }

    model = PPO(config['policy_type'], env, verbose = 1,tensorboard_log=f"runs/{run.id}", policy_kwargs=policy_kwargs,device= 'mps',  **kwargs)

    model.learn(total_timesteps=config['total_timesteps'],
            callback=WandbCallback(
            gradient_save_freq=10000,
            model_save_freq=int(100000/N_ENVS),
            model_save_path=f"models/{run.id}",
            verbose=2,
        ))