import rl_env  # This will automatically register your custom environment
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from wandb.integration.sb3 import WandbCallback

import wandb

# Sweep configuration
sweep_config = {
    "method": "bayes",  # Choose from 'grid', 'random', 'bayes'
    "metric": {
        "name": "rollout/ep_rew_mean",  # Metric to optimize
        "goal": "maximize",  # Goal of the metric
    },
    "parameters": {
        "n_steps": {"values": [256, 512, 1024]},
        "n_epochs": {"values": [1, 2, 4, 8]},
        "batch_size": {"values": [256 , 512, 1024, 2048]},
        "gae_lambda": {"min": 0.8, "max": 1.0},
        "learning_rate": {"min": 1e-5, "max": 1e-2, "distribution": "log_uniform_values"},
        "clip_range": {"min": 0.1, "max": 0.4},
        "ent_coef": {"min": 0.0, "max": 0.1},
    },
}

def train():
    # Initialize the W&B run
    run = wandb.init(sync_tensorboard=True, monitor_gym=True, save_code=True)
    config = wandb.config

    N_ENVS = 2
    env_kwargs = {
        'params_filepath': '/Users/edwardclark/Documents/SURREY/survey-simulation/rl_env/params.txt',
        'save_logs': False,
        'obs_type': 'coverage_occupancy'
    }
    env = make_vec_env("BasicEnv-v0", n_envs=N_ENVS, env_kwargs=env_kwargs)
    env = VecMonitor(env)

    # Policy configuration
    policy_kwargs = {
        'activation_fn': 'tanh',
        'normalize_images': False,
    }

    # Create the PPO model
    model = PPO(
        "CnnPolicy",
        env,
        n_steps=config.n_steps,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        gae_lambda=config.gae_lambda,
        gamma=0.99,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        learning_rate=config.learning_rate,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        device='mps',
    )

    # Train the model
    model.learn(
        total_timesteps=5e5,
        callback=WandbCallback(
            gradient_save_freq=10000,
            model_save_freq=int(100000 / N_ENVS),
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )

    env.close()

if __name__ == "__main__":
    # Initialize and run the sweep
    sweep_id = wandb.sweep(sweep_config, project="surrey")  # Create a sweep
    wandb.agent(sweep_id, function=train, count=10)  # Run the sweep with 10 experiments
