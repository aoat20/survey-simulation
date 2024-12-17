import rl_env  # This will automatically register your custom environment
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import SubprocVecEnv, make_vec_env

LEARN_STEPS = 10000


env_kwargs = {
        'params_filepath': 'rl_env/params.txt',
        'save_logs': False,
        'obs_type': 'coverage_occupancy'
    }
env = make_vec_env('BasicEnv-v0', n_envs=4, seed=0, env_kwargs=env_kwargs)



kwargs = {
    'n_steps': 500,
    'n_epochs': 2, 
}

policy_kwargs = {
    'activation_fn': 'tanh',
    'normalize_images':False,
}

model = PPO("CnnPolicy", env, verbose = 1, policy_kwargs=policy_kwargs, **kwargs)




model.learn(total_timesteps=LEARN_STEPS)