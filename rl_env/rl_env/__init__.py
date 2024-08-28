# rl_env/__init__.py

from gymnasium.envs.registration import register

register(
    id='BasicEnv-v0',
    entry_point='rl_env.envs:BasicEnv',
)



# Optional: Import your environment here for easy access
from rl_env.envs import BasicEnv
