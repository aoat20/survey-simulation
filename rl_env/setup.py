# rl_env/setup.py

import setuptools

setuptools.setup(
    name='rl_env',
    version='0.1',
    author='Alfie Anthony Treloar and Edward Clark',
    author_email='eprc20@bath.ac.uk',
    description='A custom RL environment package',
    license='Not yet licensed',
    packages=['rl_env'],
    install_requires=[
        'gymnasium',
        'wanb',
        'stable-baselines3',
    ],
)