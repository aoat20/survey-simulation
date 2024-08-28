import gymnasium as gym
import rl-env  # This will automatically register your custom environment


def run_environment(env_id, **kwargs):
    """
    Runs a specified Gymnasium environment with given keyword arguments.

    Args:
        env_id (str): The ID of the Gymnasium environment to run.
        kwargs: Additional keyword arguments passed to the environment.

    Returns:
        None
    """
    # Create the environment with the provided kwargs
    env = gym.make(env_id, **kwargs)
    
    # Reset the environment to start
    observation = env.reset()
    
    done = False
    total_reward = 0
    
    while not done:
        # Sample a random action from the environment's action space
        action = env.action_space.sample()
        
        # Take a step in the environment
        observation, reward, terminated, truncated, info  = env.step(action)
        
        total_reward += reward
        
        # Print the current step's information
        print (f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    
    print(f"Total Reward: {total_reward}")

if __name__ == "__main__":
    # Specify the environment ID and any kwargs
    environment_id = 'BasicEnv-v0'
    kwargs = {
        'save_logs': False,
        'obs_type': 'coverage_occupancy'
    }
    
    # Run the environment
    run_environment(environment_id, **kwargs)
