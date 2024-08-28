import gymnasium as gym
import rl_env  # This will automatically register your custom environment


def run_environment(env_id, steps = None,  **kwargs):
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
    
    # Run the environment for the specified number of steps if not None otherwise run until done

    if steps is not None:
        for _ in range(steps):
            # Sample a random action from the environment's action space
            action = env.action_space.sample()
            
            # Take a step in the environment
            observation, reward, terminated, truncated, info  = env.step(action)
            
            total_reward += reward
            print (reward)
            # Print the current step's information
            # print (f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    else:
        while not done:
            # Sample a random action from the environment's action space
            action = env.action_space.sample()
            
            # Take a step in the environment
            observation, reward, terminated, truncated, info  = env.step(action)
            
            total_reward += reward
            print (reward)
            
            # Print the current step's information
            # print (f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

    print(f"Total Reward: {total_reward}")

if __name__ == "__main__":
    # Specify the environment ID and any kwargs
    environment_id = 'BasicEnv-v0'
    kwargs = {
        'params_filepath': '/Users/edwardclark/Documents/SURREY/survey-simulation/rl_env/params.txt',
        'save_logs': False,
        'obs_type': 'coverage_occupancy'
    }
    
    # Run the environment
    run_environment(environment_id, **kwargs)
