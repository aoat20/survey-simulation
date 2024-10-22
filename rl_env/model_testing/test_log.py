
from rl_env.utils.reward import RewardFunction

# log_path = '/Users/edwardclark/Documents/SURREY/survey-simulation/rl_env/model_testing/data/Episode7'

log_path = '/Users/edwardclark/Documents/SURREY/data/Episode14'

# log_path ='/Users/edwardclark/Downloads/Episode4'

reward = RewardFunction(type='log_file', log_file=log_path).get_reward()
print (reward)



# #create env 
# # Define a function to find the closest action
# def find_closest_action(env, desired_angle_degrees):
    
#     diffs = env.actions - desired_angle_degrees
#     diffs = np.abs(diffs)

#     closest_action = np.argmin(diffs)
#     return int(closest_action)




    
# env_kwargs = {
#         'params_filepath': '/Users/edwardclark/Documents/SURREY/survey-simulation/rl_env/params.txt',
#         'save_logs': True,
#         'obs_type': 'coverage_occupancy'
#     }

# env = gym.make('BasicEnv-v0', **env_kwargs)

# #print the actions

# print (env.action_space)

#make sure correct for your local path
# episode = 66
# log_path = f'/Users/edwardclark/Documents/SURREY/survey-simulation/rl_env/model_testing/data/Episode{episode}/'





# #open the actions in the log episode (Episode0_ACTIONS) this format is the same for all logs so can use regex from the log folder 

# actions = []
# action_path = os.path.join(log_path, f'Episode{episode}_ACTIONS')
# with open(action_path, 'r') as f:
#     for line in f:
#         #log files during move 1 move 445.0 23.0 
#         #if move in line then strip move and calculate the angle between the actions 
#         if 'move' in line:
#             action = line.strip().split(' ')[2:]
#             actions.append(action)
#             #get closest action from the action space to angle

# actions = np.array(actions).astype(float)
# #calculate the angle between the locations at every points 
# moves = np.diff(actions, axis=0)
# angles = np.arctan2(moves[:,1], moves[:,0])
# angles = np.rad2deg(angles)
# print (angles)
# #convert to 0-360
# angles = (angles + 360) % 360
# action_list = [find_closest_action(env, angle) for angle in angles]
# print (action_list)

# obs,_ = env.reset()
# env.survey_simulation.agent.xy = actions[0]
# # print (env.survey_simulation.agent.xy)
# #move agent ot the first location in the log
# terminated = False
# while not terminated:
#     print ('location',env.survey_simulation.agent.xy)
#     action = action_list.pop(0)
#     print ('action',action)
#     print('direction',env.actions[action])
#     obs, reward, terminated, truncated, info = env.step(action)

#     # print (reward)
#     # print (terminated)
#     # print (truncated)
#     # print (info)
#     print ('---')
#     if terminated:
#         break
