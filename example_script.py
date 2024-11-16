import numpy as np
from survey_simulation import SurveySimulationGrid

# # Running manual mode
# ss = SurveySimulationGrid('manual')

# # # Running test mode
# # ss = SurveySimulationGrid('test',
# #                     save_dir='data')

# for n in range(500):
#     rnd_heading = np.random.randint(-70,70)

#     ss.new_action('move', rnd_heading)
#     obs = ss.next_step()

#     if ss.end_episode:
#         ss.reset()

# Running playback
# path = '/Users/edwardclark/Documents/SURREY/data/'
path = '/Users/edward/Documents/university/coding/survey-simulation/data/'
# path = '/Users/edwardclark/Documents/SURREY/survey-simulation/rl_env/model_testing/data/'


#episode 4, 15
ss = SurveySimulationGrid('playback', 
                            save_dir=path,
                            ep_n=3,
                            plotter =1,
                            agent_viz = 0,
                            )