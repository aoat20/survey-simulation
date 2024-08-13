from survey_simulation import SurveySimulation
from survey_simulation import SurveySimulationGrid
from survey_simulation import SEASSimulation
import numpy as np
import time
import matplotlib.pyplot as plt

# ss = SEASSimulation()

ss = SurveySimulationGrid('manual',
                          save_dir='data')

ss = SurveySimulationGrid('playback', 
                        save_dir='data/',
                        ep_n=2)

if 0:
    start = time.time() 
 
    ss = SurveySimulationGrid('test',
                        save_dir='data')
    for n in range(100):
        rnd_mv = np.random.randint(-70,70)
        ss.agent.set_speedandcourse(ss.agent.speed0,
                                    rnd_mv)
        obs = ss.next_step()
        if ss.end_episode:
            end = time.time() 
            print("Execution time: ",end - start) 
            ss.reset()
            start = time.time() 

#     t, cov_map, contacts, mp_msk = ss.new_action('move', rnd_mv)

    # print(contacts)

    # #At two arbitrary steps, demo group and ungroup actions
    # if n==45: 
    #     ss.new_action('group', [0,1])
    # if n==65:
    #     ss.new_action('ungroup', [0])
    #     ss.new_action('group', [1,2])


