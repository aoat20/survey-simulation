# import cv2
import time
from multiprocessing import Pipe, Process, Queue

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from survey_simulation import (SEASSimulation, SurveySimulation,
                               SurveySimulationGrid)

# ss = SEASSimulation()

# ss = Process(target=SEASSimulation)
# ss.start()

# print('continue')

# time.sleep(4)
# print(ss.state)

# print('continued')

# time.sleep(3.1)
# ss.set_speed_course(speed=5, course=0)

# ss = SurveySimulationGrid('manual',
#                           save_dir='data')

# ss = SurveySimulation('manual',
#                       save_dir='data',
#                       agent_start=[120,100])

# ss = SurveySimulation('playback', 
#                         save_dir='data/',
#                         ep_n=9)

if 1:
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
            ss.save_episode()
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

# ss.save_episode(2)

# comms1, comms2 = Pipe()

# kw = {'mode':'test', 
#       'save_dir': 'data', 
#       'agent_start': [120,100],
#       'child_conn': comms2}

# p = Process(target=SurveySimulation, kwargs=kw)
# time.sleep(1)
# p.start()
# while 1:
#     time.sleep(1)
#     print(comms1.recv())
#     time.sleep(1) 
#     comms1.send(['Omg heeeey'])
