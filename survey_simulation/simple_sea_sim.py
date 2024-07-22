import numpy as np
import os
import time
from multiprocessing import Process, Queue
import sys
from survey_simulation.sim_plotter import SEASPlotter
from survey_simulation.sim_classes import Timer, Playback, Logger, Map, Agent
import json
import pyproj

class SEASSimulation():
    """_summary_
    """

    def __init__(self):
        # Load parameters 
        mode = 'manual'
        self.playspeed = 1

        # Load the desired scene 
        agent, vessels, xy_lim = self.load_boats("SEASscenario1.json")

        # Get all the other vessels start points
        vessels_xy = [v.xy for v in vessels]

        if mode == 'manual':
            plotter = SEASPlotter(map_lims=([sum(x) for x in zip(xy_lim, [-10000, 10000, -10000, 10000])]),
                                    grid_size=(xy_lim[2]-xy_lim[0])/4,
                                    xy_start=agent.xy,
                                    xy_start_vessels = vessels_xy)

        # Set up comms queues and keywords
        q_toplotter = Queue(maxsize=1)
        q_tomain = Queue(maxsize=1)
        kw = {'q_toplotter': q_toplotter,
              'q_tomain': q_tomain,
              'agent': agent,
              'vessels': vessels}
        
        # set up process and run
        p = Process(target=self.run_mainloop, 
                    kwargs=kw,
                    daemon=True)
        p.start()
        
        if mode == 'manual':
            self.plotting_loop(plotter,
                               q_toplotter,
                               q_tomain)
        elif mode == 'test':
            
            self.test_loop(q_toplotter,
                          q_tomain)

    def plotting_loop(self, 
                      plotter,
                      q_toplotter,
                      q_tomain):
            # Controls callback
            plotter.ax.figure.canvas.mpl_connect('key_press_event',
                                                    self.on_key)
            plotter.ax.figure.canvas.mpl_connect('key_release_event', 
                                                 self.on_release)
            plotter.show(blocking=False)
            
            self.play = True
            self.turnleft = False
            self.turnright = False
            self.slowdown = False
            self.speedup = False
            self.ps_change = False

            while 1:
                if self.play:
                    tic = time.time()

                    if q_toplotter.full():
                        state = q_toplotter.get()
                        self.update_agentplots(plotter.agent,
                                               state['agent'])
                        n = 0
                        for v in state['vessels']:
                            self.update_agentplots(plotter.vessels[n],
                                                   v)
                            n += 1
                    
                    # Control handlers
                    if self.turnleft:
                        q_tomain.put({'course': state['agent'].course-5})   
                    elif self.turnright:
                        q_tomain.put({'course': state['agent'].course+5})   
                    elif self.slowdown:
                        q_tomain.put({'speed': state['agent'].speed*0.95})    
                    elif self.speedup:
                        q_tomain.put({'speed': state['agent'].speed*1.05})    
                    elif self.ps_change:
                        q_tomain.put({'playspeed':self.playspeed})
                        self.ps_change = False
                        plotter.updateps(self.playspeed)

                    # Sleep to maintain 25fps
                    time.sleep(np.clip(0.01-(time.time()-tic),0,0.01))
                else:
                    time.sleep(0.04)
                plotter.draw()

    def update_agentplots(self, plot_obj, agent):
        plot_obj.updateagent(agent.xy)
        plot_obj.updatecourse(agent.xy,
                                    agent.course)
        plot_obj.addspeedandcourse(agent.xy,
                                        agent.speed,
                                        agent.course)
        if len(agent.xy_hist)>1:
            plot_obj.updatetrackhist(agent.xy_hist)

    def test_loop(self,
                  q_toplotter,
                  q_tomain):
        while True:
            if q_toplotter.full():
                self.state = q_toplotter.get()

            
            # time.sleep(0.01)

    def run_mainloop(self, 
                     q_toplotter,
                     q_tomain,
                     agent, 
                     vessels):
        
        playspeed = 1
        while 1:
            tic = time.time()

            # receive any updated command
            if q_tomain.full():
                command = q_tomain.get()
                if 'move' in command.keys():
                    mv = command['move']
                    agent.move_to_position(mv)
                elif 'speed' in command.keys() and 'course' in command.keys():
                    agent.set_speedandcourse(command['speed'],
                                             command['course'])
                elif 'speed' in command.keys():
                    agent.set_speed(command['speed'])
                elif 'course' in command.keys():
                    agent.set_course(command['course'] % 360)
                elif 'playspeed' in command.keys():
                    playspeed = command['playspeed']

            agent.advance_one_step(0.04*playspeed)

            for v in vessels:
                v.advance_one_step(0.04*playspeed)

            # output sim state
            state = {"agent": agent,
                     "vessels": vessels}
            q_toplotter.put(state)

            time.sleep(np.clip(0.04-(time.time()-tic),0,0.04))

    def load_boats(self,
                   config_file):
        # load the boat locations and speeds from the conf file

        # Set up utm conversion
        p = pyproj.Proj(proj='utm', zone=32, ellps='WGS84')

        # Open and load the config file
        f = open(config_file)
        conf = json.load(f)

        # initialise the other vessels 
        vessels = []
        xy_lim_tmp = []
        for v in conf['vessel_details']:

            xy_lim_tmp.extend(v["waypoints"])
            # Get the vessel details
            xy_st = p(v["waypoints"][0][1],
                      v["waypoints"][0][0])
            speed = v["speed"]
            course = v["course"]
            # AI agent
            if v['vessel']=="CandidateMass":
                agent = Agent(xy_start=xy_st,
                              speed=speed,
                              course=course)
            
            elif v['vessel']=="CruiseLiner":
                vessels.append(Agent(xy_start=xy_st,
                                    speed=speed,
                                    course=course))
        f.close()

        xy_lim_tmp_np = np.array(xy_lim_tmp)
        xy_lim_utm = p(xy_lim_tmp_np[:,1],
                       xy_lim_tmp_np[:,0])

        # Get limits of travel for vessel travel
        xy_lim = [xy_lim_utm[0].min(),
                  xy_lim_utm[0].max(),
                  xy_lim_utm[1].min(),
                  xy_lim_utm[1].max()]
        return agent, vessels, xy_lim

    def set_speedandcourse(self,
                           q_tomain,
                           speed, 
                           course):
        q_tomain.put({'speed': speed, 
                           'course': course})   

    def reset(self):
        pass

    def on_key(self, event):

        # up and down key to control speed
        if event.key == "up":
            self.speedup = True
        elif event.key == "down":
            self.slowdown = True
        # left and right key to control course
        elif event.key == "left":
            self.turnleft = True
        elif event.key == "right":
            self.turnright = True
        elif event.key == "=":
            self.playspeed = self.playspeed*1.5
            self.ps_change = True
        elif event.key == "-":
            self.playspeed = self.playspeed/1.5
            self.ps_change = True
        elif event.key == " ":
            self.play = not self.play

        elif event.key == "enter":
            self.reset()

    def on_release(self, event):
        if event.key == "left":
            self.turnleft = False
        elif event.key == "right":
            self.turnright = False
        elif event.key == "up":
            self.speedup = False
        elif event.key == "down":
            self.slowdown = False

if __name__ == '__main__':
    ss = SEASSimulation()
