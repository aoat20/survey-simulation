import numpy as np
import os
import time
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
        params, self.agent, self.vessels, xy_lim = self.load_scene(
            "SEASscenario1.json")

        # Get all the other vessels start points
        vessels_xy = [v.xy for v in self.vessels]

        if mode == 'manual' or mode == 'test':
            plotter = SEASPlotter(map_lims=([sum(x) for x in zip(xy_lim,
                                                                 [-10000,
                                                                  10000,
                                                                  -10000,
                                                                  10000])]),
                                  xy_start=self.agent.xy,
                                  xy_start_vessels=vessels_xy)

        self.plotting_loop(plotter)

    def plotting_loop(self,
                      plotter: SEASPlotter):
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

        while True:
            if self.play:
                self.next_step()
                self.update_plot(plotter)
                # Control handlers
                if self.turnleft:
                    self.agent.set_course(self.agent.course - 5)
                elif self.turnright:
                    self.agent.set_course(self.agent.course + 5)
                elif self.slowdown:
                    self.agent.set_speed(self.agent.speed*0.95)
                elif self.speedup:
                    self.agent.set_speed(self.agent.speed*1.05)
                elif self.ps_change:
                    self.ps_change = False
                    plotter.updateps(self.playspeed)
            time.sleep(0.04)
            plotter.draw()

    def next_step(self):
        self.agent.advance_one_step(0.04*self.playspeed)

        for v in self.vessels:
            v.advance_one_step(0.04*self.playspeed)

    def update_plot(self,
                    plotter: SEASPlotter):

        self.update_agentplots(plotter.agent,
                               self.agent)
        n = 0
        for v in self.vessels:
            self.update_agentplots(plotter.vessels[n],
                                   v)
            n += 1

    def update_agentplots(self,
                          plot_obj: SEASPlotter.AgentPlot,
                          agent: Agent):
        plot_obj.updateagent(agent.xy)
        plot_obj.updatecourse(agent.xy,
                              agent.course)
        plot_obj.addspeedandcourse(agent.xy,
                                   agent.speed,
                                   agent.course)
        if len(agent.xy_hist) > 1:
            plot_obj.updatetrackhist(agent.xy_hist)

    def load_scene(self,
                   config_file):
        # load the boat locations and speeds from the conf file

        # Set up utm conversion
        p = pyproj.Proj(proj='utm', zone=32, ellps='WGS84')

        # Open and load the config file
        f = open(config_file)
        conf = json.load(f)

        params = conf['params']

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
            if v['vessel'] == "CandidateMass":
                agent = Agent(xy_start=xy_st,
                              speed=speed,
                              course=course)

            elif v['vessel'] == "CruiseLiner":
                vessels.append(Agent(xy_start=xy_st,
                                     speed=speed,
                                     course=course))
        f.close()

        xy_lim_tmp_np = np.array(xy_lim_tmp)
        xy_lim_utm = p(xy_lim_tmp_np[:, 1],
                       xy_lim_tmp_np[:, 0])

        # Get limits of travel for vessel travel
        xy_lim = [xy_lim_utm[0].min(),
                  xy_lim_utm[0].max(),
                  xy_lim_utm[1].min(),
                  xy_lim_utm[1].max()]
        return params, agent, vessels, xy_lim

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
