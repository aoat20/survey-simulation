import numpy as np
import os
import time
import sys
from survey_simulation.sim_plotter import SEASPlotter
from survey_simulation.sim_classes import Timer, Playback, Logger, Map, Agent
import json
import pyproj
from matplotlib.widgets import TextBox, Button


class SEASSimulation():
    """_summary_
    """

    def __init__(self,
                 scenario_n=1,
                 mode='manual',
                 plotter=True):
        # Load parameters
        self.playspeed = 1

        # Load the desired scene
        params, self.agent, self.vessels, xy_lim = self.load_scene(
            f"SEASscenario{scenario_n}.json")

        self.timer = Timer(time_lim=60*60,
                           t_step=0.1)

        if plotter:
            plotter_obj = SEASPlotter(map_lims=([sum(x) for x in zip(xy_lim,
                                                                     [-10000,
                                                                      10000,
                                                                      -10000,
                                                                      10000])]),
                                      agent=self.agent,
                                      vessels=self.vessels)
        plotter_obj.add_minutecounter()
        pos_tmp = plotter_obj.ax.get_position()
        plotter_obj.ax.set_position([pos_tmp.x0,
                                     pos_tmp.y0,
                                     pos_tmp.width,
                                     pos_tmp.height*0.9])

        self.add_controls(plotter_obj)
        self.plotting_loop(plotter_obj)

    def check_failure_conditions(self):
        # Check distance to each other vessel
        for v in self.vessels:
            if self.m_to_yds(self.check_distances(self.agent.xy,
                                                  v.xy)) < 1000:
                print('you lose')

    def check_distances(self, xy1, xy2):
        d_m = np.sqrt((xy1[0]-xy2[0])**2 + (xy1[1]-xy2[1])**2)
        return d_m

    def compute_cpa(self,
                    xy1, course1, speed1,
                    xy2, course2, speed2):

        # Compute differences
        dv_x = speed1*np.sin(np.deg2rad(course1)) - \
            speed2*np.sin(np.deg2rad(course2))
        dv_y = speed1*np.cos(np.deg2rad(course1)) - \
            speed2*np.cos(np.deg2rad(course2))
        dx = xy1[0] - xy2[0]
        dy = xy1[1] - xy2[1]
        # CPA and TCPA
        cpa = np.abs(dv_y*dx - dv_x*dy)/np.sqrt(dv_x**2 + dv_y**2)
        cpa_yds = self.m_to_yds(cpa)
        tcpa_s = - (dv_x*dx + dv_y*dy)/(dv_x**2 + dv_y**2)
        if tcpa_s < 0:
            tcpa_s = 0
        return cpa_yds, tcpa_s

    def m_to_yds(self,
                 m):
        yds = m*1.09361
        return yds

    def add_controls(self,
                     plotter: SEASPlotter):
        # Controls callback
        plotter.ax.figure.canvas.mpl_connect('key_press_event',
                                             self.on_key)
        # Course
        course_ax = plotter.fig.add_axes([0.25, 0.9, 0.1, 0.075])
        self.course_req_box = TextBox(course_ax,
                                      "Course change (deg)",
                                      textalignment="left")
        self.course_req_box.set_val(f"{self.agent.course:.0f}")
        self.course_req_box.on_submit(self.change_course)
        ax_c_pl1 = plotter.fig.add_axes([0.35, 0.9375, 0.05, 0.0375])
        self.b_c_pl1 = Button(ax_c_pl1, '+1')
        self.b_c_pl1.on_clicked(self.course_add_one)
        ax_c_pl2 = plotter.fig.add_axes([0.35, 0.9, 0.05, 0.0375])
        self.b_c_pl2 = Button(ax_c_pl2, '-1')
        self.b_c_pl2.on_clicked(self.course_minus_one)

        # Speed
        speed_ax = plotter.fig.add_axes([0.7, 0.9, 0.1, 0.075])
        self.speed_req_box = TextBox(speed_ax,
                                     "Speed change (kn)",
                                     textalignment="left")
        ag_sp_kn = self.agent.speed*1.944
        self.speed_req_box.set_val(f"{ag_sp_kn:.1f}")
        self.speed_req_box.on_submit(self.change_speed)
        ax_s_pl1 = plotter.fig.add_axes([0.8, 0.9375, 0.05, 0.0375])
        self.b_s_pl1 = Button(ax_s_pl1, '+1')
        self.b_s_pl1.on_clicked(self.speed_add_one)
        ax_s_pl2 = plotter.fig.add_axes([0.8, 0.9, 0.05, 0.0375])
        self.b_s_pl2 = Button(ax_s_pl2, '-1')
        self.b_s_pl2.on_clicked(self.speed_minus_one)

        # Play speed
        ax_playspeed1 = plotter.fig.add_axes([0.65, 0.82, 0.05, 0.05])
        self.playspeed_button1 = Button(ax_playspeed1, 'RT')
        self.playspeed_button1.on_clicked(self.playspeed_rt)
        ax_playspeed2 = plotter.fig.add_axes([0.7, 0.82, 0.05, 0.05])
        self.playspeed_button2 = Button(ax_playspeed2, 'x2')
        self.playspeed_button2.on_clicked(self.playspeed_x2)
        ax_playspeed3 = plotter.fig.add_axes([0.75, 0.82, 0.05, 0.05])
        self.playspeed_button3 = Button(ax_playspeed3, 'x10')
        self.playspeed_button3.on_clicked(self.playspeed_x10)

    def plotting_loop(self,
                      plotter_obj: SEASPlotter):
        plotter_obj.show(blocking=False)

        self.play = True
        self.ps_change = False

        while True:
            if self.play:
                self.next_step()
                self.update_plot(plotter_obj)
                # Control handlers
                if self.ps_change:
                    self.ps_change = False
                    plotter_obj.updateps(self.playspeed)
            plotter_obj.pause(self.timer.t_step/self.playspeed)
            plotter_obj.draw()

    def next_step(self):
        self.timer.update_time(self.timer.t_step*self.playspeed)
        self.agent.advance_one_step(self.timer.t_step*self.playspeed)
        for v in self.vessels:
            v.advance_one_step(self.timer.t_step*self.playspeed)
            v.cpa_yds, v.tcpa_s = self.compute_cpa(self.agent.xy,
                                                   self.agent.course,
                                                   self.agent.speed,
                                                   v.xy,
                                                   v.course,
                                                   v.speed)
        self.check_failure_conditions()

    def update_plot(self,
                    plotter_obj: SEASPlotter):
        plotter_obj.update_minutecounter(self.timer.time_elapsed)
        self.update_agentplots(plotter_obj.agent,
                               self.agent)
        n = 0
        for v in self.vessels:
            self.update_agentplots(plotter_obj.vessels[n],
                                   v)
            n += 1

    def update_agentplots(self,
                          plot_obj: SEASPlotter.AgentPlot,
                          agent: Agent):

        plot_obj.updateagent(agent.xy)
        plot_obj.updatecourse(agent.xy,
                              agent.course)
        plot_obj.update_label(agent.xy,
                              agent.speed*1.944,
                              agent.course,
                              agent.cpa_yds,
                              agent.tcpa_s)
        if len(agent.xy_hist) > 1:
            plot_obj.updatetrackhist(agent.xy_hist)

    def load_scene(self,
                   config_file):
        # load the boat locations and speeds from the conf file

        # Set up utm conversion
        p = pyproj.Proj(proj='utm',
                        zone=30,
                        ellps='WGS84',
                        preserve_units=False)

        grid_conv = 1.5

        # Open and load the config file
        f = open(config_file)
        conf = json.load(f)

        params = conf['params']

        # initialise the other vessels
        vessels = []
        xy_lim_tmp = []
        for v in conf['vessel_details']:

            # Get the vessel details
            way_points = []
            for wp in v["waypoints"]:
                way_points.append(p(self.convert_dms_to_dec(wp[1]),
                                    self.convert_dms_to_dec(wp[0])))
            xy_lim_tmp.extend(way_points)

            speed_mps = 0.5144*v["speed"]
            course = v["course"]
            # AI agent
            if v['vessel'] == "CandidateMass":
                agent = Agent(xy_start=way_points[0],
                              speed=speed_mps,
                              course=course+grid_conv,
                              vessel_type='agent',
                              waypoints=way_points)
                agent.course_change_rate = v["turning_rate"]
                agent.speed_change_rate = v["speed_change_rate"]
                # agent.speed_max = v['max_speed']*0.5144
            elif v['vessel'] == "CruiseLiner":
                vessels.append(Agent(xy_start=way_points[0],
                                     speed=speed_mps,
                                     course=course+grid_conv,
                                     vessel_type='cruise liner',
                                     waypoints=way_points))
        f.close()

        xy_lim_tmp_np = np.array(xy_lim_tmp)
        print(xy_lim_tmp_np[0, :])
        # Get limits of travel for vessel travel
        xy_lim = [xy_lim_tmp_np[:, 0].min(),
                  xy_lim_tmp_np[:, 0].max(),
                  xy_lim_tmp_np[:, 1].min(),
                  xy_lim_tmp_np[:, 1].max()]
        return params, agent, vessels, xy_lim

    def convert_dms_to_dec(self, tude):
        multiplier = 1 if tude[-1] in ['N', 'E'] else -1
        coord_dec = multiplier * \
            sum(float(x) / 60 ** n for n, x in enumerate(tude[:-1].split('-')))
        return coord_dec

    def reset(self):
        pass

    def change_course(self,
                      new_course):
        if float(new_course) != self.agent.course:
            self.course_req = float(new_course)
            self.agent.course_req = self.course_req

    def change_speed(self,
                     new_speed_kn):
        if abs(float(new_speed_kn) - self.agent.speed*1.944) > 0.1:
            speed_mps_req = float(new_speed_kn)*0.5144
            # if self.agent.speed_max > self.agent.speed_req:
            self.agent.speed_req = speed_mps_req

    def update_speed_box(self, change):
        sp_new = (self.agent.speed_req*1.944)+change
        self.speed_req_box.set_val(f"{sp_new:.1f}")

    def speed_add_one(self, event):
        self.update_speed_box(1)

    def speed_minus_one(self, event):
        self.update_speed_box(-1)

    def update_course_box(self, change):
        course_new = self.agent.course_req+change
        self.course_req_box.set_val(f"{course_new:.0f}")

    def course_add_one(self, event):
        self.update_course_box(1)

    def course_minus_one(self, event):
        self.update_course_box(-1)

    def playspeed_rt(self, event):
        self.playspeed = 1
        self.ps_change = True

    def playspeed_x2(self, event):
        self.playspeed = self.playspeed*2
        self.ps_change = True

    def playspeed_x10(self, event):
        self.playspeed = self.playspeed*10
        self.ps_change = True

    def on_key(self, event):

        if event.key == "up":
            self.playspeed = self.playspeed*1.5
            self.ps_change = True
        elif event.key == "down":
            self.playspeed = self.playspeed/1.5
            self.ps_change = True
        elif event.key == " ":
            self.play = not self.play

        elif event.key == "enter":
            self.reset()


if __name__ == '__main__':
    ss = SEASSimulation()
