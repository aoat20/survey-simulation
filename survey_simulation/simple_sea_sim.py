import numpy as np
from survey_simulation.sim_plotter import SEASPlotter
from survey_simulation.sim_classes import Timer, Agent, LoggerBMT, PlaybackBMT
import json
import pyproj
import os
from matplotlib.widgets import TextBox, Button


class SEASSimulation():
    """_summary_
    """

    def __init__(self,
                 scenario_n: int | str = "",
                 mode: str = 'manual',
                 plotter: bool = True,
                 log_dir: str = "logs/"):
        # Load parameters
        self._playspeed = 1
        self.mission_finished = False
        self.termination_reason = ""
        self.course_reached = True
        self.speed_reached = True

        if scenario_n == "":
            raise ValueError("scenario_n argument not set." +
                             "Must either be the number of the " +
                             "desired scenario or a str containing" +
                             " the path to the custom scenario json file.")
        elif isinstance(scenario_n, int):
            scen_pth = os.path.join('SEA_scenarios',
                                    f"SEASscenario{scenario_n}.json")
        elif isinstance(scenario_n, str):
            scen_pth = scenario_n

        # Load the desired scene
        if mode == 'playback':
            self._playback = PlaybackBMT(scen_pth)
            self._agent, self._vessels = self._playback.get_vessels(n=0)
            t_tmp, _, _ = self._playback.get_time_req(0)
            xy_lim = self._playback.get_xy_lims()
            self._timer = Timer()
            self._timer.update_time(t_tmp)
            plotter = True
            self.n = 0
        else:
            params, self._agent, self._vessels, xy_lim = self._load_scene(
                scen_pth)
            self._logger = LoggerBMT(save_dir=log_dir)
            self._timer = Timer(time_lim=params['t_max'],
                                t_step=params["t_step"])

        if plotter:
            self._plotter_obj = SEASPlotter(map_lims=([sum(x)
                                                       for x in zip(xy_lim,
                                                                    [-10000,
                                                                     10000,
                                                                     -10000,
                                                                     10000])]),
                                            agent=self._agent,
                                            vessels=self._vessels)
            self._plotter_obj.show(blocking=False)

            if mode == 'manual':
                pos_tmp = self._plotter_obj.ax.get_position()
                self._plotter_obj.ax.set_position([pos_tmp.x0,
                                                  pos_tmp.y0,
                                                  pos_tmp.width,
                                                  pos_tmp.height*0.9])

                self._add_controls()
                self._plotting_loop()

            elif mode == 'playback':
                pos_tmp = self._plotter_obj.ax.get_position()
                self._plotter_obj.ax.set_position([pos_tmp.x0,
                                                  pos_tmp.y0,
                                                  pos_tmp.width,
                                                  pos_tmp.height*0.9])

                self._add_controls()
                self._plotting_loop_playback()

    def get_obs(self):
        obs_dict = {}
        obs_dict['time_s'] = self._timer.time_elapsed
        obs_dict['next_waypoint'] = self._agent.waypoints[self._agent.waypoint_n]
        obs_dict['agent'] = {"speed_kn": self._agent.speed*1.944,
                             "course": self._agent.course,
                             "coords_utm": self._agent.xy}
        v: Agent
        for v in self._vessels:
            obs_dict[v.vessel_type] = {"speed_kn": v.speed*1.944,
                                       "course": v.course,
                                       "coords_utm": v.xy,
                                       "range_yds": v.range_yds,
                                       "cpa_yds": v.cpa_yds,
                                       "tcpa_s": v.tcpa_s}

        return obs_dict

    def set_course(self,
                   course):
        self.course_reached = False
        self._change_course(course)
        self._logger.add_course_req(course)

    def set_speed(self,
                  speed_kn):
        self.speed_reached = False
        self._change_speed(speed_kn)
        self._logger.add_speed_req(speed_kn)

    def _check_failure_conditions(self):
        # Check if agent has reached the final waypoint
        wp_d_yds = self._get_distance(self._agent.xy,
                                      self._agent.waypoints[-1])
        if wp_d_yds < 1000:
            self.mission_finished = True
            self.termination_reason = f"SUCCESS. Reached final waypoint"

        # Check distance to each other vessel
        v: Agent
        for v in self._vessels:
            if v.range_yds < 1000:
                self.mission_finished = True
                self.termination_reason = f"FAILED. Got too close to {v.vessel_type}"

        # Check whether time has run out
        if self._timer.time_remaining < 0:
            self.mission_finished = True
            self.termination_reason = "FAILED. Time ran out"

        # Check when/where action is taken

        # Check

        if self.mission_finished:
            # Save log file
            if hasattr(self, '_logger'):
                self._logger.save_log_file()

    def _get_distance(self, xy1, xy2):
        d_m = np.sqrt((xy1[0]-xy2[0])**2 + (xy1[1]-xy2[1])**2)
        return d_m

    def _compute_cpa(self,
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
        cpa_yds = self._m_to_yds(cpa)
        tcpa_s = - (dv_x*dx + dv_y*dy)/(dv_x**2 + dv_y**2)
        if tcpa_s < 0:
            tcpa_s = 0
        return cpa_yds, tcpa_s

    def _m_to_yds(self,
                  m):
        yds = m*1.09361
        return yds

    def _add_controls(self):
        # Controls callback
        self._plotter_obj.ax.figure.canvas.mpl_connect('key_press_event',
                                                       self._on_key)
        # Course
        course_ax = self._plotter_obj.fig.add_axes([0.25, 0.9, 0.1, 0.075])
        self._course_req_box = TextBox(course_ax,
                                       "Course change (deg)",
                                       textalignment="left")
        self._course_req_box.set_val(f"{self._agent.course:.1f}")
        self._course_req_box.on_submit(self._change_course)
        ax_c_pl1 = self._plotter_obj.fig.add_axes([0.35, 0.9375, 0.05, 0.0375])
        self._b_c_pl1 = Button(ax_c_pl1, '+1')
        self._b_c_pl1.on_clicked(self._course_add_one)
        ax_c_pl2 = self._plotter_obj.fig.add_axes([0.35, 0.9, 0.05, 0.0375])
        self._b_c_pl2 = Button(ax_c_pl2, '-1')
        self._b_c_pl2.on_clicked(self._course_minus_one)

        # Speed
        speed_ax = self._plotter_obj.fig.add_axes([0.7, 0.9, 0.1, 0.075])
        self._speed_req_box = TextBox(speed_ax,
                                      "Speed change (kn)",
                                      textalignment="left")
        ag_sp_kn = self._agent.speed*1.944
        self._speed_req_box.set_val(f"{ag_sp_kn:.1f}")
        self._speed_req_box.on_submit(self._change_speed)
        ax_s_pl1 = self._plotter_obj.fig.add_axes([0.8, 0.9375, 0.05, 0.0375])
        self._b_s_pl1 = Button(ax_s_pl1, '+1')
        self._b_s_pl1.on_clicked(self._speed_add_one)
        ax_s_pl2 = self._plotter_obj.fig.add_axes([0.8, 0.9, 0.05, 0.0375])
        self._b_s_pl2 = Button(ax_s_pl2, '-1')
        self._b_s_pl2.on_clicked(self._speed_minus_one)

        # Play speed
        ax_playspeed1 = self._plotter_obj.fig.add_axes(
            [0.65, 0.82, 0.05, 0.05])
        self._playspeed_button1 = Button(ax_playspeed1, 'RT')
        self._playspeed_button1.on_clicked(self._playspeed_rt)
        ax_playspeed2 = self._plotter_obj.fig.add_axes([0.7, 0.82, 0.05, 0.05])
        self._playspeed_button2 = Button(ax_playspeed2, 'x2')
        self._playspeed_button2.on_clicked(self._playspeed_x2)
        ax_playspeed3 = self._plotter_obj.fig.add_axes(
            [0.75, 0.82, 0.05, 0.05])
        self._playspeed_button3 = Button(ax_playspeed3, 'x10')
        self._playspeed_button3.on_clicked(self._playspeed_x10)

    def _plotting_loop(self):
        self._play = True
        self._ps_change = False

        while True:
            if self._play and not self.mission_finished:
                self._next_step_manual()
                # Control handlers
                if self._ps_change:
                    self._ps_change = False
                    self._plotter_obj.updateps(self._playspeed)
                if self.mission_finished:
                    # self._plotter_obj.explode(self._agent.xy)
                    self._plotter_obj.ax.set_title(self.termination_reason)

            self._plotter_obj.pause(1/25)

    def _plotting_loop_playback(self):
        self._play = True
        self._ps_change = False

        while True:
            if self._play:
                self._next_step_playback()
                # Control handlers
                if self._ps_change:
                    self._ps_change = False
                    self._plotter_obj.updateps(self._playspeed)
                if self.mission_finished:
                    self._plotter_obj.ax.set_title(self.termination_reason)

            self._plotter_obj.pause(1/self._playspeed)

    def next_step(self):
        self._adv_time(self._timer.t_step)

    def _next_step_manual(self):
        t = self._timer.t_step*self._playspeed/25
        self._adv_time(t)

    def _next_step_playback(self):
        t, speed_req, course_req = self._playback.get_time_req(self.n)
        self._agent, self._vessels = self._playback.get_vessels(self.n)

        self._timer.update_time(t)

        # Update all other vessels
        v: Agent
        for v in self._vessels:
            v.cpa_yds, v.tcpa_s = self._compute_cpa(self._agent.xy,
                                                    self._agent.course,
                                                    self._agent.speed,
                                                    v.xy,
                                                    v.course,
                                                    v.speed)
            v.range_yds = self._get_distance(self._agent.xy,
                                             v.xy)

        # Update the plotter
        self._update_plot()
        self.n += 1

    def _adv_time(self, t):
        if not self.mission_finished:
            self._timer.update_time(t)
            self._agent.advance_one_step(t)
            # check if course and speed changes have been reached
            if self._agent.course_req == self._agent.course:
                self.course_reached = True
            if self._agent.speed_req == self._agent.speed:
                self.speed_reached = True

            # Update all other vessels
            v: Agent
            for v in self._vessels:
                v.advance_one_step(t)
                v.cpa_yds, v.tcpa_s = self._compute_cpa(self._agent.xy,
                                                        self._agent.course,
                                                        self._agent.speed,
                                                        v.xy,
                                                        v.course,
                                                        v.speed)
                v.range_yds = self._get_distance(self._agent.xy,
                                                 v.xy)
            # Check for failure
            self._check_failure_conditions()

            # Update the plotter
            if hasattr(self, '_plotter_obj'):
                self._update_plot()
                self._plotter_obj.pause(0.0001)

            # Update the logger
            if hasattr(self, '_logger'):
                self._logger.next_step()
                self._logger.add_time(self._timer.time_elapsed)
                self._logger.log_vessel(self._agent)
                for v in self._vessels:
                    self._logger.log_vessel(v)
                self._logger.add_time(self._timer.time_elapsed)

    def _update_plot(self):
        self._plotter_obj.update_minutecounter(self._timer.time_elapsed)
        self._update_agentplots(self._plotter_obj.agent,
                                self._agent)
        n = 0
        for v in self._vessels:
            self._update_agentplots(self._plotter_obj.vessels[n],
                                    v)
            n += 1

    def _update_agentplots(self,
                           plot_obj: SEASPlotter.AgentPlot,
                           agent: Agent):

        plot_obj.updateagent(agent.xy)
        plot_obj.updatecourse(agent.xy,
                              agent.course)
        plot_obj.update_label(agent.xy,
                              agent.speed*1.944,
                              agent.course,
                              agent.cpa_yds,
                              agent.tcpa_s,
                              agent.range_yds)
        if len(agent.xy_hist) > 1:
            plot_obj.updatetrackhist(agent.xy_hist)

    def _load_scene(self,
                    config_file):
        # load the boat locations and speeds from the conf file

        # Set up utm conversion
        p = pyproj.Proj(proj='utm',
                        zone=30,
                        ellps='WGS84',
                        preserve_units=False)

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
                way_points.append(p(self._convert_dms_to_dec(wp[1]),
                                    self._convert_dms_to_dec(wp[0])))
            xy_lim_tmp.extend(way_points)

            speed_mps = 0.5144*v["speed_kn"]
            course = np.rad2deg(np.arctan2(way_points[1][0]-way_points[0][0],
                                           way_points[1][1]-way_points[0][1]))
            # AI agent
            if v['vessel'] == "agent":
                agent = Agent(xy_start=way_points[0],
                              speed=speed_mps,
                              course=course,
                              vessel_type='agent',
                              waypoints=way_points)
                agent.course_change_rate = v["turning_rate"]
                agent.speed_change_rate = v["speed_change_rate"]
                # agent.speed_max = v['max_speed_kn']*0.5144
            else:
                vessels.append(Agent(xy_start=way_points[0],
                                     speed=speed_mps,
                                     course=course,
                                     vessel_type=v['vessel'],
                                     waypoints=way_points))
        f.close()

        xy_lim_tmp_np = np.array(xy_lim_tmp)
        # Get limits of travel for vessel travel
        xy_lim = [xy_lim_tmp_np[:, 0].min(),
                  xy_lim_tmp_np[:, 0].max(),
                  xy_lim_tmp_np[:, 1].min(),
                  xy_lim_tmp_np[:, 1].max()]
        return params, agent, vessels, xy_lim

    def _convert_dms_to_dec(self, tude):
        multiplier = 1 if tude[-1] in ['N', 'E'] else -1
        coord_dec = multiplier * \
            sum(float(x) / 60 ** n for n, x in enumerate(tude[:-1].split('-')))
        return coord_dec

    def _reset(self):
        if hasattr(self, '_logger'):
            self._logger.save_log_file()

    def _on_close(self, event):
        exit(1)

    def _change_course(self,
                       new_course):
        if float(new_course) != self._agent.course:
            self._course_req = float(new_course)
            self._agent.course_req = self._course_req

    def _change_speed(self,
                      new_speed_kn):
        if abs(float(new_speed_kn) - self._agent.speed*1.944) > 0.1:
            speed_mps_req = float(new_speed_kn)*0.5144
            # if self._agent.speed_max > self._agent.speed_req:
            self._agent.speed_req = speed_mps_req

    def _update_speed_box(self, change):
        sp_new = (self._agent.speed_req*1.944)+change
        self._speed_req_box.set_val(f"{sp_new:.1f}")

    def _speed_add_one(self, event):
        self._update_speed_box(1)

    def _speed_minus_one(self, event):
        self._update_speed_box(-1)

    def _update_course_box(self, change):
        course_new = self._agent.course_req+change
        self._course_req_box.set_val(f"{course_new:.1f}")

    def _course_add_one(self, event):
        self._update_course_box(1.)

    def _course_minus_one(self, event):
        self._update_course_box(-1.)

    def _playspeed_rt(self, event):
        self._playspeed = 1
        self._ps_change = True

    def _playspeed_x2(self, event):
        self._playspeed = self._playspeed*2
        self._ps_change = True

    def _playspeed_x10(self, event):
        self._playspeed = self._playspeed*10
        self._ps_change = True

    def _on_key(self, event):

        if event.key == "up":
            self._playspeed = self._playspeed*1.5
            self._ps_change = True
        elif event.key == "down":
            self._playspeed = self._playspeed/1.5
            self._ps_change = True
        elif event.key == " ":
            self._play = not self._play
        elif event.key == "enter":
            self._reset()


if __name__ == '__main__':
    ss = SEASSimulation()
