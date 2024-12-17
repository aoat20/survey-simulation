import numpy as np
import os
import time
from survey_simulation.sim_plotter import SurveyPlotter, AgentViz
from survey_simulation.survey_classes import ContactDetections, CoverageMap
from survey_simulation.sim_classes import Timer, Playback, Logger, Map, Agent, GriddedData
from survey_simulation.reward import RewardFunction
import matplotlib.pyplot as plt


class SurveySimulationGrid():
    """_summary_
    """

    def __init__(self,
                 mode='manual',
                 params_filepath='params.txt',
                 save_dir='data',
                 ep_n=0,
                 log_file='',
                 **kwargs):

        # Check whether mode of operation is valid
        modes = ['manual',
                 'test',
                 'playback']
        if mode not in modes:
            raise ValueError("Invalid mode. Expected one of: %s" % modes)

        # initiate flags
        self.end_episode = False
        self.play = False
        self.action_id = 0

        if mode == 'manual':
            self.groupswitch = True
            self.snaptoangle = False
        if mode == 'playback':
            self.action_id_prev = 0

        # Load parameters from file
        if mode == "manual" or mode == "test":
            params = self.load_params(params_filepath)
        elif mode == "playback":
            if log_file != '':
                ep_dir = log_file
            else:
                ep_dir = os.path.join(save_dir, "Episode"+str(ep_n))
            print('Loading log file: '+ep_dir)
            l_dir = os.listdir(ep_dir)
            p_path = [x for x in l_dir if 'META' in x][0]
            map_path = [x for x in l_dir if 'MAP' in x][0]
            params = self.load_params(os.path.join(ep_dir, p_path))

        # Modify parameters from arguments
        fta = True
        for key, value in kwargs.items():
            if fta:
                print('Overwriting the following parameters:')
                fta = False
            print("%s = %s" % (key, value))
            # check if it's a map parameter and if so remove other map parameters
            if 'map' in key:
                params = {key: val for key, val in params.items()
                          if not 'map' in key}
            # check if it's the start position and remove others
            if 'start' in key:
                params = {key: val for key, val in params.items()
                          if not 'start' in key}
                # Add parameter to dictionary
            params[key] = value

        self.msa = params['min_scan_angle_diff']

        # Set random seed if required for debugging purposes
        if mode == "manual" or mode == 'test':
            if 'rand_seed' in params.keys():
                np.random.seed(params['rand_seed'])

        ################ instantiate all the class objects ###############

        # Set up the map
        # if a default map is needed
        # else get the map from the location map_path
        # else make a random map
        # else just make an empty space to move around in
        if mode == 'playback':
            self.map_obj = Map(map_path=os.path.join(ep_dir, map_path))
        else:
            if 'map_n' in params:
                self.map_obj = Map(map_n=params['map_n'])
            elif 'map_path' in params:
                self.map_obj = Map(map_path=params['map_path'])
            elif 'map_random' in params:
                print('Random map being generated')
                self.map_obj = Map(random_map=True)
            else:
                try:
                    self.map_obj = Map(map_lims=params['map_area_lims'])
                except:
                    raise Exception('Need to include some map specification' +
                                    ' (map_n, map_path, map_random or map_area_lims )')

        if mode == "playback":
            self.playback = Playback(ep_dir)

        # Set the agent start position
        if mode == 'playback':
            d_temp = self.playback.get_data(0)
            agent_start = d_temp[1]
        elif params.get('random_start') == 1:
            # Random start anywhere
            self.map_obj.ag_st_mode = 2
        elif params.get('random_start') == 2:
            # Random start edges
            self.map_obj.ag_st_mode = 3
        else:
            # use specified start if available, else default for map
            if 'agent_start' in params:
                self.map_obj.ag_st_mode = 1
                self.map_obj.agent_st = params['agent_start']
            else:
                self.map_obj.ag_st_mode = 0

        if mode != 'playback':
            agent_start = self.map_obj.get_agent_start()

        # Get t step and grid spacing
        self.agent = Agent(xy_start=agent_start,
                           speed=params['agent_speed'],
                           scan_thr=params['scan_thr'])
        # Check if agent start position is in the map bounds
        if self.map_obj.is_occupied(self.agent.xy):
            raise Exception("Agent position is not valid")
        self.contacts = ContactDetections(loc_uncertainty=params['loc_uncertainty'],
                                          n_targets=params['n_targets'],
                                          det_probs=params['det_probs'],
                                          clutter_density=params['clutter_dens'],
                                          det_probs_clutter=params['det_probs_clutter'],
                                          clutter_ori_mean=params['clutter_or_mean'],
                                          clutter_ori_std=params['clutter_or_std'])
        self.covmap = CoverageMap(map_lims=self.map_obj.map_lims,
                                  leadinleadout=params['leadinleadout'],
                                  min_scan_l=params['min_scan_l'],
                                  scan_width=params['scan_width'],
                                  nadir_width=params['nadir_width'])

        self.griddata = GriddedData(map_lims=self.map_obj.map_lims,
                                    angle_diff=params['min_scan_angle_diff'],
                                    occ_map=self.map_obj.occ,
                                    agent_xy=self.agent.xy)
        self.timer = Timer(params['time_lim'],
                           params['t_step'])
        self.reward = RewardFunction()

        if mode == 'playback':
            self.run_to_end()

        if params.get('agent_viz'):
            self.agent_viz = AgentViz(map_dims=self.map_obj.map_lims,
                                      occ_map=self.map_obj.occ)

        if mode == "manual" or mode == 'test':
            # Generate contact locations
            self.contacts.generate_targets(self.map_obj.occ)
            # instantiate extra objects
            self.logger = Logger(agent_start=self.agent.xy,
                                 time_lim=params['time_lim'],
                                 map_lims=self.map_obj.map_lims,
                                 params=params,
                                 gnd_trth=self.contacts.truth,
                                 save_dir=save_dir,
                                 map_img=self.map_obj.img)

        if mode == "manual" or params['plotter']:
            self.plotter = SurveyPlotter(map_lims=self.map_obj.map_lims,
                                         map_img=self.map_obj.img,
                                         xy0=self.agent.xy,
                                         time_lim=params['time_lim'],
                                         leadinleadout=self.covmap.leadinleadout,
                                         min_scan_l=self.covmap.min_scan_l,
                                         scan_width=self.covmap.scan_width,
                                         min_scan_ang=params['min_scan_angle_diff'],
                                         n_angles=params['N_angles'],
                                         n_looks=params['N_looks'])
            self.plotter.show(blocking=False)
            self.plotter.draw()

        # Set up event handlers
        if mode == "manual":
            # mouse and key event handlers
            self.plotter.ax.figure.canvas.mpl_connect('button_press_event',
                                                      self.on_click)
            self.plotter.ax.figure.canvas.mpl_connect('motion_notify_event',
                                                      self.mouse_move)
            self.plotter.ax.figure.canvas.mpl_connect('pick_event',
                                                      self.on_pick)
            self.plotter.ax.figure.canvas.mpl_connect('key_press_event',
                                                      self.on_key_manual)
        elif mode == 'playback' and hasattr(self, 'plotter'):
            self.plotter.ax.figure.canvas.mpl_connect('key_press_event',
                                                      self.on_key_playback)

        # Set the simulator running
        if mode == "manual":
            self.plotting_loop()
        elif mode == "playback" and hasattr(self, 'plotter'):
            self.plotting_loop_pb()

    def plotting_loop(self):
        while True:
            tic = time.time()
            if self.agent.speed != 0:
                self.next_step()
            t_el = np.clip(time.time() - tic, 0, 0.04)
            self.plotter.pause(0.0401 - t_el)

    def plotting_loop_pb(self):
        self.plotter.draw()
        while True:
            if self.play:
                if self.action_id >= self.playback.ep_end:
                    self.plotter.pause(0.1)
                else:
                    self.action_id += 1
                    # self.plotter.pause(0.04)
            if self.action_id_prev != self.action_id:
                self.next_step_pb()

            self.plotter.pause(0.0001)

    def playback_step(self):
        if self.action_id >= self.playback.ep_end:
            return 0
        else:
            self.action_id += 1
            self.next_step_pb()
            return 1

    def get_reward(self):
        # Get all the observations for the rewards function
        obs_dict = {'cov_map': self.griddata.cov_map,
                    'path_length': self.agent.get_current_path_len()}
        return self.reward.get_reward(obs_dict=obs_dict)

    def run_to_end(self):
        running = True
        while running:
            running = self.playback_step()
            self.get_reward()

        self.run_to_start()

    def run_to_start(self):
        while self.action_id > 0:
            self.action_id -= 1
            self.next_step_pb()
        self.griddata.reset()

    def load_params(self,
                    param_file: str):
        param_dict = {}
        with open(param_file) as fh:
            for line in fh:
                # handle empty lines and commented
                if line.strip() and not line[0] == '#':
                    # split keys and values
                    key, value = [x.strip()
                                  for x in line.strip().split(':', 1)]
                    # check whether array
                    if "(" in value or "[" in value:
                        value = value.replace("(", "").replace(")", "")
                        value = value.replace("[", "").replace("]", "")
                        # check whether float
                        if "." in value:
                            param_dict[key] = [float(item)
                                               for item in value.split(',')]
                        else:
                            param_dict[key] = [int(item)
                                               for item in value.split(',')]
                    elif '"' in value:
                        param_dict[key] = value[1:-1]
                    else:
                        if "." in value:
                            param_dict[key] = float(value)
                        else:
                            param_dict[key] = int(value)

        return param_dict

    def check_termination(self):
        # Termination conditions
        # Timer runs out
        if self.timer.time_remaining < 0:
            self.terminate_episode()
            self.termination_reason = "Run out of time"
        # Hits land
        if self.map_obj.is_occupied(self.agent.xy):
            self.terminate_episode()
            self.termination_reason = "Grounded"
        # Returns home

    def new_action(self,
                   action_type: str,
                   action):
        # for 'move', action = course
        # for 'group', action = c_inds
        # for 'ungroup' action = g_ind
        action_types = ['move',
                        'group',
                        'ungroup']

        # Check action type
        if action_type not in action_types:
            raise ValueError("Invalid action type. " +
                             "Expected one of: %s" % action_types)

        # do action
        if action_type == 'move':
            # check move is valid
            if isinstance(action, int) or isinstance(action, float):
                self.agent.set_speedandcourse(self.agent.speed0,
                                              action)
            else:
                raise ValueError("Invalid move action. Should be "
                                 "a float/int representing the course.")

        elif action_type == 'group':
            self.add_group(action)
        elif action_type == 'ungroup':
            self.remove_group(action)

    def next_step(self):
        success = False
        if not self.end_episode:
            self.action_id += 1
            self.timer.update_time(self.timer.t_step)
            if self.agent.speed > 0:
                self.agent.advance_one_step(self.timer.t_step)
                self.logger.add_move(self.agent.xy)
                self.griddata.add_agent_pos(self.agent.xy)

            # check if path is still straight and if not,
            # compute the previous coverage and contacts
            ind0 = self.agent.check_path_straightness()
            if ind0 is not None:
                rc, ang, success = self.covmap.add_scan(self.agent.xy_hist[ind0],
                                                        self.agent.xy_hist[-2])
                # check contact detections
                if success:
                    obs_str, n_dets = self.contacts.add_dets(rc, ang)
                    # Add to logger
                    self.logger.add_covmap(self.covmap.map_stack[-1])
                    self.logger.add_observation(obs_str,
                                                self.timer.time_remaining)
                    # Add to gridded summary
                    self.griddata.add_cov_map(self.covmap.map_stack[-1])
                    if n_dets > 0:
                        self.griddata.add_contacts(
                            [self.contacts.detections[-n_dets]])

        ag_pos = self.griddata.agent
        occ_map = self.griddata.occ_map
        cov_map = self.griddata.cov_map
        cts = self.griddata.cts

        self.get_reward()

        if hasattr(self, 'agent_viz'):
            self.agent_viz.update(ag_pos,
                                  occ_map,
                                  cov_map[0],
                                  cts)

        if hasattr(self, 'plotter'):
            if success:
                self.plotter.updatecovmap(self.griddata.cov_map)
            self.plotter.update_plots(self.contacts.detections,
                                      self.agent.xy,
                                      self.agent.xy_hist,
                                      self.timer.time_remaining)
            self.plotter.update_rewards(self.reward.rewards[-1])

        self.check_termination()

        return self.timer.time_remaining, ag_pos, occ_map, cov_map, cts

    def get_obs(self):
        ag_pos = self.griddata.agent
        occ_map = self.griddata.occ_map
        cov_map = self.griddata.cov_map
        cts = self.griddata.cts

        return self.timer.time_remaining, ag_pos, occ_map, cov_map, cts

    def prev_step(self):
        if self.action_id > 0:
            # rewind the agent
            rm_cov = self.agent.rewind_one_step()
            self.griddata.add_agent_pos(self.agent.xy)
            self.timer.update_time(-self.timer.t_step)
            if rm_cov:
                if self.agent.ind0 != 0:
                    # delete the cov_maps
                    self.griddata.remove_cov_map(self.covmap.map_stack.pop())
                    # check how many scans have been done
                    n_scans = len(self.covmap.map_stack)
                    det_rm = []
                    # Check whether the contacts need to be removed
                    for d in self.contacts.detections[-1::]:
                        if d.scan_n >= n_scans:
                            det_rm.append(self.contacts.detections.pop())
                            print(det_rm)
                    self.griddata.remove_contacts(det_rm)

            if hasattr(self, 'plotter'):
                self.plotter.updatecovmap(self.griddata.cov_map)
                self.plotter.update_plots(self.contacts.detections,
                                          self.agent.xy,
                                          self.agent.xy_hist,
                                          self.timer.time_remaining)
                self.plotter.update_rewards(self.reward.rewards[-1])

            self.action_id -= 1

    def next_step_pb(self):
        # Get current action_id data and following step cov_map
        t, ap, ip, cm, cn, N_g, ah = self.playback.get_next_step(
            self.action_id)
        _, _, _, cm2, _, _, _ = self.playback.get_next_step(self.action_id+1)

        # Move the agent
        self.agent.move_to_position(ap)

        # Assemble the groups list
        grps = []
        [grps.append(self.contacts.group_loc(cn, n)) for n in range(N_g)]

        # Update the time based on whether the boat has gone forward or back
        if self.action_id > self.action_id_prev:
            self.timer.update_time(self.timer.t_step)
        else:
            self.timer.update_time(-self.timer.t_step)

        # Update the griddata outputs
        self.griddata.add_agent_pos(ap)
        self.griddata.add_contacts(cn)
        if self.action_id > self.action_id_prev:
            if isinstance(cm, np.ndarray):
                self.griddata.add_cov_map(cm)
        else:
            if isinstance(cm2, np.ndarray):
                self.griddata.remove_cov_map(cm2)

        if hasattr(self, 'plotter'):
            self.plotter.updatetime(self.timer.time_remaining,
                                    self.timer.time_lim)
            self.plotter.agent_plt.updateagent(ap)
            self.plotter.agent_plt.updatetrackhist(ah)
            self.plotter.agent_plt.updatetarget(ip, ip)
            self.plotter.updatecovmap(self.griddata.cov_map)
            self.plotter.updatecontacts(cn)
            self.plotter.updategroups(grps)
            self.plotter.update_rewards(self.reward.rewards[self.action_id],
                                        self.reward.rewards[-1])
            self.plotter.draw()

        if hasattr(self, 'agent_viz'):
            ag_pos = self.griddata.agent
            occ_map = self.griddata.occ_map
            cov_map = self.griddata.cov_map
            cts = self.griddata.cts
            self.agent_viz.update(ag_pos,
                                  occ_map,
                                  cov_map[0],
                                  cts)
        self.action_id_prev = self.action_id

    def add_group(self, c_inds):
        # Add the contacts to a cluster and add to log
        self.contacts.dets_to_clus(c_inds)
        g_n, c_inds = self.contacts.add_group()
        self.logger.add_group(g_n, c_inds)

    def remove_group(self, g_inds):
        # Get rid of a group
        self.contacts.remove_group(g_inds)
        self.logger.ungroup(g_inds)

    def add_aux_info(self, aux_info):
        self.logger.add_auxinfo(aux_info)

    def round_to_angle(self, x0, y0, x1, y1):
        ang_r = 10
        r_r = 1
        x_d = x1 - x0
        y_d = y1 - y0
        rho = np.sqrt(x_d**2 + y_d**2)
        phi = np.arctan2(y_d, x_d)
        phi_r = np.deg2rad(ang_r)*np.round(phi/np.deg2rad(ang_r))
        rho_r = r_r*np.round(rho/r_r)
        x = rho_r * np.cos(phi_r) + x0
        y = rho_r * np.sin(phi_r) + y0
        return x, y

    def reset(self):
        self.map_obj.setup()
        self.contacts.generate_targets(self.map_obj.occ)
        # reinitialise everything
        # get new st positions
        ag_st = self.map_obj.get_agent_start()
        self.agent.reset(new_st_pos=ag_st)
        self.covmap.reset()
        self.contacts.reset()
        self.timer.reset()
        self.logger.reset(self.agent.xy,
                          self.contacts.truth,
                          self.map_obj.map_lims,
                          self.timer.time_lim)
        self.griddata.reset(agent_xy=self.agent.xy)
        self.reward.reset()
        if hasattr(self, 'plotter'):
            self.plotter.reset(new_agent_start=self.agent.xy,
                               new_map_lims=self.map_obj.map_lims,
                               new_map_img=self.map_obj.img)
        self.end_episode = False

    def save_episode(self, ep_n=None):

        self.logger.save_data(ep_n)
        self.end_episode = True

    def terminate_episode(self):
        if hasattr(self, 'plotter'):
            self.plotter.reveal_targets(self.contacts.truth)
            self.plotter.remove_temp()
        self.timer.time_remaining = 0
        self.end_episode = True

    # mouse and keyboard callback functions
    def mouse_move(self, event):
        # check it's inside the axes
        if event.inaxes != self.plotter.ax.axes:
            return
        # grab the x,y coords
        x_i, y_i = event.xdata, event.ydata
        if not self.end_episode and self.groupswitch:
            x0, y0 = self.agent.xy[0], self.agent.xy[1]
            if self.snaptoangle:
                x, y = self.round_to_angle(x0, y0, x_i, y_i)
            else:
                x, y = x_i, y_i
            self.timer.update_temp((x0, y0),
                                   (x, y),
                                   self.agent.xy_hist[0],
                                   self.agent.speed0)
            self.plotter.update_temp(x0, y0,
                                     x, y,
                                     self.covmap.leadinleadout,
                                     self.covmap.min_scan_l,
                                     self.covmap.scan_width)
            self.plotter.updatetime(self.timer.time_temp,
                                    self.timer.time_lim)

    def on_key_manual(self, event):
        # normal operation if episode is ongoing
        if not self.end_episode:
            if event.key == "z":
                self.snaptoangle = not self.snaptoangle
            elif event.key == 'shift':
                self.plotter.remove_temp()
                self.groupswitch = not self.groupswitch

            elif event.key == 'a':
                # group the selected
                g_n, c_inds = self.contacts.add_group()
                # only add if there's any points in the group
                if c_inds:
                    # update plot
                    self.plotter.updategroups(self.contacts.det_grp)
                    # log new grouping
                    self.logger.add_group(g_n, c_inds)
            elif event.key == '#':
                self.terminate_episode()
            elif event.key == '=':
                print('Reveal locations with # before saving')
        else:
            # can save file if episode is finished
            if event.key == '=':
                self.save_episode()
        # can reset at any time
        if event.key == 'enter':
            self.reset()

    def on_key_playback(self, event):
        if event.key == 'left':
            if self.action_id == 0:
                return
            else:
                self.play = False
                if self.action_id == self.action_id_prev:
                    self.action_id -= 1

        elif event.key == 'right':
            if self.action_id >= self.playback.ep_end:
                return
            else:
                self.play = False
                if self.action_id == self.action_id_prev:
                    self.action_id += 1

        elif event.key == ' ':
            self.play = not self.play

    def on_click(self, event):
        if event.inaxes != self.plotter.ax.axes:
            return

        if not self.end_episode and self.groupswitch:
            # add new coordinate to agent
            x_i, y_i = event.xdata, event.ydata
            x0, y0 = self.agent.xy[0], self.agent.xy[1]

            # check if the requested position is in an occupied space
            if not self.map_obj.occ[round(y_i), round(x_i)]:
                # check that the path doesn't go through occupied spaces
                cp = self.map_obj.check_path(x0, y0, x_i, y_i)
                if cp:
                    # snap to angle or not
                    if self.snaptoangle:
                        x, y = self.round_to_angle(x0, y0, x_i, y_i)
                    else:
                        x, y = x_i, y_i
                    self.agent.destination_req((x, y))
                else:
                    self.plotter.p.set_facecolor((1, 1, 1))
            else:
                self.plotter.p.set_facecolor((1, 1, 1))

    def on_pick(self, event):
        if not self.end_episode and not self.groupswitch:
            inds = list(event.ind)
            # check if it's a detection
            if event.artist == self.plotter.det_plt:
                plot_det_inds = [n.det_n for n in self.contacts.detections
                                 if n.group_n == None]
                c_inds = [plot_det_inds[n] for n in inds]
                self.contacts.dets_to_clus(c_inds)
                self.plotter.updatecontacts(self.contacts.detections)
            elif event.artist == self.plotter.det_grp_plt:
                # get groud indices
                g_inds = [self.contacts.det_grp[n].group_n for n in inds]
                self.contacts.remove_group(g_inds)
                # update plot
                self.plotter.updatecontacts(self.contacts.detections)
                self.plotter.updategroups(self.contacts.det_grp)
                # log ungrouped
                self.logger.ungroup(g_inds)


if __name__ == '__main__':
    ss = SurveySimulationGrid('manual',
                              'params.txt',
                              'data')
