import numpy as np
import os
import time
from survey_simulation.sim_plotter import SurveyPlotter, AgentViz
from survey_simulation.survey_classes import ContactDetections, CoverageMap
from survey_simulation.sim_classes import Timer, Playback, Logger, Map, Agent, GriddedData
from survey_simulation.reward import RewardFunction
import matplotlib.pyplot as plt
from skimage.measure import block_reduce


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
        self._play = False
        self._fig_closed = False
        self._action_id = 0

        if mode == 'manual':
            self._groupswitch = True
            self._snaptoangle = False
        if mode == 'playback':
            self._action_id_prev = 0

        # Load parameters from file
        if mode == "manual" or mode == "test":
            params = self._load_params(params_filepath)
        elif mode == "playback":
            if log_file != '':
                ep_dir = log_file
            else:
                ep_dir = os.path.join(save_dir, "Episode"+str(ep_n))
            print('Loading log file: '+ep_dir)
            l_dir = os.listdir(ep_dir)
            p_path = [x for x in l_dir if 'META' in x][0]
            map_path = [x for x in l_dir if 'MAP' in x][0]
            params = self._load_params(os.path.join(ep_dir, p_path))

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
            self._map_obj = Map(map_path=os.path.join(ep_dir, map_path))
        else:
            if 'map_n' in params:
                self._map_obj = Map(map_n=params['map_n'])
            elif 'map_path' in params:
                self._map_obj = Map(map_path=params['map_path'])
            elif 'map_random' in params:
                print('Random map being generated')
                self._map_obj = Map(random_map=True)
            else:
                try:
                    self._map_obj = Map(map_lims=params['map_area_lims'])
                except:
                    raise Exception('Need to include some map specification' +
                                    ' (map_n, map_path, map_random or map_area_lims )')

        if mode == "playback":
            self._playback = Playback(ep_dir)

        # Set the agent start position
        if mode == 'playback':
            d_temp = self._playback.get_data(0)
            agent_start = d_temp[1]
        elif params.get('random_start') == 1:
            # Random start anywhere
            self._map_obj.ag_st_mode = 2
        elif params.get('random_start') == 2:
            # Random start edges
            self._map_obj.ag_st_mode = 3
        else:
            # use specified start if available, else default for map
            if 'agent_start' in params:
                self._map_obj.ag_st_mode = 1
                self._map_obj.agent_st = params['agent_start']
            else:
                self._map_obj.ag_st_mode = 0

        if mode != 'playback':
            agent_start = self._map_obj.get_agent_start()

        # Get t step and grid spacing
        self._agent = Agent(xy_start=agent_start,
                            speed=params['agent_speed'],
                            scan_thr=params['scan_thr'])
        # Check if agent start position is in the map bounds
        if self._map_obj.is_occupied(self._agent.xy):
            raise Exception("Agent position is not valid")
        self._contacts = ContactDetections(loc_uncertainty=params['loc_uncertainty'],
                                           n_targets=params['n_targets'],
                                           det_probs=params['det_probs'],
                                           clutter_density=params['clutter_dens'],
                                           det_probs_clutter=params['det_probs_clutter'],
                                           clutter_ori_mean=params['clutter_or_mean'],
                                           clutter_ori_std=params['clutter_or_std'])
        self._covmap = CoverageMap(map_lims=self._map_obj.map_lims,
                                   leadinleadout=params['leadinleadout'],
                                   min_scan_l=params['min_scan_l'],
                                   scan_width=params['scan_width'],
                                   nadir_width=params['nadir_width'])

        if 'output_mat_size' in params:
            self.gd_res = params['output_mat_size']
        else:
            self.gd_res = 0
        self._griddata = GriddedData(map_lims=self._map_obj.map_lims,
                                     angle_diff=params['min_scan_angle_diff'],
                                     occ_map=self._map_obj.occ,
                                     agent_xy=self._agent.xy)
        self._timer = Timer(params['time_lim'],
                            params['t_step'])
        if "reward_id" in params:
            self._reward = RewardFunction(reward_id=params['reward_id'])
        else:
            self._reward = RewardFunction()

        if mode == 'playback':
            self._run_to_end()

        if params.get('agent_viz'):
            self._agent_viz = AgentViz(map_dims=self._map_obj.map_lims,
                                       occ_map=self._map_obj.occ)

        if mode == "manual" or mode == 'test':
            # Generate contact locations
            self._contacts.generate_targets(self._map_obj.occ)
            # instantiate extra objects
            self._logger = Logger(agent_start=self._agent.xy,
                                  time_lim=params['time_lim'],
                                  map_lims=self._map_obj.map_lims,
                                  params=params,
                                  gnd_trth=self._contacts.truth,
                                  save_dir=save_dir,
                                  map_img=self._map_obj.img)

        if mode == "manual" or params['plotter']:
            self._plotter = SurveyPlotter(map_lims=self._map_obj.map_lims,
                                          map_img=self._map_obj.img,
                                          xy0=self._agent.xy,
                                          time_lim=params['time_lim'],
                                          leadinleadout=self._covmap.leadinleadout,
                                          min_scan_l=self._covmap.min_scan_l,
                                          scan_width=self._covmap.scan_width,
                                          min_scan_ang=params['min_scan_angle_diff'],
                                          n_angles=params['N_angles'],
                                          n_looks=params['N_looks'])
            self._plotter.show(blocking=False)
            self._plotter.draw()

        # Set up event handlers
        if mode == "manual":
            # mouse and key event handlers
            self._plotter.ax.figure.canvas.mpl_connect('button_press_event',
                                                       self._on_click)
            self._plotter.ax.figure.canvas.mpl_connect('motion_notify_event',
                                                       self._mouse_move)
            self._plotter.ax.figure.canvas.mpl_connect('pick_event',
                                                       self._on_pick)
            self._plotter.ax.figure.canvas.mpl_connect('key_press_event',
                                                       self._on_key_manual)
        elif mode == 'playback' and hasattr(self, '_plotter'):
            self._plotter.ax.figure.canvas.mpl_connect('key_press_event',
                                                       self._on_key_playback)
            self._plotter.show_reward_graph(self._reward.rewards)

        # Set the simulator running
        if mode == "manual":
            self._plotting_loop()
        elif mode == "playback" and hasattr(self, '_plotter'):
            self._plotting_loop_pb()

# Public functions
    def waypoint_reached(self):
        if self._agent.speed == 0:
            return True
        else:
            return False

    def get_reward(self,
                   inst_or_cum='i'):
        if inst_or_cum == 'i':
            return self._reward.reward
        elif inst_or_cum == 'c':
            return self._reward.rewards
        else:
            raise ValueError("Invalid reward. Expected i or c")

    def get_map_shape(self):
        return self._map_obj.occ.shape

    def new_action(self,
                   action_type: str,
                   action):
        # for 'move', action = course
        # for 'waypoint', action = (x, y)
        # for 'group', action = c_inds
        # for 'ungroup' action = g_ind
        action_types = ['move',
                        'waypoint',
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
                self._agent.distance_dest = np.inf
                self._agent.set_speedandcourse(self._agent.speed0,
                                               action)
            else:
                raise ValueError("Invalid move action. Should be "
                                 "a float/int representing the course.")

        elif action_type == 'waypoint':
            self._agent.destination_req(action)

        elif action_type == 'group':
            self._add_group(action)
        elif action_type == 'ungroup':
            self._remove_group(action)

    def next_step(self):
        success = False
        if not self.end_episode:
            self._action_id += 1
            self._timer.update_time(self._timer.t_step)
            if self._agent.speed > 0:
                self._agent.advance_one_step(self._timer.t_step)
                self._logger.add_move(self._agent.xy)
                self._griddata.add_agent_pos(self._agent.xy)

            # check if path is still straight and if not,
            # compute the previous coverage and contacts
            ind0 = self._agent.check_path_straightness()
            if ind0 is not None:
                rc, ang, success = self._covmap.add_scan(self._agent.xy_hist[ind0],
                                                         self._agent.xy_hist[-2])
                # check contact detections
                if success:
                    obs_str, n_dets = self._contacts.add_dets(rc, ang)
                    # Add to logger
                    self._logger.add_covmap(self._covmap.map_stack[-1])
                    self._logger.add_observation(obs_str,
                                                 self._timer.time_remaining)
                    # Add to gridded summary
                    self._griddata.add_cov_map(self._covmap.map_stack[-1])
                    if n_dets > 0:
                        self._griddata.add_contacts(
                            [self._contacts.detections[-n_dets]])

        t_rem, ag_pos, occ_map, cov_map, cts = self.get_obs()
        self._compute_reward()

        if hasattr(self, '_agent_viz'):
            self._agent_viz.update(ag_pos,
                                   occ_map,
                                   cov_map[0],
                                   cts)

        if hasattr(self, '_plotter'):
            if success:
                self._plotter.updatecovmap(self._griddata.cov_map)
            self._plotter.update_plots(self._contacts.detections,
                                       self._agent.xy,
                                       self._agent.xy_hist,
                                       self._timer.time_remaining)

            if all(self._agent.destination != self._agent.xy_hist[0]):
                self._plotter.agent_plt.updatetarget(self._agent.destination,
                                                     self._agent.xy0)
            self._plotter.update_rewards(self._reward.rewards[-1])

        self._check_termination()

        return t_rem, ag_pos, occ_map, cov_map, cts

    def get_obs(self):
        t = self._timer.time_remaining

        if self.gd_res == 0:
            ag_pos = self._griddata.agent
            occ_map = self._griddata.occ_map
            cov_map = self._griddata.cov_map
            cts = self._griddata.cts
        else:
            ag_pos = self.ds_array(self._griddata.agent)
            occ_map = self.ds_array(self._griddata.occ_map)
            cov_map = self.ds_array(self._griddata.cov_map)
            cts = self.ds_array(self._griddata.cts)

        return t, ag_pos, occ_map, cov_map, cts

    def ds_array(self,
                 array_in):

        array_in_np = np.array(array_in)
        # Make the matrix square before decimating
        array_in_sq = self.squarify_pow2(array_in_np,
                                         0)
        ds_factor = int(array_in_sq.shape[1]/self.gd_res)
        if len(array_in_np.shape) == 2:
            bl_size = (ds_factor, ds_factor)
        elif len(array_in_np.shape) == 3:
            bl_size = (1, ds_factor, ds_factor)
        array_out = block_reduce(array_in_sq,
                                 block_size=bl_size,
                                 func=np.max)

        return array_out

    def squarify_pow2(self, M, val=0):

        if len(M.shape) == 2:
            (a, b) = M.shape
        elif len(M.shape) == 3:
            (c, a, b) = M.shape

        # find next power of two of largest dimension
        d_pow2 = int(2**np.ceil(np.log2(np.max((a, b)))))
        padding = ((0, d_pow2-a), (0, d_pow2-b))
        if len(M.shape) == 3:
            padding = ((0, 0), padding[0], padding[1])

        return np.pad(M, padding, mode='constant', constant_values=val)

    def prev_step(self):
        if self._action_id > 0:
            # rewind the agent
            rm_cov = self._agent.rewind_one_step()
            self._griddata.add_agent_pos(self._agent.xy)
            self._timer.update_time(-self._timer.t_step)
            if rm_cov:
                if self._agent.ind0 != 0:
                    # delete the cov_maps
                    self._griddata.remove_cov_map(self._covmap.map_stack.pop())
                    # check how many scans have been done
                    n_scans = len(self._covmap.map_stack)
                    det_rm = []
                    # Check whether the contacts need to be removed
                    for d in self._contacts.detections[-1::]:
                        if d.scan_n >= n_scans:
                            det_rm.append(self._contacts.detections.pop())
                    self._griddata.remove_contacts(det_rm)

            if hasattr(self, '_plotter'):
                self._plotter.updatecovmap(self._griddata.cov_map)
                self._plotter.update_plots(self._contacts.detections,
                                           self._agent.xy,
                                           self._agent.xy_hist,
                                           self._timer.time_remaining)
                self._plotter.update_rewards(self._reward.rewards[-1])

            self._action_id -= 1

    def add_aux_info(self, aux_info):
        self._logger.add_auxinfo(aux_info)

    def reset(self):
        self._map_obj.setup()
        self._contacts.generate_targets(self._map_obj.occ)
        # reinitialise everything
        # get new st positions
        ag_st = self._map_obj.get_agent_start()
        self._agent.reset(new_st_pos=ag_st)
        self._covmap.reset()
        self._contacts.reset()
        self._timer.reset()
        self._logger.reset(self._agent.xy,
                           self._contacts.truth,
                           self._map_obj.map_lims,
                           self._timer.time_lim)
        self._griddata.reset(agent_xy=self._agent.xy)
        self._reward.reset()
        if hasattr(self, '_plotter'):
            self._plotter.reset(new_agent_start=self._agent.xy,
                                new_map_lims=self._map_obj.map_lims,
                                new_map_img=self._map_obj.img)
        self.end_episode = False

    def save_episode(self, ep_n=None):

        self._logger.save_data(ep_n)
        self.end_episode = True

# Other functions
    def _plotting_loop(self):
        while True:
            tic = time.time()
            if self._agent.speed != 0:
                self.next_step()
            t_el = np.clip(time.time() - tic, 0, 0.04)
            self._plotter.pause(0.0401 - t_el)

    def _plotting_loop_pb(self):
        self._plotter.draw()
        while True:
            if self._play:
                if self._action_id >= self._playback.ep_end:
                    self._plotter.pause(0.04)
                else:
                    self._action_id += 1
                    self._plotter.pause(0.04)
            if self._action_id_prev != self._action_id:
                self._next_step_pb()

            self._plotter.pause(0.04)

    def _playback_step(self):
        if self._action_id >= self._playback.ep_end:
            return 0
        else:
            self._action_id += 1
            self._next_step_pb()
            return 1

    def _compute_reward(self):
        # Get all the observations for the rewards function
        obs_dict = {'cov_map': self._griddata.cov_map,
                    'path_length': self._agent.get_current_path_len()}
        self._reward.get_reward(obs_dict=obs_dict)

    def _run_to_end(self):
        running = True
        while running:
            running = self._playback_step()
            self._compute_reward()

        self._run_to_start()

    def _run_to_start(self):
        while self._action_id > 0:
            self._action_id -= 1
            self._next_step_pb()

    def _load_params(self,
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

    def _check_termination(self):
        # Termination conditions
        # Timer runs out
        if self._timer.time_remaining < 0:
            self._terminate_episode()
            self.termination_reason = "Run out of time"
        # Hits land
        if self._map_obj.is_occupied(self._agent.xy):
            self._terminate_episode()
            self.termination_reason = "Grounded"
        # Returns home

    def _next_step_pb(self):
        # Get current action_id data and following step cov_map
        t, ap, ip, cm, cn, N_g, ah = self._playback.get_next_step(
            self._action_id)
        _, _, _, cm2, _, _, _ = self._playback.get_next_step(self._action_id+1)

        # Move the agent
        self._agent.move_to_position(ap)

        # Assemble the groups list
        grps = []
        [grps.append(self._contacts.group_loc(cn, n)) for n in range(N_g)]

        # Update the time based on whether the boat has gone forward or back
        if self._action_id > self._action_id_prev:
            self._timer.update_time(self._timer.t_step)
        else:
            self._timer.update_time(-self._timer.t_step)

        # Update the griddata outputs
        self._griddata.add_agent_pos(ap)
        self._griddata.add_contacts(cn)
        if self._action_id > self._action_id_prev:
            if isinstance(cm, np.ndarray):
                self._griddata.add_cov_map(cm)
        else:
            if isinstance(cm2, np.ndarray):
                self._griddata.remove_cov_map(cm2)

        if hasattr(self, '_plotter'):
            self._plotter.updatetime(self._timer.time_remaining,
                                     self._timer.time_lim)
            self._plotter.agent_plt.updateagent(ap)
            self._plotter.agent_plt.updatetrackhist(ah)
            self._plotter.agent_plt.updatetarget(ip, ip)
            self._plotter.updatecovmap(self._griddata.cov_map)
            self._plotter.updatecontacts(cn)
            self._plotter.updategroups(grps)
            self._plotter.update_rewards(self._reward.rewards[self._action_id],
                                         self._reward.rewards[-1])
            self._plotter.update_rew_time(self._timer.time_elapsed)
            self._plotter.draw()

        if hasattr(self, '_agent_viz'):
            ag_pos = self._griddata.agent
            occ_map = self._griddata.occ_map
            cov_map = self._griddata.cov_map
            cts = self._griddata.cts
            self._agent_viz.update(ag_pos,
                                   occ_map,
                                   cov_map[0],
                                   cts)
        self._action_id_prev = self._action_id

    def _add_group(self, c_inds):
        # Add the contacts to a cluster and add to log
        self._contacts.dets_to_clus(c_inds)
        g_n, c_inds = self._contacts.add_group()
        self._logger.add_group(g_n, c_inds)

    def _remove_group(self, g_inds):
        # Get rid of a group
        self._contacts.remove_group(g_inds)
        self._logger.ungroup(g_inds)

    def _round_to_angle(self, x0, y0, x1, y1):
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

    def _terminate_episode(self):
        if hasattr(self, '_plotter'):
            self._plotter.reveal_targets(self._contacts.truth)
            self._plotter.remove_temp()
        self._timer.time_remaining = 0
        self.end_episode = True

# mouse and keyboard callback functions
    def _mouse_move(self, event):
        # check it's inside the axes
        if event.inaxes != self._plotter.ax.axes:
            return
        # grab the x,y coords
        x_i, y_i = event.xdata, event.ydata
        if not self.end_episode and self._groupswitch:
            x0, y0 = self._agent.xy[0], self._agent.xy[1]
            if self._snaptoangle:
                x, y = self._round_to_angle(x0, y0, x_i, y_i)
            else:
                x, y = x_i, y_i
            self._timer.update_temp((x0, y0),
                                    (x, y),
                                    self._agent.xy_hist[0],
                                    self._agent.speed0)
            self._plotter.update_temp(x0, y0,
                                      x, y,
                                      self._covmap.leadinleadout,
                                      self._covmap.min_scan_l,
                                      self._covmap.scan_width)
            self._plotter.updatetime(self._timer.time_temp,
                                     self._timer.time_lim)

    def _on_key_manual(self, event):
        # normal operation if episode is ongoing
        if not self.end_episode:
            if event.key == "z":
                self._snaptoangle = not self._snaptoangle
            elif event.key == 'shift':
                self._plotter.remove_temp()
                self._groupswitch = not self._groupswitch

            elif event.key == 'a':
                # group the selected
                g_n, c_inds = self._contacts.add_group()
                # only add if there's any points in the group
                if c_inds:
                    # update plot
                    self._plotter.updategroups(self._contacts.det_grp)
                    # log new grouping
                    self._logger.add_group(g_n, c_inds)
            elif event.key == '#':
                self._terminate_episode()
            elif event.key == '=':
                print('Reveal locations with # before saving')
        else:
            # can save file if episode is finished
            if event.key == '=':
                self.save_episode()
        # can reset at any time
        if event.key == 'enter':
            self.reset()

    def _on_key_playback(self, event):
        if event.key == 'left':
            if self._action_id == 0:
                return
            else:
                self._play = False
                if self._action_id == self._action_id_prev:
                    self._action_id -= 1

        elif event.key == 'right':
            if self._action_id >= self._playback.ep_end:
                return
            else:
                self._play = False
                if self._action_id == self._action_id_prev:
                    self._action_id += 1

        elif event.key == ' ':
            self._play = not self._play

    def _on_click(self, event):
        if event.inaxes != self._plotter.ax.axes:
            return

        if not self.end_episode and self._groupswitch:
            # add new coordinate to agent
            x_i, y_i = event.xdata, event.ydata
            x0, y0 = self._agent.xy[0], self._agent.xy[1]

            # check if the requested position is in an occupied space
            if not self._map_obj.occ[round(y_i), round(x_i)]:
                # check that the path doesn't go through occupied spaces
                cp = self._map_obj.check_path(x0, y0, x_i, y_i)
                if cp:
                    # snap to angle or not
                    if self._snaptoangle:
                        x, y = self._round_to_angle(x0, y0, x_i, y_i)
                    else:
                        x, y = x_i, y_i
                    self._agent.destination_req((x, y))
                else:
                    self._plotter.p.set_facecolor((1, 1, 1))
            else:
                self._plotter.p.set_facecolor((1, 1, 1))

    def _on_pick(self, event):
        if not self.end_episode and not self._groupswitch:
            inds = list(event.ind)
            # check if it's a detection
            if event.artist == self._plotter.det_plt:
                plot_det_inds = [n.det_n for n in self._contacts.detections
                                 if n.group_n == None]
                c_inds = [plot_det_inds[n] for n in inds]
                self._contacts.dets_to_clus(c_inds)
                self._plotter.updatecontacts(self._contacts.detections)
            elif event.artist == self._plotter.det_grp_plt:
                # get groud indices
                g_inds = [self._contacts.det_grp[n].group_n for n in inds]
                self._contacts.remove_group(g_inds)
                # update plot
                self._plotter.updatecontacts(self._contacts.detections)
                self._plotter.updategroups(self._contacts.det_grp)
                # log ungrouped
                self._logger.ungroup(g_inds)


if __name__ == '__main__':
    ss = SurveySimulationGrid('manual',
                              'params.txt',
                              'data')
