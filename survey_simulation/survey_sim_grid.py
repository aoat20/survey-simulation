import numpy as np
import os
import time
from survey_simulation.sim_plotter import SurveyPlotter, AgentViz
from survey_simulation.survey_classes import ContactDetections, CoverageMap
from survey_simulation.sim_classes import Timer, Playback, Logger, Map, Agent
import matplotlib.pyplot as plt

class SurveySimulationGrid():
    """_summary_
    """

    def __init__(self,
                 mode='manual',
                 params_filepath='params.txt',
                 save_dir='data',
                 ep_n=0, 
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
        if mode == 'manual':
            self.groupswitch = True
            self.snaptoangle = False
        if mode == 'playback':
            self.action_id = 0
        
        # Load parameters from file
        if mode == "manual" or mode == "test":
            params = self.load_params(params_filepath)
        elif mode == "playback":
            ep_dir = os.path.join(save_dir, "Episode"+str(ep_n))
            l_dir = os.listdir(ep_dir)
            p_path = [x for x in l_dir if 'META' in x][0]
            params = self.load_params(os.path.join(ep_dir, p_path))

        # Modify parameters from arguments
        fta = True
        for key, value in kwargs.items():
            if fta:
                print('Overwriting the following parameters:')
                fta = False
            print("%s = %s" % (key, value))
            params[key] = value

        # Set random seed if required for debugging purposes
        if mode == "manual" or mode == 'test':
            if 'rand_seed' in params.keys():
                np.random.seed(params['rand_seed'])     

        ################ instantiate all the class objects ###############

        # Set up the map
        # if a default map is needed
        # else get the map from the location map_path
        # else just make an empty space to move around in 
        if 'map_n' in params:
            self.map_obj = Map(map_n=params['map_n'])
        elif 'map_path' in params:
            self.map_obj = Map(map_path=params['map_path'])
        else:
            self.map_obj = Map(scan_lims=params['scan_area_lims'],
                               map_lims=params['map_area_lims'])

        # Set the agent start position 
        if params.get('random_start') == 1:
            # Random start anywhere
            agent_start = self.map_obj.random_start()
        elif params.get('random_start') == 2:
            # Random start edges
            agent_start = self.map_obj.random_start_edges()
        else:
            # use specified start if available, else default for map
            if 'agent_start' in params:
                agent_start = params['agent_start']
            else:
                agent_start = self.map_obj.default_start()

        # Get t step and grid spacing
        self.agent = Agent(xy_start=agent_start,
                           speed=params['agent_speed'],
                           scan_thr = params['scan_thr'])
        # Check if agent start position is in the map bounds
        self.map_obj.is_occupied(self.agent.xy)
        self.contacts = ContactDetections(loc_uncertainty=params['loc_uncertainty'],
                                          scan_lims=self.map_obj.scan_lims, 
                                          n_targets=params['n_targets'],
                                          det_probs=params['det_probs'],
                                          clutter_density=params['clutter_dens'],
                                          det_probs_clutter=params['det_probs_clutter'],
                                          clutter_ori_mean=params['clutter_or_mean'],
                                          clutter_ori_std=params['clutter_or_std'])
        self.covmap = CoverageMap(scan_lims=self.map_obj.scan_lims,
                                  leadinleadout=params['leadinleadout'],
                                  min_scan_l=params['min_scan_l'],
                                  scan_width=params['scan_width'],
                                  nadir_width=params['nadir_width'])
        
        if params.get('agent_viz'):
            self.agent_viz = AgentViz(map_dims=self.map_obj.map_lims)
        
        if mode == "manual" or mode == 'test':
            # Generate contact locations
            self.contacts.generate_targets(self.map_obj.occ)
            # instantiate extra objects
            self.timer = Timer(params['time_lim'],
                               params['t_step'])
            self.logger = Logger(agent_start=self.agent.xy,
                                 time_lim=params['time_lim'],
                                 scan_lims=self.map_obj.scan_lims,
                                 params=params,
                                 gnd_trth=self.contacts.truth,
                                 save_dir=save_dir)
        elif mode == "playback": 
            self.playback = Playback(ep_dir)

        if mode == "manual" or mode == "playback" or params['plotter']:
            self.plotter = SurveyPlotter(map_lims=self.map_obj.map_lims,
                                         scan_lims=self.map_obj.scan_lims,
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
        elif mode == 'playback':
            self.plotter.ax.figure.canvas.mpl_connect('key_press_event',
                                                      self.on_key_playback)
        
        # Set the simulator running 
        if mode == "manual":
            self.plotting_loop()
        elif mode == "playback":
            self.plotting_loop_pb()

    def plotting_loop(self):
        while True:
            self.next_step()
            time.sleep(0.04)

    def plotting_loop_pb(self):
        while True:
            if self.play:
                if self.action_id >= self.playback.ep_end:
                    return
                else:
                    self.action_id += 1
            self.next_step_pb()
            time.sleep(0.04)

    def load_params(self,
                    param_file: str):
        param_dict = {}
        with open(param_file) as fh:
            for line in fh:
                # handle empty lines and commented
                if line.strip() and not line[0]=='#':
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

    def get_gridded_obs(self,
                        agent_xy,
                        cov_map,
                        contacts):
        # Agent position on a grid
        agent_pos_grid = np.zeros((self.map_obj.map_lims[3],
                                   self.map_obj.map_lims[1]), 
                                   dtype=int)
        ag_pos_rnd = np.int16(np.floor(agent_xy))
        if not self.map_obj.is_occupied(agent_xy):
            agent_pos_grid[ag_pos_rnd[1],
                           ag_pos_rnd[0]] = 1
            
        # Occupancy map 
        occ_map_grid = self.map_obj.occ

        # Coverage map summary 
        cov_map_grid = np.zeros((self.map_obj.map_lims[3],
                                 self.map_obj.map_lims[1]), dtype=int)
        if cov_map:
            sa = self.map_obj.scan_lims
            cov_map_grid[sa[2]:sa[3],
                         sa[0]:sa[1]] = np.count_nonzero(~np.isnan(cov_map),
                                                         axis=0)
        cov_map_grid = np.flip(cov_map_grid, 0)

        # Contacts
        cts_grid = np.zeros((self.map_obj.map_lims[3],
                             self.map_obj.map_lims[1]), dtype=int)
        base = 1
        for cts in contacts:
            xy = [cts.x, cts.y]
            c_xy_rnd =  np.int16(base * np.round(np.divide(xy, base)))
            cts_grid[c_xy_rnd[1], 
                     c_xy_rnd[0]] += 1
            
        return agent_pos_grid, occ_map_grid, cov_map_grid, cts_grid
            
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
            if isinstance(action,int) or isinstance(action,float):
                self.agent.set_speedandcourse(self.agent.speed0,
                                              action)
            else:
                raise ValueError("Invalid move action. Should be " 
                                 "a float/int representing the course.")

        elif action_type == 'group':
            self.add_group(action)
        elif action_type == 'ungroup':
            self.remove_group(action)

    def updateplots(self):
        # plotting
        if self.covmap.map_stack:
            self.plotter.updatecovmap(self.covmap.map_stack)
        self.plotter.updatecontacts(self.contacts.detections)
        self.plotter.agent.updateagent(self.agent.xy)
        if len(self.agent.xy_hist) > 1:
            self.plotter.agent.updatetrackhist(self.agent.xy_hist)
        self.plotter.updatetime(self.timer.time_remaining,
                                self.timer.time_remaining)
        # self.plotter.remove_temp()
        self.plotter.draw()

    def next_step(self):

        self.timer.update_time(self.timer.t_step)
        if self.agent.speed>0:
            self.agent.advance_one_step(self.timer.t_step)
            self.logger.addmove(self.agent.xy)

        # check if path is still straight and if not, 
        # compute the previous coverage and contacts
        ind0 = self.agent.check_path_straightness()
        if ind0 is not None:
            rc, ang = self.covmap.add_scan(self.agent.xy_hist[ind0],
                                           self.agent.xy_hist[-2])
            # check contact detections
            obs_str = self.contacts.add_dets(rc, ang)
            self.logger.addcovmap(self.covmap.map_stack[-1])
            self.logger.addobservation(obs_str, self.timer.time_remaining)

        if hasattr(self,'plotter'):
            self.updateplots()

        (ag_pos, 
         occ_map, 
         cov_map, 
         cts) = self.get_gridded_obs(self.agent.xy,
                                     cov_map=self.covmap.map_stack,
                                     contacts=self.contacts.detections)
        if hasattr(self, 'agent_viz'):
            self.agent_viz.update(ag_pos,
                                occ_map,
                                cov_map,
                                cts)

        self.check_termination()

        return self.timer.time_remaining, ag_pos, occ_map, cov_map, cts
        
    def next_step_pb(self):
        t, bp, ip, cm, cn, N_g, ah = self.playback.get_data(self.action_id)
        grps = []
        [grps.append(self.contacts.group_loc(cn, n)) for n in range(N_g)]
        self.plotter.updatetime(t, t)
        self.plotter.agent.updateagent(bp)
        self.plotter.agent.updatetrackhist(ah)
        self.plotter.agent.updatetarget(ip, ip)
        self.plotter.updatecovmap(cm)
        self.plotter.updatecontacts(cn)
        self.plotter.updategroups(grps)
        self.plotter.draw()

        if hasattr(self, 'agent_viz'):
            cm_arr = [np.array(c_tmp) for c_tmp in cm]
            (ag_pos, 
             occ_map, 
             cov_map, 
             cts) = self.get_gridded_obs(bp,
                                         cm_arr,
                                         cn)
            self.agent_viz.update(ag_pos,
                                  occ_map,
                                  cov_map,
                                  cts)

    def add_group(self, c_inds):
        # Add the contacts to a cluster and add to log
        self.contacts.dets_to_clus(c_inds)
        g_n, c_inds = self.contacts.add_group()
        self.logger.addgroup(g_n, c_inds)

    def remove_group(self, g_inds):
        # Get rid of a group
        self.contacts.remove_group(g_inds)
        self.logger.ungroup(g_inds)

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
        self.contacts.generate_targets(self.map_obj.occ)
        # reinitialise everything
        self.agent.reset()
        self.covmap.reset()
        self.contacts.reset()
        self.timer.reset()
        self.logger.reset(self.agent.xy, 
                          self.contacts.truth, 
                          self.map_obj.scan_lims,
                          self.timer.time_lim)
        if hasattr(self,'plotter'):
            self.plotter.reset()
        self.end_episode = False

    def save_episode(self, ep_n=None):
        self.logger.save_data(ep_n)
        self.end_episode = True

    def terminate_episode(self):
        if hasattr(self,'plotter'):
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
            self.plotter.updatetime(self.timer.time_remaining,
                                    self.timer.time_temp)
            # self.plotter.fig.canvas.draw_idle()

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
                    self.logger.addgroup(g_n, c_inds)
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

        self.plotter.fig.canvas.draw_idle()

    def on_key_playback(self, event):
        if event.key == 'left':
            if self.action_id == 0:
                return
            else:
                self.action_id -= 1

        elif event.key == 'right':
            if self.action_id >= self.playback.ep_end:
                return
            else:
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
                cp = self.map_obj.check_path(x0,y0,x_i,y_i)
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
            self.plotter.fig.canvas.draw_idle()

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
            self.plotter.fig.canvas.draw_idle()

if __name__ == '__main__':
    ss = SurveySimulationGrid('manual',
                              'params.txt',
                              'data')
