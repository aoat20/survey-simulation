import numpy as np
import os
from multiprocessing import Process, Queue, Pipe 
import threading
from survey_simulation.sim_plotter import SurveyPlotter
from survey_simulation.survey_classes import ContactDetections, CoverageMap
from survey_simulation.sim_classes import Timer, Playback, Logger, Map, Agent

class SurveySimulation():
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
        if 'agent_start' in params:
            agent_start = params['agent_start']
        else:
            agent_start = self.map_obj.default_start()
        # Check if agent start position is in the map bounds
        self.map_obj.is_occupied(agent_start)

        self.agent = Agent(xy_start=agent_start,
                           speed=params['agent_speed'])
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
        if mode == "manual" or mode == 'test':
            # Generate contact locations
            self.contacts.generate_targets(self.map_obj.occ)
            # instantiate extra objects
            self.timer = Timer(params['time_lim'])
            self.logger = Logger(agent_start=self.agent.xy,
                                 time_lim=params['time_lim'],
                                 scan_lims=self.map_obj.scan_lims,
                                 params=params,
                                 gnd_trth=self.contacts.truth,
                                 save_dir=save_dir)
        elif mode == "playback": 
            self.playback = Playback(ep_dir)

        if mode == "manual" or mode == "playback":
            self.plotter = SurveyPlotter(map_lims=self.map_obj.map_lims,
                                         scan_lims=self.map_obj.scan_lims,
                                         grid_size=params['grid_res'],
                                         map_img=self.map_obj.img,
                                         xy0=self.agent.xy,
                                         time_lim=params['time_lim'],
                                         leadinleadout=self.covmap.leadinleadout,
                                         min_scan_l=self.covmap.min_scan_l,
                                         scan_width=self.covmap.scan_width,
                                         min_scan_ang=params['min_scan_angle_diff'],
                                         n_angles=params['N_angles'],
                                         n_looks=params['N_looks'])

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
        if mode == "manual" or mode == "playback":
            self.plotter.show()

        if params['rt']:
            if mode == "manual":
                self.run_manualRT()
            elif mode == "test":
                self.run_testRT()


    def load_params(self,
                    param_file):
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

    def new_action(self,
                   action_type,
                   action):
        # for 'move', action = xy
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
            if not all([isinstance(a, int) for a in action]) and len(action) != 2:
                raise ValueError("Invalid move action. Should be [x,y]")
            else:
                self.add_newxy(action[0], action[1])
                cov_map = self.covmap.map_stack[-1]
                t = self.timer.time_remaining
                cntcts = self.contacts.detections
                return t, cov_map, cntcts, self.map_obj.occ

        elif action_type == 'group':
            self.add_group(action)
        elif action_type == 'ungroup':
            self.remove_group(action)

    def run_manualRT(self):
        pass

    def run_testRT(self):
        pass

    def updateplots(self):
        # plotting
        self.plotter.updatecovmap(self.covmap.map_stack)
        self.plotter.updatecontacts(self.contacts.detections)
        self.plotter.agent.updateagent(self.agent.xy)
        self.plotter.agent.updatetrackhist(self.agent.xy_hist)
        self.plotter.updatetime(self.timer.time_remaining,
                                self.timer.time_remaining)
        self.plotter.remove_temp()
        self.plotter.fig.canvas.draw_idle()

    def add_newxy(self, x, y):
        # add new scan to coverage map
        x0, y0 = self.agent.xy[0], self.agent.xy[1]
        rc, ang = self.covmap.add_scan(x0, y0,
                                        x, y)
        # check contact detections
        obs_str = self.contacts.add_dets(rc, ang)

        self.agent.move_to_position([x, y])
        self.timer.update((x0, y0), (x, y), self.agent.speed)

        # logging
        self.logger.addmove(x, y)
        self.logger.addcovmap(self.covmap.map_stack[-1])
        self.logger.addobservation(obs_str, self.timer.time_remaining)

        # if manual mode, also plot
        if hasattr(self,'plotter'):
            self.updateplots()

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
        self.plotter.reset()

    def save_episode(self, ep_n=None):
        self.logger.save_data(ep_n)
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
                                   self.agent.speed)
            self.plotter.update_temp(x0, y0,
                                     x, y, 
                                     self.covmap.leadinleadout,
                                     self.covmap.min_scan_l,
                                     self.covmap.scan_width)
            self.plotter.updatetime(self.timer.time_remaining,
                                    self.timer.time_temp)
            self.plotter.fig.canvas.draw_idle()

    def on_key_manualrt(self,event):
        if event.key == " ":
            self.play = not self.play
        elif event.key == "up":
            # increase playback speed
            self.play_speed *= 2
            print('play speed = ' + str(self.play_speed))
            self.compute_movement_step()
        elif event.key == "down": 
            # decrease playback speed
            self.play_speed *= 0.5
            print('play speed = ' + str(self.play_speed))
            self.compute_movement_step()

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
                self.plotter.reveal_targets(self.contacts.truth)
                self.plotter.remove_temp()
                #self.plotter.fig.canvas.draw()
                self.end_episode = True

            elif event.key == '=':
                print('Reveal locations with # before saving')
        else:   # can save file if episode is finished
            if event.key == '=':
                self.save_episode()

        # can reset at any time
        if event.key == 'enter':
            self.reset()
            self.end_episode = False

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

        t, bp, ip, cm, cn, N_g, ah = self.playback.get_data(self.action_id)
        grps = []
        [grps.append(self.contacts.group_loc(cn, n)) for n in range(N_g)]
        self.plotter.updatetime(t, t)
        self.plotter.agent.updateagent([bp[0], bp[1]])
        self.plotter.agent.updatetrackhist(ah)
        self.plotter.agent.updatetarget(ip,ip)
        self.plotter.updatecovmap(cm)
        self.plotter.updatecontacts(cn)
        self.plotter.updategroups(grps)
        self.plotter.fig.canvas.draw_idle()

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
                    self.add_newxy(x, y)
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
    ss = SurveySimulation()
