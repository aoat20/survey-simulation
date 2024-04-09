import matplotlib.pylab as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import math
import os
import shutil
from dataclasses import dataclass
from PIL import Image

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
        self.mode = mode
        self.save_dir = save_dir
        self.end_episode = False
        self.move_complete = True
        self.play = True
           
        # Load parameters from file
        if mode == "manual" or mode == "test":
            self.params = self.load_params(params_filepath)
        elif mode == "playback":
            ep_dir = os.path.join(save_dir, "Episode"+str(ep_n))
            l_dir = os.listdir(ep_dir)
            p_path = [x for x in l_dir if 'META' in x][0]
            self.params = self.load_params(os.path.join(ep_dir, p_path))

        # Modify parameters from arguments
        for key, value in kwargs.items():
            print('Overwriting the following parameters:')
            print("%s = %s" % (key, value))
            self.params[key] = value

        # Set random seed
        if mode == "manual" or mode == 'test':
            if 'rand_seed' in self.params.keys():
                np.random.seed(self.params['rand_seed'])     

        # Set up the map
        # if no map specified, set to no map
        if 'map_n' in self.params:
            map_n = self.params['map_n']
            map_path = 'maps/Map'+str(map_n)+'.png'
        elif 'map_path' in self.params:
            map_path = self.params['map_path']
        else:
            map_path = ''
        map_img = self.map_setup(map_path)

        # Set the agent start position for each default map
        if not 'agent_start' in self.params:
            if map_n==1:
                self.params['agent_start'] = (58.,192.)
            elif map_n==2:
                self.params['agent_start'] = (31.,55.)
            elif map_n==3:
                self.params['agent_start'] = (200., 175)

        # Generate contact locations
        if mode == "manual" or mode == 'test':
            self.generate_contacts(self.params)

        # Set up flags and initial values
        self.rt = self.params['rt']
        self.agent_pos = self.params['agent_start']
        self.agent_pos_hist = [np.array(self.params['agent_start'])]
        # Check if agent start position is in the map bounds
        ml = self.params['map_area_lims']
        ap = self.agent_pos
        if ap[0]<ml[0] or ap[0]>ml[1] or ap[1]<ml[2] or ap[1]>ml[3]:
            raise Exception("agent_start is not within the bounds of the map")
        if self.map_mask[ap[0],ap[1]]:
            raise Exception("agent_start is on an occupied coordinate of the map")

        if mode == 'manual':
            self.groupswitch = True
            self.snaptoangle = False
            self.xy_temp = [0,0]
        if mode == 'playback':
            self.action_id = 0

        # instantiate all the class objects
        if mode == "manual" or mode == 'test':
            # instantiate everything
            self.ais_locs = self.load_aislocs()
            self.ais_loc = self.get_other_boat_locs(0)
            self.covmap = self.CoverageMap(self.params)
            self.contacts = self.ContactDetections(self.params)
            self.timer = self.Timer(self.params)
            self.logger = self.Logger(self.params,
                                      self.contacts_t,
                                      save_dir)
        if mode == "manual": 
            self.plotter = self.Plotter(self.params, map_img)
        elif mode == "playback": 
            self.contacts = self.ContactDetections(self.params)
            self.playback = self.Playback(ep_dir)
            self.plotter = self.Plotter(self.params, map_img)

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
            
        # Set the simulator off running
        if self.rt and mode == "manual":
            self.run_realtime()
        elif mode == "manual" or mode == "playback":
            plt.show()

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
                if not self.rt: 
                    cov_map = self.covmap.map_stack[-1]
                    t = self.timer.time_remaining
                    cntcts = self.contacts.detections
                    return t, cov_map, cntcts, self.map_mask

        elif action_type == 'group':
            self.add_group(action)
        elif action_type == 'ungroup':
            self.remove_group(action)

    def run_realtime(self):
        rfr = 0.05
        self.move_req = False
        
        # get params
        self.play_speed = self.params['play_speed']

        # initialise positions
        self.agent_xy0 = self.agent_pos
        self.xy_target = self.agent_pos
        self.d = 0
        self.d_final = 0

        self.compute_movement_step()

        # run at the rfr rate
        timer = self.plotter.fig.canvas.new_timer(interval=rfr*1000)
        timer.add_callback(self.rt_loop_callback,rfr)
        timer.start()
        plt.show()

    def rt_loop_callback(self,rfr):
        if self.play: 
            # decrease time
            self.timer.update_time(self.play_speed*rfr)
            # update other boat locations
            self.ais_loc = self.get_other_boat_locs(self.timer.time_elapsed)
            # if there's a move request travel
            if self.move_req:
                self.travel_one_step()
                self.plotter.updateagent(self.agent_pos)

            # update plots
            self.plotter.updatetime(self.timer.time_remaining,
                                    self.timer.time_temp)
            self.plotter.update_temp(self.agent_pos[0], self.agent_pos[1], 
                                        self.xy_temp[0], self.xy_temp[1])
            self.plotter.updateaislocs(self.ais_loc)
            self.plotter.fig.canvas.draw_idle()

    def updatescans(self):
        # add scan and contacts
        xend, yend = self.agent_pos[0], self.agent_pos[1]
        x0, y0  = self.agent_xy0[0], self.agent_xy0[1]
        rc, ang = self.covmap.add_scan(x0, y0,
                            xend, yend)
        obs_str = self.contacts.add_dets(self.contacts_t, rc, ang)
        self.agent_pos_hist = np.append(self.agent_pos_hist,[[xend,yend]],axis=0)

        if self.mode == 'manual':
            self.updateplots()
        # add move request to logger
        if not self.move_complete:
            self.logger.addplanchange(self.agent_pos[0],self.agent_pos[1])
        self.logger.addcovmap(self.covmap.map_stack[-1])
        self.logger.addobservation(obs_str, self.timer.time_remaining)

    def updateplots(self):
        # plotting
        self.plotter.updatecovmap(self.covmap.map_stack)
        self.plotter.updatecontacts(self.contacts.detections)
        self.plotter.updateagent(self.agent_pos)
        self.plotter.updatetrackhist(self.agent_pos_hist)
        self.plotter.updatetime(self.timer.time_remaining,
                                self.timer.time_remaining)
        self.plotter.remove_temp()
        self.plotter.fig.canvas.draw_idle()

    def movement_request(self, x, y): 

        self.xy_target = (x, y)
        self.move_req = True
        self.move_complete = False
        # Get the start position of the scan
        x0, y0 = self.agent_pos[0], self.agent_pos[1]
        self.agent_xy0 = (x0, y0)
        self.xy_prev = (x0, y0)
        self.d = 0 

        if self.mode == 'manual':
            self.plotter.updatetarget((x,y), (x0, y0))

        # get move complete distance
        self.d_final = np.sqrt((x-x0)**2 + (y-y0)**2)
        self.compute_movement_step()
        
        # add movement request to logger
        self.logger.addmove(x,y)

    def compute_movement_step(self):
        agent_speed = self.params['agent_speed']

        x0, y0 = self.agent_xy0[0], self.agent_xy0[1]
        x, y = self.xy_target[0], self.xy_target[1]

        heading = np.arctan2(x-x0, y-y0)
        self.d_step = agent_speed*self.play_speed
        self.xy_step = (self.d_step*np.sin(heading), self.d_step*np.cos(heading))
        
    def travel_one_step(self):
        self.d = self.d+self.d_step
        if self.d<self.d_final:
            # move towards target
            self.agent_pos = [self.xy_prev[0]+self.xy_step[0], 
                                self.xy_prev[1]+self.xy_step[1]]
            self.xy_prev = (self.agent_pos[0], self.agent_pos[1])
        else: 
            self.agent_pos = self.xy_target
            self.move_complete = True
            self.move_req = False
            self.updatescans()

    def continue_to_target(self):
        self.travel_one_step()
        # output whether the move is complete, the current position, how long left and 
        # TODO the locations of other vessels 
        return self.move_complete, self.agent_pos, self.timer.time_remaining

    def add_newxy(self, x, y):
        # if real-time switch is on, travel to target. If off, instantaneously arrive at target
        if self.rt:
            self.movement_request(x,y)
            if self.mode == 'manual':
                pass
                #self.travel_to_target()
            else: 
                self.travel_one_step()
        else:
            # add new scan to coverage map
            x0, y0 = self.agent_pos[0], self.agent_pos[1]
            rc, ang = self.covmap.add_scan(x0, y0,
                                        x, y)
            # check contact detections
            obs_str = self.contacts.add_dets(self.contacts_t, rc, ang)
            self.agent_pos_hist = np.append(self.agent_pos_hist,[[x,y]],axis=0)

            self.agent_pos = [x, y]
            self.timer.update((x0, y0), (x, y))

            # logging
            self.logger.addmove(x, y)
            self.logger.addcovmap(self.covmap.map_stack[-1])
            self.logger.addobservation(obs_str, self.timer.time_remaining)

            # if manual mode, also plot
            if self.mode == 'manual':
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

    def generate_contacts(self, params):
        # get relevant params
        nt = params['n_targets']
        nc = params['n_clutter']
        dp = params['det_probs']
        dpc = params['det_probs_clutter']
        com = params['clutter_or_mean']
        cos = params['clutter_or_std']
        nz_i = np.where(self.map_mask == 0)

        self.contacts_t = []
        # generate target contacts
        for n in np.arange(nt):
            n_rnd = np.random.randint(0,len(nz_i[0]))
            self.contacts_t.append(TargetObject(n,
                                                'Target',
                                                (nz_i[1][n_rnd],
                                                 nz_i[0][n_rnd]),
                                                round(np.random.uniform(0, 360)),
                                                dp))
        # generate non-target contacts
        for n in np.arange(nc):
            n_rnd = np.random.randint(0,len(nz_i[0]))
            self.contacts_t.append(TargetObject(n+nt,
                                                'Clutter',
                                                (nz_i[1][n_rnd],
                                                 nz_i[0][n_rnd]),
                                                np.random.normal(com, cos), 
                                                dpc))

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

    def map_setup(self, map_path):
        if map_path:
            # setup for a default map
            img = np.asarray(Image.open(map_path))
            img_tmp = img[:,:,0]
            img_nz = np.where(img_tmp==0)
            sa_bounds = (min(img_nz[1]), max(img_nz[1]), 
                        min(img_nz[0]), max(img_nz[0]))

            self.params['scan_area_lims'] = sa_bounds
            self.params['map_area_lims'] = (0, img_tmp.shape[1], 
                                            0, img_tmp.shape[0])
            self.map_mask  = np.where(img_tmp==0, 0, 1)
            return img
        else:
            mp = self.params['map_area_lims']
            self.map_mask = np.zeros((mp[3],mp[1]))
            return np.array([])

    def reset(self):
        self.generate_contacts(self.params)
        # reinitialise everything
        self.agent_pos = self.params['agent_start']
        self.agent_pos_hist = [np.array(self.params['agent_start'])]
        self.covmap.__init__(self.params)
        self.contacts.__init__(self.params)
        self.timer.__init__(self.params)
        self.logger.__init__(self.params, self.contacts_t, self.save_dir)
        self.plotter.reset()

    def save_episode(self, ep_n=None):
        if not ep_n:
            i = 0
            while os.path.exists(os.path.join(self.save_dir,
                                            f"Episode{i}",
                                            "COVERAGE")):
                i += 1
            self.logger.save_data(str(i))
            print('Saving output to Episode '+str(i))
            self.end_episode = True
        else: 
            self.logger.save_data(ep_n)
            print('Saving output to Episode '+str(ep_n))
            self.end_episode = True

    def check_path(self, x1i, y1i, x2i, y2i): 
        
        x1, y1 = round(x1i), round(y1i)
        x2, y2 = round(x2i), round(y2i)
        
        dx = x2 - x1
        dy = y2 - y1

        xsign = 1 if dx > 0 else -1
        ysign = 1 if dy > 0 else -1

        dx = abs(dx)
        dy = abs(dy)
        if dx > dy:
            xx, xy, yx, yy = xsign, 0, 0, ysign
        else:
            dx, dy = dy, dx
            xx, xy, yx, yy = 0, ysign, xsign, 0

        p = 2*dy - dx
        y = 0

        for x in range(dx + 1):
            xout, yout =  x1 + x*xx + y*yx, y1 + x*xy + y*yy
            if p >= 0:
                y += 1
                p -= 2*dx
            p += 2*dy   
               
            if self.map_mask[yout,xout]: 
                return 0 
        return 1

    def load_aislocs(self):
        ais_locstemp = []
        with open('AISLocations') as fh: 
            for line in fh: 
                if line.strip():
                    ais_locstemp.append([float(item)
                                   for item in line.strip().split()])
        ais_locs = []
        for n_ind in range(int(ais_locstemp[-1][0])+1):      
            ais_locs.append(np.array([lt[1:] for lt in ais_locstemp 
                                if int(lt[0]) == n_ind]))
        return ais_locs

    def get_other_boat_locs(self, t):
        locs = [[np.interp(t, self.ais_locs[n2][:,0], self.ais_locs[n2][:,n]) for n in [1, 2]] 
                for n2 in range(len(self.ais_locs))]
        return locs

    class Playback:
        """ Playback of survey data
        Requires location of survey data, save_dir, 
        containing files for ACTIONS, META, OBSERVATIONS and TRUTH, 
        and a COVERAGE folder containing coverage maps
        """

        def __init__(self, save_dir):
            # load all data
            self.actions = self.load_textfiles(save_dir, 'ACTIONS')
            self.ep_end = self.actions[-1][0]
            self.truth = self.load_textfiles(save_dir, 'TRUTH')
            self.cov_map, self.cov_inds = self.load_covmaps(save_dir)
            self.observations = self.load_textfiles(save_dir, 'OBSERVATIONS')

        def get_data(self, action_id):
            print("Action ID: ", action_id)
            # get all the data as it would have been at that action

            # check whether there was a plan change at the desired action
            pc = [n for n in self.actions if n[1] == 'update' 
                  and n[0] == action_id]
            if pc: 
                agent_pos = pc[0][2:]
                intended_pos = [n for n in self.actions if n[1] == 'move'
                            if n[0] <= action_id][-1][2:]
            else: 
                agent_pos = [n for n in self.actions if n[1] == 'move'
                            if n[0] <= action_id][-1][2:]
                intended_pos = agent_pos
            agent_hist = np.array([n[2:] for n in self.actions if n[1] == 'move'
                        if n[0] <= action_id])           

            n_cov = [n for n in range(len(self.cov_inds))
                     if self.cov_inds[n] <= action_id][-1]
            cov_map = self.cov_map[:n_cov+1]
            t = [x[1] for x in self.observations
                 if x[0] <= action_id][-1]
            
            # Assemble the contacts
            contacts = [Detection(x[2],
                                  x[3],
                                  x[4],
                                  x[5],
                                  x[6],
                                  [],
                                  None) for x in self.observations
                        if x[0] <= action_id if len(x) > 2]
            # add contact clusters
            # get ungrouped groups
            ug_inds = [x[2] for x in self.actions
                       if x[0] <= action_id
                       if x[1] == 'ungroup']
            # add all grouped contacts
            c_inds = [x[3:] for x in self.actions
                      if x[0] <= action_id
                      if x[1] == 'group'
                      if x[2] not in ug_inds]
            g_n = 0
            for n in c_inds:
                for n2 in n:
                    contacts[n2].group_n = g_n
                g_n += 1
            N_g = g_n

            return t, agent_pos, intended_pos, cov_map, contacts, N_g, agent_hist

        def load_textfiles(self, save_dir, file_name):
            # find the file that matches
            dir = os.listdir(save_dir)
            f_dir = [x for x in dir if file_name in x][0]
            # open and read
            f = open(os.path.join(save_dir, f_dir))
            lines = f.readlines()
            # split out the lines
            f_out = [line.split(' ')[0:-1] for line in lines]
            # convert ints and floats
            data = [[int(item) if item.isdigit()
                    else float(item) if '.' in item
                    else item for item in line] for line in f_out]
            return data

        def load_covmaps(self, dir):
            f_dir = os.path.join(dir, 'COVERAGE')
            l_dir = os.listdir(f_dir)
            # get and sort all file names
            files = [filename.split('_') for filename in l_dir]
            f_nums = [[int(f) for f in fs if f.isdigit()][0] for fs in files]
            n_srt = sorted(range(len(f_nums)), key=lambda k: f_nums[k])
            # get all of the sorted coverage maps
            cov_map = [np.loadtxt(os.path.join(f_dir, l_dir[n])).tolist()
                       for n in n_srt]
            inds = sorted(f_nums)

            return cov_map, inds

    class CoverageMap:
        """_summary_
        """

        def __init__(self, params):
            # reset coverage map variable
            self.map_stack = []
            # unpack relevant parameters
            self.sa = params['scan_area_lims']
            self.sa = [int(x) for x in self.sa]
            self.ma = params['min_scan_angle_diff']
            self.li = params['leadinleadout']
            self.ms = params['min_scan_l']
            self.sw = params['scan_width']
            self.nw = params['nadir_width']

        def add_scan(self, x1, y1, x2, y2):
            # cov_temp: (cov_l,
            #            cov_r,
            #            cov_corners_l,
            #            cov_corners_r,
            #            orientation)
            cov_temp = self.pixelrect(x1 - self.sa[0],
                                      x2 - self.sa[0],
                                      y1 - self.sa[2],
                                      y2 - self.sa[2])

            # Compute scan angle
            ang_L_deg = np.rad2deg(cov_temp[4]+np.pi/2)
            ang_R_deg = np.rad2deg(cov_temp[4]-np.pi/2)
            # round to 0.1degrees
            ang_L = 0.1 * round(ang_L_deg/0.1)
            ang_R = 0.1 * round(ang_R_deg/0.1)
            # wrap to -180 180
            ang_L = (ang_L + 180) % (2*180) - 180
            ang_R = (ang_R + 180) % (2*180) - 180
            angs = (ang_L, ang_R)
            # Combine with mask
            cL_temp = cov_temp[0].copy()
            cL_temp[cL_temp == 0] = np.nan
            cR_temp = cov_temp[1].copy()
            cm = cL_temp*(ang_L)
            cm[cR_temp == 1] = ang_R

            # add to the stack of orientations
            self.map_stack.append(cm)
            # output the corners and the scan angles
            rect_corners = [cov_temp[2], cov_temp[3]]

            return rect_corners, angs

        def pixelrect(self, x1, x2, y1, y2):
            # compute length and rotation of rectangle
            rec_l = math.dist((x2, y2),
                              (x1, y1)) - 2*self.li
            rec_rot = np.arctan2(y2 - y1,
                                 x2 - x1)

            dy_sa = self.sa[3]-self.sa[2]
            dx_sa = self.sa[1]-self.sa[0]

            # Check that the length of scan is above the minimum scan length
            if rec_l > self.ms:
                # account for the leadinleadout length
                x1 += np.cos(rec_rot)*self.li
                y1 += np.sin(rec_rot)*self.li
                x2 -= np.cos(rec_rot)*self.li
                y2 -= np.sin(rec_rot)*self.li
                # compute scan
                dy = np.sin((np.pi/2) + rec_rot)*self.sw/2
                dx = np.cos((np.pi/2) + rec_rot)*self.sw/2
                # compute nadir
                dy_nad = np.sin((np.pi/2) + rec_rot)*self.nw/2
                dx_nad = np.cos((np.pi/2) + rec_rot)*self.nw/2
                # compute corners of scan
                rect_corners_1 = np.array([[x1+dx_nad, y1+dy_nad],
                                           [x1+dx, y1+dy],
                                           [x2+dx, y2+dy],
                                           [x2+dx_nad, y2+dy_nad]], np.int32)
                rect_corners_2 = np.array([[x1-dx, y1-dy],
                                           [x1-dx_nad, y1-dy_nad],
                                           [x2-dx_nad, y2-dy_nad],
                                           [x2-dx, y2-dy]], np.int32)
                # get the mask (flipped because image)
                rect_temp = np.flip(cv2.fillConvexPoly(np.zeros((dy_sa,
                                                                 dx_sa)),
                                                       rect_corners_1,
                                                       1), 0)
                rect_temp2 = np.flip(cv2.fillConvexPoly(np.zeros((dy_sa,
                                                                 dx_sa)),
                                                        rect_corners_2,
                                                        1), 0)
            else:
                # if too short, fill with empty data
                rect_temp = np.flip(np.ones((dy_sa,
                                             dx_sa))*np.nan, 0)
                rect_temp2 = np.flip(np.ones((dy_sa,
                                              dx_sa))*np.nan, 0)
                rect_corners_1 = np.array([[0, 0],
                                           [0, 0],
                                           [0, 0],
                                           [0, 0]], np.int32)
                rect_corners_2 = np.array([[0, 0],
                                           [0, 0],
                                           [0, 0],
                                           [0, 0]], np.int32)
            return rect_temp, rect_temp2, rect_corners_1, rect_corners_2, rec_rot

    class ContactDetections:
        def __init__(self, params):
            # initialise vars
            self.detections = []
            self.det_grp = []
            self.det_n = 0
            self.scan_n = 0
            self.group_n = 0
            # unpack relevant params
            self.lu = params['loc_uncertainty']
            self.sa = params['scan_area_lims']

        def dets_to_clus(self, c_inds):
            # add detections to cluster
            self.detections = self.add_det_group(self.detections,
                                                 self.group_n,
                                                 c_inds)

        def add_det_group(self, dets, group_n, c_inds):
            for n in c_inds:
                dets[n].group_n = group_n
            return dets

        def remove_group(self, grp_ind):
            for n in grp_ind:
                inds = [d.det_n for d in self.detections
                        if d.group_n == n]
                for n2 in inds:
                    self.detections[n2].group_n = None
                ind = [g.group_n for g in self.det_grp].index(n)
                self.det_grp.pop(ind)

        def add_group(self):
            inds = [d.det_n for d in self.detections
                    if d.group_n == self.group_n]
            if len(inds):
                # get average position of all selected points and increment
                grp_tmp = self.group_loc(self.detections,
                                         self.group_n)
                self.det_grp.append(grp_tmp)
                self.group_n += 1
            return self.group_n-1, inds

        def group_loc(self, detections, group_n):
            x_arr = [d.x for d in detections
                     if d.group_n == group_n]
            y_arr = [d.y for d in detections
                     if d.group_n == group_n]
            g_x = np.mean(x_arr)
            g_y = np.mean(y_arr)
            # get the list of angles
            ang_arr = [d.angle for d in detections
                       if d.group_n == group_n]
            # make new group
            grp_tmp = DetGroup(g_x,
                               g_y,
                               ang_arr,
                               group_n)
            return grp_tmp

        def add_dets(self, contacts, rect_corners, scan_angles):
            # output to recorder
            obs_str = []
            if np.array(rect_corners).any():
                # for each side of new scan
                for n in [0, 1]:
                    # check if inside scan area and detected
                    rect_c_tmp = rect_corners[n] + [self.sa[0],
                                                    self.sa[2]]
                    x_out, y_out = self.inside_rect(contacts,
                                                    rect_c_tmp,
                                                    scan_angles[n])
                    rng = 0
                    if x_out:
                        # add new detections
                        for n2 in range(len(x_out)):
                            # make new detection
                            det_temp = Detection(self.det_n,
                                                x_out[n2],
                                                y_out[n2],
                                                rng,
                                                scan_angles[n],
                                                self.scan_n,
                                                None)
                            self.detections.append(det_temp)
                            # output to recorder
                            obs_str.append([self.det_n,
                                            x_out[n2],
                                            y_out[n2],
                                            rng,
                                            scan_angles[n]])
                            self.det_n += 1
            self.scan_n += 1
            return obs_str

        def inside_rect(self, contacts, rect_corners, scan_angle):
            # initialise outputs
            x_out = []
            y_out = []

            # compute edge vectors
            AB = rect_corners[1] - rect_corners[0]
            BC = rect_corners[2] - rect_corners[1]
            CD = rect_corners[3] - rect_corners[2]
            DA = rect_corners[0] - rect_corners[3]

            for xy in list(contacts):
                # add some uncertainty to the location of the contact
                xy_temp = xy.location + np.random.normal(0,
                                                         self.lu,
                                                         size=(1, 2))[0]
                # make vectors from corners to point
                AP = xy_temp - rect_corners[0]
                BP = xy_temp - rect_corners[1]
                CP = xy_temp - rect_corners[2]
                DP = xy_temp - rect_corners[3]
                # compute the sign of the cross product for each corner
                ABCD_s = [np.sign(np.cross(AB, AP)),
                          np.sign(np.cross(BC, BP)),
                          np.sign(np.cross(CD, CP)),
                          np.sign(np.cross(DA, DP))]
                # if all signs the same, point is in rectangle
                if len(set(ABCD_s)) == 1 and rect_corners.any():
                    # check whether it's target or clutter
                    #if xy.obj_class == 'Target':
                    rng_tmp = xy.det_probs[1] - xy.det_probs[0]
                    wrapped_angdiff = np.arctan(np.tan(np.deg2rad(xy.orientation)
                                                - np.deg2rad(scan_angle)))
                    det_prob_temp = (np.abs(np.rad2deg(wrapped_angdiff))-1)/90
                    det_prob = (xy.det_probs[0]
                                + det_prob_temp*rng_tmp)

                    #elif xy.obj_class == 'Clutter':
                    #    det_prob = self.dpc
                    # check whether it gets detected
                    if det_prob > np.random.uniform(0, 1):
                        x_out.append(xy_temp[0])
                        y_out.append(xy_temp[1])
            return x_out, y_out

    class Timer:
        def __init__(self, params):
            # get params
            tl = params['time_lim']
            self.bs = params['agent_speed']
            self.time_remaining = tl
            self.time_elapsed = 0
            self.time_temp = tl

        def update_temp(self, xy1, xy2, xy0):
            self.time_temp  = (self.time_remaining
                          - self.elapsedtime(xy1, xy2)
                          - self.elapsedtime(xy2, xy0))
            #return t_rem_temp

        def update(self, xy1, xy2):
            t_temp = self.elapsedtime(xy1, xy2)
            self.time_remaining -= t_temp
            self.time_elapsed += t_temp

        def update_time(self, t):
            self.time_remaining -= t
            self.time_elapsed += t

        def elapsedtime(self, xy1, xy2):
            t_elapsed = math.dist(xy1,
                                  xy2)/self.bs
            return t_elapsed

    class Plotter:
        """_summary_
        """

        def __init__(self, params, map_img):
            # unpack parameters              
            ma = params['map_area_lims']
            gr = params['grid_res']
            sa = params['scan_area_lims']
            self.sa = [int(x) for x in sa]
            self.na = params['N_angles']
            self.nl = params['N_looks']
            self.msa = params['min_scan_angle_diff']
            self.bs = params['agent_start']
            self.sw = params['scan_width']
            self.li = params['leadinleadout']
            self.tl = params['time_lim']
            self.ms = params['min_scan_l']

            # set up empty plot
            self.fig, self.ax = plt.subplots()
            #self.ax.axis('equal')
            self.ax.set_xlim(xmin=ma[0],
                             xmax=ma[1])
            self.ax.set_ylim(ymin=ma[2],
                             ymax=ma[3])
            self.ax.grid(color='lightgray',
                         linestyle='-',
                         linewidth=0.2)
            self.ax.set_axisbelow(True)
            self.ax.set_xticks(np.arange(ma[0],
                                         ma[1], gr))
            self.ax.set_yticks(np.arange(ma[2],
                                         ma[3], gr))
            
            self.ax.set_xlabel('Easting, m')
            self.ax.set_ylabel('Northing, m')

            # get the map image and show
            if map_img.any():
                self.ax.imshow(map_img)

            # agent position
            self.agentpos, = self.ax.plot(self.bs[0], self.bs[1],
                                         marker="o",
                                         markersize=10,
                                         markeredgecolor="blue",
                                         markerfacecolor=[0.1, 0.6, 1],
                                         zorder=5)
            self.target_pos, = self.ax.plot([],
                                           [],
                                           marker="x",
                                            markersize=10,
                                            markeredgecolor="red",
                                            markerfacecolor="red",
                                            zorder=4)
            # agent track 
            # temp
            self.track_plt, = self.ax.plot(self.bs[0], self.bs[1],
                                           '--',
                                           color='lightgrey',
                                           zorder=4)
            # intended
            self.track_int_plt, = self.ax.plot(self.bs[0], self.bs[1], 
                                               ':', 
                                               color = 'white', 
                                               zorder=3)
            # history
            self.track_hist_plt,  = self.ax.plot(self.bs[0], self.bs[1], 
                                                 ':',
                                                 color='darkgrey',
                                                 zorder=3)

            # other boat locations
            self.aisplt = []

            # coverage map
            self.cov_plt = self.ax.imshow(np.zeros((self.sa[3]-self.sa[2],
                                                    self.sa[1]-self.sa[0],
                                                    3)),
                                          zorder=-1,
                                          extent=(self.sa[0], self.sa[1],
                                                  self.sa[2], self.sa[3]),
                                          interpolation='bilinear')
            # contacts
            self.det_plt, = self.ax.plot([],
                                         [],
                                         'o',
                                         markeredgecolor="black",
                                         markerfacecolor='white',
                                         picker=True,
                                         zorder=3)
            # contact scan directions
            self.detectionslineplt = []
            # contact clusters
            self.det_cls_plt, = self.ax.plot([],
                                             [],
                                             'o',
                                             markeredgecolor="black",
                                             markerfacecolor="grey",
                                             zorder=2)
            # contact groups
            self.det_grp_plt, = self.ax.plot([],
                                             [],
                                             'o',
                                             markersize=10,
                                             markeredgecolor="black",
                                             markerfacecolor="green",
                                             picker=True,
                                             zorder=4)
            # ground truth
            self.targ_plt, = self.ax.plot([], [],
                                          'X',
                                          markersize=8,
                                          markerfacecolor='White',
                                          markeredgecolor='black',
                                          zorder=5)
            # initialise patch for temp
            self.p_empty = self.make_patch(0, 0, 0, 0)
            self.p = self.ax.add_patch(self.p_empty)
            # add time to title
            self.ax.set_title("Time remaining: {:.0f} of {:.0f}secs".format(self.tl, self.tl))

            self.ax.figure.canvas.draw()

        def reset(self):
            self.updateagent([self.bs[0], self.bs[1]])
            self.cov_plt.set_data(np.zeros((self.sa[3]-self.sa[2],
                                            self.sa[1]-self.sa[0],
                                            3)))
            self.det_plt.set_data([], [])
            self.det_cls_plt.set_data([], [])
            self.det_grp_plt.set_data([], [])
            self.targ_plt.set_data([], [])
            self.target_pos.set_data([], [])
            self.track_int_plt.set_data([], [])
            self.track_hist_plt.set_data(self.bs[0], self.bs[1])
            self.updatetime(self.tl, self.tl)
            for p in self.detectionslineplt:
                p.remove()
            self.detectionslineplt = []

        def update_temp(self, x1, y1, x2, y2):
            # remove previous patch
            self.p.remove()
            # add track line
            self.track_plt.set_data([x1, x2], [y1, y2])
            # add new patch
            p_temp = self.make_patch(x1, x2, y1, y2)
            self.p = self.ax.add_patch(p_temp)
        
        def remove_temp(self):
            self.p.remove()
            self.p = self.ax.add_patch(self.p_empty)
            self.track_plt.set_data([0], [0])

        def updateagent(self, pos):            
            x, y = pos[0], pos[1]
            self.agentpos.set_data([x], [y])

        def updatetrackhist(self, xy_hist):
            self.track_hist_plt.set_data(xy_hist[:,0],
                                         xy_hist[:,1])

        def updatetarget(self, xy, xy0):            
            x, y = xy[0], xy[1]
            self.target_pos.set_data([x], [y])
            self.track_int_plt.set_data((xy0[0],xy[0]), (xy0[1],xy[1]))

        def updateaislocs(self, pos):
            l_p = len(pos)
            
            if self.aisplt:
                # remove previous plots
                for n in range (l_p):
                    self.aisplt[-1].remove()
                    self.aisplt.pop()
                
            for n in range(l_p):
                self.aisplt.append(self.ax.scatter(pos[n][0],
                                                pos[n][1],
                                                color='white',
                                                edgecolor='white',
                                                marker=(3,
                                                        0,
                                                        45),
                                                s=200,
                                                zorder=4))
                
        def updatecovmap(self, map_stack):
            m_x = len(map_stack[0][0])
            m_y = len(map_stack[0])
            self.map_hsv = np.zeros((m_y,
                                     m_x,
                                     3))
            if map_stack:
                # count the unique elements of each pixel
                map_or_rnd = self.msa * np.round(np.array(map_stack)/self.msa)
                map_or_rnd.sort(axis=0)
                m_dif = np.diff(map_or_rnd, axis=0) > 0
                map_or_cnt = m_dif.sum(axis=0)+1
                # count number of times
                map_cnt = np.count_nonzero(~np.isnan(map_stack), axis=0)
                # set the hue value
                hue_val = np.clip(270 - map_or_cnt*(120/self.na), 120, 250)
            else:
                hue_val = 250
            # make hsv image
            self.map_hsv[:, :, 0] = np.ones((m_y,
                                             m_x))*hue_val
            self.map_hsv[:, :, 1] = np.ones((m_y,
                                             m_x))
            self.map_hsv[:, :, 2] = np.clip(map_cnt/self.nl, 0, 1)
            # plot new coverage map
            self.cov_plt.set_data(cv2.cvtColor(np.float32(self.map_hsv),
                                               cv2.COLOR_HSV2BGR))

        def updatecontacts(self, detections):
            # detections not in groups
            dx_w = [n.x for n in detections if n.group_n == None]
            dy_w = [n.y for n in detections if n.group_n == None]
            self.det_plt.set_data(dx_w, dy_w)
            # detections in groups
            dx_g = [n.x for n in detections if n.group_n != None]
            dy_g = [n.y for n in detections if n.group_n != None]
            self.det_cls_plt.set_data(dx_g, dy_g)

            # work out how many more contacts have been added and add new ones
            l_d = len(detections)
            l_p = len(self.detectionslineplt)
            for n in range (l_p - l_d):
                self.detectionslineplt[-1].remove()
                self.detectionslineplt.pop()
                
            for n in range(l_d - l_p):
                self.detectionslineplt.append(plt.scatter(detections[l_p+n].x,
                                                          detections[l_p+n].y,
                                                          color='white',
                                                          edgecolor='white',
                                                          marker=(1,
                                                                  1,
                                                                  detections[l_p+n].angle-90),
                                                          s=300,
                                                          zorder=1))

        def updategroups(self, groups):
            # get group coordinates
            g_x = [n.x for n in groups]
            g_y = [n.y for n in groups]
            self.det_grp_plt.set_data(g_x, g_y)

        def updatetime(self, t, t_tmp):
            self.ax.set_title("Time remaining: {:.0f} of {:.0f}secs".format(t_tmp, t))

        def reveal_targets(self, contacts_t):
            n_targets = [i for i, x in enumerate(contacts_t, 1)
                         if x.obj_class == 'Target'][-1]
            x = []
            y = []
            for n in np.arange(n_targets):
                x.append(contacts_t[n].location[0])
                y.append(contacts_t[n].location[1])
            self.targ_plt.set_data(x, y)

        def make_patch(self, x1, x2, y1, y2):
            # compute length of path
            rec_l = np.clip(math.dist((x2, y2),
                            (x1, y1)) - 2*self.li, 0, None)
            if rec_l < self.ms:
                rec_l = 0

            # compute the rotation of the path
            rec_rot = np.arctan2(y2 - y1,
                                 x2 - x1)
            x1 += np.cos(rec_rot)*self.li
            y1 += np.sin(rec_rot)*self.li
            rec_rot_deg = np.rad2deg(rec_rot)
            y0 = y1 - np.sin((np.pi/2) + rec_rot)*self.sw/2
            x0 = x1 - np.cos((np.pi/2) + rec_rot)*self.sw/2

            return patches.Rectangle(xy=(x0, y0),
                                     width=rec_l,
                                     height=self.sw,
                                     angle=rec_rot_deg,
                                     #rotation_point=(x0, y0),
                                     alpha=0.5,
                                     zorder=1)

    class Logger:
        """_summary_
        """

        def __init__(self, params, gnd_trth, save_dir) -> None:
            # unpack params
            bs = params['agent_start']
            tl = params['time_lim']
            sa = params['scan_area_lims']
            self.save_dir = save_dir
            # initialse
            self.action_id = 0
            self.action_id_cov = []
            self.actions = []
            self.observations = []
            self.truth = []
            self.cov = []
            # add initial conditions
            self.addmove(bs[0],
                         bs[1])
            self.addcovmap(np.flip(np.ones((sa[3]-sa[2],
                                         sa[1]-sa[0]))*np.nan, 0))
            self.addtruth(gnd_trth)
            self.addobservation([], tl)
            self.addmeta(params)

        def addmove(self, x, y):
            self.actions.append(" ".join([str(self.action_id),
                                          "move",
                                          "{:.1f}".format(x),
                                          "{:.1f}".format(y),
                                          '\n']))
            self.action_id_cov.append(self.action_id)
            self.action_id += 1

        def addcovmap(self, cov_map):
            self.cov.append(cov_map)

        def addgroup(self, group_n, c_ind):
            c_ind.sort()
            c_str = " ".join([str(n) for n in c_ind])
            self.actions.append(" ".join([str(self.action_id),
                                          "group",
                                          str(group_n),
                                          c_str,
                                          '\n']))
            self.action_id += 1

        def ungroup(self, group_n):
            g_str = " ".join([str(n) for n in group_n])
            self.actions.append(" ".join([str(self.action_id),
                                          "ungroup",
                                          g_str,
                                          '\n']))
            self.action_id += 1

        def addplanchange(self, x, y): 
            self.actions.append(" ".join([str(self.action_id-1),
                                "update",
                                "{:.1f}".format(x),
                                "{:.1f}".format(y),
                                '\n']))

        def addobservation(self, obs, t):
            if obs:
                for ob in obs:
                    self.observations.append(" ".join([str(self.action_id-1),  # actionID
                                                       "{:.1f}".format(
                                                           t),  # time
                                                       # det ind
                                                       str(ob[0]),
                                                       "{:.1f}".format(
                        ob[1]),  # x
                        "{:.1f}".format(
                        ob[2]),  # y
                        "{:.1f}".format(
                        ob[3]),  # r
                        "{:.1f}".format(
                        ob[4]),  # th
                        "\n"]))
            else:
                self.observations.append(" ".join([str(self.action_id-1),
                                                   "{:.1f}".format(t),
                                                   "\n"]))

        def addtruth(self, contacts):
            for c in contacts:
                if c.obj_class == 'Target':
                    self.truth.append(' '.join([str(c.id),
                                                'target',
                                                str(c.location[0]),
                                                str(c.location[1]),
                                                str(c.orientation),
                                                '\n']))
                else:
                    self.truth.append(' '.join([str(c.id),
                                                'clutter',
                                                str(c.location[0]),
                                                str(c.location[1]),
                                                '\n']))

        def addmeta(self, params):
            self.metadata = []
            for s in params:
                if isinstance(params[s],str):
                    val = str('"'+params[s]+'"')
                else:
                    val = str(params[s])
                self.metadata.append('\n'+': '.join([s, val]))

        def save_data(self, ep_ID):
            ep_str = 'Episode' + str(ep_ID)
            fold_dir = os.path.join(self.save_dir, ep_str)

            # if the folder already exists, delete it
            if os.path.exists(os.path.join(fold_dir,'COVERAGE')):
                shutil.rmtree(fold_dir)
            
            # Make the new directory
            os.makedirs(os.path.join(fold_dir, 'COVERAGE'))
            # Write actions
            self.actions.append(" ".join([str(self.action_id),
                                          "end "
                                          '\n']))
            f_act = open(os.path.join(fold_dir, ep_str+'_ACTIONS'), 'w')
            f_act.writelines(self.actions)
            # write observations
            f_obs = open(os.path.join(fold_dir, ep_str+'_OBSERVATIONS'), 'w')
            f_obs.writelines(self.observations)
            # write ground truth and metadata
            f_truth = open(os.path.join(fold_dir, ep_str+'_TRUTH'), 'w')
            f_truth.writelines(self.truth)
            f_meta = open(os.path.join(fold_dir, ep_str+'_META'), 'w')
            f_meta.writelines(self.metadata)
            # output coverage matrices
            for n in range(len(self.cov)):
                np.savetxt(os.path.join(fold_dir, 'COVERAGE',
                                        ep_str+"_"+str(self.action_id_cov[n])+"_COVERAGE"),
                           self.cov[n])

    # mouse and keyboard functions
    def mouse_move(self, event):
        # check it's inside the axes
        if event.inaxes != self.plotter.ax.axes:
            return
        # grab the x,y coords
        x_i, y_i = event.xdata, event.ydata
        if not self.end_episode and self.groupswitch:
            x0, y0 = self.agent_pos[0], self.agent_pos[1]
            if self.snaptoangle:
                x, y = self.round_to_angle(x0, y0, x_i, y_i)
            else:
                x, y = x_i, y_i
            self.xy_temp = (x,y)
            st_pos = self.params['agent_start'] 
            self.timer.update_temp((x0, y0), (x, y), st_pos)
            self.plotter.update_temp(x0, y0,
                                     x, y)
            self.plotter.updatetime(self.timer.time_remaining,
                                    self.timer.time_temp)
            if not self.play or not self.rt:
                self.plotter.fig.canvas.draw_idle()

    def on_key_manual(self, event):
        # normal operation if episode is ongoing
        if not self.end_episode:
            if event.key == " ":
                self.play = not self.play
            if event.key == "up":
                # increase playback speed
                self.play_speed *= 2
                print('play speed = ' + str(self.play_speed))
                self.compute_movement_step()
            if event.key == "down": 
                # decrease playback speed
                self.play_speed *= 0.5
                print('play speed = ' + str(self.play_speed))
                self.compute_movement_step()

            if event.key == "z":
                self.snaptoangle = not self.snaptoangle
            if event.key == 'shift':
                self.plotter.remove_temp()
                self.groupswitch = not self.groupswitch

            elif event.key == 'a':
                # group the selected
                g_n, c_inds = self.contacts.add_group()
                # only add if there's any points in the group
                if c_inds:
                    # update plot
                    self.plotter.updategroups(self.contacts.det_grp)
                    # self.plotter.fig.canvas.draw()
                    # log new grouping
                    self.logger.addgroup(g_n, c_inds)

            elif event.key == '#':
                self.plotter.reveal_targets(self.contacts_t)
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
            self.move_complete = True
            self.move_req = False
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
        self.plotter.updateagent([bp[0], bp[1]])
        self.plotter.updatetrackhist(ah)
        self.plotter.updatetarget(ip,ip)
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
            x0, y0 = self.agent_pos[0], self.agent_pos[1]

            # check if the requested position is in an occupied space
            if not(self.map_mask.any()) or not self.map_mask[round(y_i), round(x_i)]:
                # check that the path doesn't go through occupied spaces
                cp = self.check_path(x0,y0,x_i,y_i)
                if cp:
                    if self.snaptoangle:
                        x, y = self.round_to_angle(x0, y0, x_i, y_i)
                    else:
                        x, y = x_i, y_i

                    # handle if the previous move request wasn't completed
                    if not self.move_complete:
                        self.updatescans()
                    if not self.play:
                        self.play = True
                    self.add_newxy(x, y)

                else:
                    self.plotter.p.set_facecolor((1, 1, 1))
                    self.plotter.fig.canvas.draw_idle()
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
            if not self.rt:
                self.plotter.fig.canvas.draw_idle()


@dataclass
class TargetObject:
    id: float
    obj_class: str
    location: float
    orientation: float
    det_probs: list


@dataclass
class Detection:
    det_n: int
    x: float
    y: float
    range: float
    angle: float
    scan_n: int
    group_n: int


@dataclass
class DetGroup:
    x: float
    y: float
    angles: list
    group_n: int


if __name__ == '__main__':
    ss = SurveySimulation()
