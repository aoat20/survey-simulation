import math
import os
import random
from itertools import permutations

import matplotlib.pyplot as plt
from matplotlib import cm


class Agent:

    def __init__(self,
                 xy_start=[0, 0],
                 course=None,
                 speed=0,
                 scan_thr=0):
        #
        self.speed0 = speed
        self.course0 = course
        self.course = course
        self.xy_start_candidates = xy_start

        self.xy = self.xy_st_pos()
        self.xy_hist = [np.array(self.xy)]
        self.xy_step = [0, 0]
        self.destination = self.xy
        self.distance_dest = np.inf
        self.distance_travelled = 0

        # If the course is not set, don't move
        if course is None:
            self.speed = 0
        else:
            self.speed = speed
            self.compute_movement_step()

        # scan straightness threshold and index
        self.ind0 = 0
        self.scan_thr = scan_thr

    def move_to_position(self,
                         xy):
        # Move immediately to a position
        self.xy = xy
        self.xy_hist = np.append(self.xy_hist,
                                 [xy],
                                 axis=0)
        self.check_path_straightness()

    def advance_one_step(self,
                         t_elapsed):

        self.distance_travelled += self.speed*t_elapsed
        # if the agent has reached the destination, stop
        if self.distance_travelled >= self.distance_dest:
            self.set_speed(0)
            self.xy = self.destination

        self.xy = [self.xy[0]+self.xy_step[0]*t_elapsed,
                   self.xy[1]+self.xy_step[1]*t_elapsed]

        self.xy_hist = np.append(self.xy_hist,
                                 [np.array(self.xy)],
                                 axis=0)

    def destination_req(self,
                        xy):
        # start of leg
        xy0 = self.xy_hist[-1]
        # end of leg
        self.destination = xy
        # distance to the destination
        self.distance_dest = np.sqrt((xy[0]-xy0[0])**2
                                     + (xy[1]-xy0[1])**2)
        # compute course and set
        course = np.rad2deg(np.arctan2(xy[0]-xy0[0],
                                       xy[1]-xy0[1]))
        self.set_speedandcourse(self.speed0, course)
        self.distance_travelled = 0

    def compute_movement_step(self):
        # Compute the step in x and y based on the course and speed
        course_rad = np.deg2rad(self.course)
        self.xy_step = [self.speed*np.sin(course_rad),
                        self.speed*np.cos(course_rad)]

    def set_speed(self, speed):
        # set the new speed and compute the new step
        self.speed = speed
        self.compute_movement_step()

    def set_course(self, course):
        # set the new course and compute the new step
        self.course = course
        self.compute_movement_step()

    def set_speedandcourse(self, speed, course):
        # set both speed and course
        self.speed = speed
        self.course = course
        self.compute_movement_step()

    def get_current_path_len(self):
        # return the number of times agent has moved in a straight line
        p_st = self.ind0
        p_end = len(self.xy_hist)
        path_n = p_end - p_st

        return path_n

    def check_path_straightness(self):
        # Check whether path is straight
        # Get all points being conisidered
        points = self.xy_hist[self.ind0:]
        # first point
        p1 = points[0]
        # most recent point
        p2 = points[-1]
        d = []
        # Check there's at least 3 points
        if len(points) > 2:
            # Get all linear distances from the straight line between p1 and p2
            for xy in points[1:-1]:
                d.append(np.abs(np.cross(p2-p1,
                                         p1-xy))/np.linalg.norm(p2-p1))
            # Ensure it's still going in the same directions with signs
            dxdy = np.sign(np.diff(self.xy_hist[-3:], 1, 0))
            if max(d) > self.scan_thr or any(dxdy[0, :] != dxdy[1, :]):
                ind0_out = self.ind0
                self.ind0 = len(self.xy_hist)-2
                return ind0_out
            else:
                return None
        else:
            return None

    def xy_st_pos(self):
        if type(self.xy_start_candidates[0]) is float:
            xy_st = self.xy_start_candidates
        else:
            rnd_n = random.randint(0, len(self.xy_start_candidates[0])-1)
            xy_st = [self.xy_start_candidates[1][rnd_n],
                     self.xy_start_candidates[0][rnd_n]]

        return xy_st

    def reset(self,
              new_st_pos=[]):

        self.xy_start_candidates = new_st_pos

        self.course = self.course0
        if self.course0 is None:
            self.speed = 0
        else:
            self.speed = self.speed0
            self.compute_movement_step()

        self.xy = self.xy_st_pos()
        self.xy_hist = [np.array(self.xy)]
        self.xy_leg0 = self.xy
        self.destination = [0, 0]
        self.distance_dest = np.inf
        self.distance_travelled = 0
        self.ind0 = 0


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
        # print("Action ID: ", action_id)
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

    def get_next_step(self, action_id):
        # print("Action ID: ", action_id)
        # get all the data as it would have been at that action

        agent_pos = [n for n in self.actions if n[1] == 'move'
                     if n[0] <= action_id][-1][2:]
        intended_pos = agent_pos
        agent_hist = np.array([n[2:] for n in self.actions if n[1] == 'move'
                               if n[0] <= action_id])

        if action_id in self.cov_inds:
            cov_map = np.array(self.cov_map[self.cov_inds.index(action_id)])
        else:
            cov_map = []

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


class Timer:
    def __init__(self,
                 time_lim=800,
                 t_step=1):
        # initialise vars
        self.time_lim = time_lim
        self.time_remaining = time_lim
        self.time_elapsed = 0
        self.time_temp = time_lim
        self.t_step = t_step

    def update_temp(self, xy1, xy2, xy0, agent_spd):
        # update the temporary time
        self.time_temp = (self.time_remaining
                          - self.elapsedtime(xy1, xy2, agent_spd)
                          - self.elapsedtime(xy2, xy0, agent_spd))

    def update(self, xy1, xy2, agent_spd):
        # update remaining time based on distance travelled
        t_temp = self.elapsedtime(xy1, xy2, agent_spd)
        self.time_remaining -= t_temp
        self.time_elapsed += t_temp

    def update_time(self, t):
        # update time
        self.time_remaining -= t
        self.time_elapsed += t

    def elapsedtime(self, xy1, xy2, agent_spd):
        # compute travel time
        t_elapsed = math.dist(xy1,
                              xy2)/agent_spd
        return t_elapsed

    def reset(self):
        self.time_remaining = self.time_lim
        self.time_elapsed = 0
        self.time_temp = self.time_lim


class Logger:
    """_summary_
    """

    def __init__(self,
                 agent_start,
                 time_lim,
                 map_lims,
                 params,
                 gnd_trth,
                 save_dir,
                 map_img) -> None:

        self.save_dir = save_dir
        # initialse
        self.action_id = 0
        self.action_id_cov = []
        self.actions = []
        self.observations = []
        self.truth = []
        self.cov = []
        self.aux_info = []
        # add initial conditions
        self.add_move(agent_start)
        self.add_covmap(np.ones((map_lims[3],
                                map_lims[1]))*np.nan)
        self.add_truth(gnd_trth)
        self.add_observation([], time_lim)
        self.add_meta(params)
        self.add_map(map_img)

    def add_move(self, xy):
        x, y = xy[0], xy[1]
        self.actions.append(" ".join([str(self.action_id),
                                      "move",
                                      "{:.1f}".format(x),
                                      "{:.1f}".format(y),
                                      '\n']))
        self.action_id += 1

    def add_covmap(self, cov_map):
        self.action_id_cov.append(self.action_id-1)
        self.cov.append(cov_map)

    def add_group(self, group_n, c_ind):
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

    def add_observation(self, obs, t):
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

    def add_truth(self, contacts):
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

    def add_meta(self, params):
        self.metadata = []
        for s in params:
            if isinstance(params[s], str):
                val = str('"'+params[s]+'"')
            else:
                val = str(params[s])
            self.metadata.append('\n'+': '.join([s, val]))

    def add_auxinfo(self, new_aux_info):
        self.aux_info.append(new_aux_info + '\n')

    def add_map(self, map_img):
        self.im = Image.fromarray(map_img)

    def save_data(self, ep_ID):

        if not ep_ID:
            i = 0
            while os.path.exists(os.path.join(self.save_dir,
                                              f"Episode{i}",
                                              "COVERAGE")):
                i += 1
            ep_n = i
        else:
            ep_n = ep_ID
        print('Saving output to Episode '+str(ep_n))

        ep_str = 'Episode' + str(ep_n)
        fold_dir = os.path.join(self.save_dir, ep_str)

        # if the folder already exists, delete it
        if os.path.exists(os.path.join(fold_dir, 'COVERAGE')):
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

        # save auxiliary information
        f_aux = open(os.path.join(fold_dir, ep_str+'_AUX'), 'w')
        f_aux.writelines(self.aux_info)

        # Save map
        map_dir = os.path.join(fold_dir, ep_str+'_MAP.png')
        self.im.save(map_dir)

    def reset(self,
              agent_start,
              new_contactstrth,
              map_lims,
              time_lim):
        # initialse
        self.action_id = 0
        self.action_id_cov = []
        self.actions = []
        self.observations = []
        self.truth = []
        self.cov = []
        # add initial conditions
        self.add_move(agent_start)
        self.add_covmap(np.ones((map_lims[3],
                                map_lims[1]))*np.nan)
        self.add_truth(new_contactstrth)
        self.add_observation([], time_lim)


class Map:

    """ summary
    """

    def __init__(self,
                 map_n="",
                 map_path="",
                 map_lims=[0, 100, 0, 100],
                 random_map=False):

        self.ag_st_mode = 0
        self.map_lims = map_lims

        self.map_path = map_path

        # if map_n or map path exists, load map and parameters
        if map_n:
            self.map_n = map_n
            self.map_path = 'maps/Map'+str(map_n)+'.png'
        elif random_map:
            self.map_path = '-1'

        self.setup()

    def setup(self):
        if self.map_path == '':
            self.occ = np.zeros((self.map_lims[3],
                                 self.map_lims[1]))
            self.img = np.zeros((self.map_lims[3],
                                 self.map_lims[1]))
        else:
            if self.map_path != '-1':
                im = Image.open(self.map_path)
            else:
                im = self.generate_random_map()
            img = np.asarray(im)
            # Check the alpha channel for transparency
            img_tmp = img[:, :, 3]
            self.map_lims = (0, img_tmp.shape[1],
                             0, img_tmp.shape[0])
            self.occ = np.where(img_tmp == 0, 0, 1)
            self.img = img

    def is_occupied(self, xy):
        ml = self.map_lims
        xy_rnd = np.int16(np.floor(xy))
        # check whether the coordinate is in an occupied
        if xy[0] < ml[0] or xy[0] > ml[1] or xy[1] < ml[2] or xy[1] > ml[3]:
            return 1
            raise Exception(
                "agent position is not within the bounds of the map")
        elif self.occ[xy_rnd[1], xy_rnd[0]]:
            return 1
            raise Exception(
                "agent position is an occupied coordinate of the map")

    def get_agent_start(self):
        if self.ag_st_mode == 0:
            ag_st = self.default_start()
        elif self.ag_st_mode == 1:
            ag_st = self.agent_st
        elif self.ag_st_mode == 2:
            ag_st = self.random_start()
        elif self.ag_st_mode == 3:
            ag_st = self.random_start_edges()

        return ag_st

    def default_start(self):
        # self.ag_st_mode = 0
        # default positions for the agent based on the map
        def_starts = ((58., 192.),
                      (33., 56.),
                      (200., 175.))
        try:
            return def_starts[self.map_n-1]
        except:
            raise Exception("You need to specify an agent_start position")

    def random_start(self):
        # self.ag_st_mode = 2
        coords_unocc = np.where(self.occ == 0)
        return coords_unocc

    def random_start_edges(self):
        # self.ag_st_mode = 3
        dxy_surr = np.unique([xy for xy in permutations([-1, -1, 0, 1, 1], 2)],
                             axis=0)
        coords_unocc = np.transpose(np.where(self.occ == 0))
        coords_edge = []
        for xy in coords_unocc:
            # if it's not at the edge of the map already
            if (xy[0] > self.map_lims[0]+2
                    and xy[0] < self.map_lims[1]-2
                    and xy[1] > self.map_lims[2]+2
                    and xy[1] < self.map_lims[3]-2):
                # check each surrounding pixel for land
                xy_surr = xy+dxy_surr
                if any([self.occ[x, y] for x, y in xy_surr]):
                    coords_edge.append(xy)

        coords_edge_np = np.transpose(np.array(coords_edge))

        return coords_edge_np

    def check_path(self,
                   x1i,
                   y1i,
                   x2i,
                   y2i):
        # round the coordinates
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
            xout, yout = x1 + x*xx + y*yx, y1 + x*xy + y*yy
            if p >= 0:
                y += 1
                p -= 2*dx
            p += 2*dy

            if self.occ[yout, xout]:
                return 0
        return 1

    def generate_perlin_noise_2d(self,
                                 shape,
                                 res):
        """Generate a 2D numpy array of perlin noise.

        Args:
            shape: The shape of the generated array (tuple of two ints).
                This must be a multple of res.
            res: The number of periods of noise to generate along each
                axis (tuple of two ints). Note shape must be a multiple of
                res.

        Returns:
            A numpy array of shape shape with the generated noise.

        Raises:
            ValueError: If shape is not a multiple of res.
        """
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
            .transpose(1, 2, 0) % 1
        # Gradients
        angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        # Change to true if it needs to be tileable
        if False:
            gradients[-1, :] = gradients[0, :]
        if False:
            gradients[:, -1] = gradients[:, 0]
        gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
        g00 = gradients[:-d[0], :-d[1]]
        g10 = gradients[d[0]:, :-d[1]]
        g01 = gradients[:-d[0], d[1]:]
        g11 = gradients[d[0]:, d[1]:]
        # Ramps
        n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
        n10 = np.sum(np.dstack((grid[:, :, 0]-1, grid[:, :, 1])) * g10, 2)
        n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1]-1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:, :, 0]-1, grid[:, :, 1]-1)) * g11, 2)
        # Interpolation
        t = grid*grid*grid*(grid*(grid*6 - 15) + 10)
        n0 = n00*(1-t[:, :, 0]) + t[:, :, 0]*n10
        n1 = n01*(1-t[:, :, 0]) + t[:, :, 0]*n11
        return np.sqrt(2)*((1-t[:, :, 1])*n0 + t[:, :, 1]*n1)

    def is_valid(self, x, y, matrix, visited, m, n):
        if (x < n and y < m and x >= 0 and y >= 0):
            if (visited[x][y] == False and matrix[x][y] == 1):
                return True
            else:
                return False
        else:
            return False

    def find_blob_size(self, matrix: np.array):
        dx = [0, 1, -1, 0]
        dy = [1, 0, 0, -1]
        m, n = matrix.shape

        # stores information about  which cell
        # are already visited in a particular BFS
        visited = [[False for i in range(m)] for j in range(n)]

        # Stores the final result grid
        result = [[0 for i in range(m)] for j in range(n)]

        # Stores the count of cells in
        # the largest connected component
        cnt = 0

        mp = {}
        mass_n = 0

        # Iterate over every cell
        for i in range(n):
            for j in range(m):
                if (visited[i][j] == False and matrix[i][j] == 1):
                    cnt = 0
                    mass_n += 1

                    # Stores the indices of the matrix cells
                    q = []

                    # Mark the starting cell as visited
                    # and push it into the queue
                    q.append([i, j])
                    visited[i][j] = True

                    # Iterate while the queue
                    # is not empty
                    while (len(q) != 0):
                        p = q[0]
                        q = q[1:]
                        x = p[0]
                        y = p[1]
                        cnt += 1

                        # Go to the adjacent cells
                        for i in range(4):
                            newX = x + dx[i]
                            newY = y + dy[i]

                            if (self.is_valid(newX, newY, matrix, visited, m, n)):
                                q.append([newX, newY])
                                visited[newX][newY] = True
                                result[newX][newY] = mass_n
                    mp[mass_n] = cnt
        return mp, np.array(result)

    def generate_random_map(self,
                            width=256,
                            height=256,
                            res=2,
                            depth=0.3,
                            crop=False,
                            show_map=False,
                            min_area=8000):

        # Example usage
        success = False
        while not success:
            p_noise = self.generate_perlin_noise_2d((height, width),
                                                    (res, res))
            p_temp = p_noise > depth

            cnt_mp, cnt_area = self.find_blob_size(p_temp)
            if len(cnt_mp) > 0:
                cnt_mx = max(cnt_mp, key=cnt_mp.get)
                if (cnt_mp[cnt_mx]) > min_area:
                    success = True
                    mp_tmp = np.ones(p_temp.shape)
                    # Get just the biggest area
                    mp_tmp[cnt_area == cnt_mx] = np.nan
                    if crop:
                        # Crop the rest of the image out
                        mp_mm_i = np.where(np.isnan(mp_tmp))
                        x_min = min(mp_mm_i[0])
                        x_max = max(mp_mm_i[0])
                        y_min = min(mp_mm_i[1])
                        y_max = max(mp_mm_i[1])
                        mp_out = mp_tmp[x_min:x_max, y_min:y_max]
                    else:
                        mp_out = mp_tmp

                    # Convert it to a green image
                    im = Image.fromarray(cm.Greens(mp_out, bytes=True),
                                         mode='RGBA')
                    if show_map:
                        im.show()
        return im


class GriddedData:

    def __init__(self,
                 map_lims: list[int],
                 angle_diff: float,
                 occ_map) -> None:
        # Initialise agent grid
        self.agent = np.zeros((map_lims[3],
                               map_lims[1]),
                              dtype=int)
        self.occ_map = occ_map

        # Initialise coverage map
        self.cov_map = [np.zeros((map_lims[3],
                                  map_lims[1]),
                                 dtype=int)]
        self.ang_diff = angle_diff
        self.bins = np.arange(0, 360, self.ang_diff)  # - self.ang_diff/2
        for _ in self.bins:
            self.cov_map.append(np.zeros((map_lims[3],
                                         map_lims[1]),
                                         dtype=int))

        # Initialise contact grid
        self.cts = np.zeros((map_lims[3],
                             map_lims[1]),
                            dtype=int)

    def add_cov_map(self, cov_map):
        # Get the full count of the scans on each pixel
        cov_temp = self.cov_map[0]
        cov_cnt = (~np.isnan(cov_map)).astype(int)
        self.cov_map[0] = cov_temp + cov_cnt

        # Get each angular count
        b_inds = np.digitize((cov_map + self.ang_diff/2) % 360, self.bins)
        b_inds[np.isnan(cov_map)] = -1
        for b in enumerate(self.bins):
            cov_temp = self.cov_map[b[0]+1]
            ang_cov_cnt = (b_inds-1 == b[0]).astype(int)
            self.cov_map[b[0]+1] = cov_temp + ang_cov_cnt

    def remove_cov_map(self, cov_map):
        # Get the full count of the scans on each pixel
        cov_temp = self.cov_map[0]
        cov_cnt = (~np.isnan(cov_map)).astype(int)
        self.cov_map[0] = cov_temp - cov_cnt

        # Get each angular count
        b_inds = np.digitize(cov_map, self.bins)
        b_inds[np.isnan(cov_map)] = 0
        for b in enumerate(self.bins):
            cov_temp = self.cov_map[b[0]+1]
            ang_cov_cnt = (b_inds == b[0]+1).astype(int)
            self.cov_map[b[0]+1] = cov_temp - ang_cov_cnt

    def add_agent_pos(self, agent_xy):
        ag_pos_rnd = np.int16(np.floor(agent_xy))
        self.agent = np.zeros(self.agent.shape,
                              dtype=int)
        if ag_pos_rnd[0] > 0 \
           and ag_pos_rnd[1] > 0 \
           and ag_pos_rnd[0] < self.occ_map.shape[1] \
           and ag_pos_rnd[1] < self.occ_map.shape[0]:
            self.agent[ag_pos_rnd[1],
                       ag_pos_rnd[0]] = 1

    def add_occ_map(self):
        pass

    def add_contacts(self,
                     contacts):
        base = 1
        for cts in contacts:
            xy = [cts.x, cts.y]
            c_xy_rnd = np.int16(base * np.round(np.divide(xy, base)))
            self.cts[c_xy_rnd[1],
                     c_xy_rnd[0]] += 1

    def remove_contacts(self,
                        contacts):
        base = 1
        for n in range(len(contacts)):
            cts = contacts[n]
            xy = [cts.x, cts.y]
            c_xy_rnd = np.int16(base * np.round(np.divide(xy, base)))
            # Check whether the contacts are on the map
            if c_xy_rnd[0] > 0 \
                    and c_xy_rnd[1] > 0 \
                    and c_xy_rnd[0] < self.occ_map.shape[1] \
                    and c_xy_rnd[1] < self.occ_map.shape[0]:
                self.cts[c_xy_rnd[1],
                         c_xy_rnd[0]] -= 1

    def reset(self,
              agent_xy):
        # Initialise agent grid
        self.add_agent_pos(agent_xy)

        # Initialise coverage map
        self.cov_map = [np.zeros(self.cov_map[0].shape,
                                 dtype=int) for _ in range(len(self.cov_map))]

        # Initialise contact grid
        self.cts = np.zeros(self.cts.shape,
                            dtype=int)
