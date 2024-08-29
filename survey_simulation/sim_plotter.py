import matplotlib.pylab as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb
import numpy as np
import math

class Plotter:
    """_summary_
    """

    def show(self,
             blocking=True):
        if not blocking:
            plt.ion()
        plt.show()

    def draw(self):
        self.fig.canvas.draw_idle()
        plt.pause(0.0001)

    def setup_plot(self, 
              map_lims):
        # set up empty plot
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(xmin=map_lims[0],
                            xmax=map_lims[1])
        self.ax.set_ylim(ymin=map_lims[2],
                            ymax=map_lims[3])
        self.ax.grid(color='lightgray',
                        linestyle='-',
                        linewidth=0.2)
        self.ax.set_axisbelow(True)        
        self.ax.set_xlabel('Easting, m')
        self.ax.set_ylabel('Northing, m')

    def setup_map(self, map_img):
        # get the map image and show
        if map_img.any():
            self.ax.imshow(map_img)

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
            
    def setup_covmap(self, 
                     scan_lims, 
                     min_scan_ang,
                     n_angles, 
                     n_looks):
        self.min_scan_ang = min_scan_ang
        self.n_angles = n_angles
        self.n_looks = n_looks

        # coverage map
        self.cov_plt = self.ax.imshow(np.zeros((scan_lims[3]-scan_lims[2],
                                                scan_lims[1]-scan_lims[0],
                                                3)),
                                        zorder=-1,
                                        extent=(scan_lims[0], scan_lims[1],
                                                scan_lims[2], scan_lims[3]),
                                        interpolation='bilinear')

    def updatecovmap(self, 
                     map_stack):
        
        m_x = len(map_stack[0][0])
        m_y = len(map_stack[0])
        map_hsv = np.zeros((m_y,
                            m_x,
                            3))
        if map_stack:
            # count the unique elements of each pixel
            map_or_rnd = self.min_scan_ang*np.round(np.array(map_stack)/self.min_scan_ang)
            map_or_rnd.sort(axis=0)
            m_dif = np.diff(map_or_rnd, axis=0) > 0
            map_or_cnt = m_dif.sum(axis=0)+1
            # count number of times
            map_cnt = np.count_nonzero(~np.isnan(map_stack), axis=0)
            # set the hue value
            hue_val = np.clip(270 - map_or_cnt*(120/self.n_angles), 120, 250)
        else:
            hue_val = 250
        # make hsv image
        map_hsv[:, :, 0] = np.ones((m_y,
                                    m_x))*hue_val
        map_hsv[:, :, 1] = np.ones((m_y,
                                    m_x))
        map_hsv[:, :, 2] = np.clip(map_cnt/self.n_looks, 0, 1)
        # plot new coverage map

        # convert hsv to rgb
        map_rgb = hsv_to_rgb(map_hsv)
                
        self.cov_plt.set_data(map_rgb)

    def setup_contacts(self):
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
                                        markersize=20,
                                        markerfacecolor='White',
                                        markeredgecolor='black',
                                        zorder=5)

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
            self.detectionslineplt.append(self.ax.scatter(detections[l_p+n].x,
                                                        detections[l_p+n].y,
                                                        color='white',
                                                        edgecolor='white',
                                                        marker=(1,
                                                                1,
                                                                detections[l_p+n].angle),
                                                        s=300,
                                                        zorder=1))

    def updategroups(self, groups):
        # get group coordinates
        g_x = [n.x for n in groups]
        g_y = [n.y for n in groups]
        self.det_grp_plt.set_data(g_x, g_y)

    def updatetime(self, t, t_tmp):
        self.ax.set_title("Time remaining: {:.0f} of {:.0f}secs".format(t_tmp, t))

    def updateps(self, playspeed):
        self.ax.set_title("Playspeed: {:.0f}x".format(playspeed))

    def reveal_targets(self, contacts_t):
        n_targets = [i for i, x in enumerate(contacts_t, 1)
                        if x.obj_class == 'Target'][-1]
        x = []
        y = []
        for n in np.arange(n_targets):
            x.append(contacts_t[n].location[0])
            y.append(contacts_t[n].location[1])
        self.targ_plt.set_data(x, y)

    def setup_scantemp(self, 
                       leadinleadout,
                        min_scan_l,
                        scan_width):
        self.p_empty = self.make_patch(0, 0, 0, 0, 
                                        leadinleadout,
                                        min_scan_l,
                                        scan_width)
        self.p = self.ax.add_patch(self.p_empty)  
        # temp
        self.track_plt, = self.ax.plot([0], [0],
                                            '--',
                                            color='grey',
                                            zorder=4)

    def update_temp(self, 
                    x1, 
                    y1, 
                    x2, 
                    y2, 
                    leadinleadout,
                    min_scan_l,
                    scan_width):
        # remove previous patch
        self.p.remove()
        # add track line
        self.track_plt.set_data([x1, x2], [y1, y2])
        # add new patch
        p_temp = self.make_patch(x1, x2, y1, y2, 
                                leadinleadout,
                                min_scan_l,
                                scan_width)
        self.p = self.ax.add_patch(p_temp)
    
    def remove_temp(self):
        self.p.remove()
        self.p = self.ax.add_patch(self.p_empty)
        self.track_plt.set_data([0], [0])

    def make_patch(self, 
                   x1, 
                   x2, 
                   y1, 
                   y2, 
                   leadinleadout,
                   min_scan_l,
                   scan_width):
        # compute length of path
        rec_l = np.clip(math.dist((x2, y2),
                        (x1, y1)) - 2*leadinleadout, 0, None)
        if rec_l < min_scan_l:
            rec_l = 0

        # compute the rotation of the path
        rec_rot = np.arctan2(y2 - y1,
                                x2 - x1)
        x1 += np.cos(rec_rot)*leadinleadout
        y1 += np.sin(rec_rot)*leadinleadout
        rec_rot_deg = np.rad2deg(rec_rot)
        y0 = y1 - np.sin((np.pi/2) + rec_rot)*scan_width/2
        x0 = x1 - np.cos((np.pi/2) + rec_rot)*scan_width/2

        return patches.Rectangle(xy=(x0, y0),
                                    width=rec_l,
                                    height=scan_width,
                                    angle=rec_rot_deg,
                                    #rotation_point=(x0, y0),
                                    alpha=0.5,
                                    zorder=1)

    class AgentPlot:
        def __init__(self, 
                     ax, 
                     xy0, 
                     speed = [], 
                     course = [],
                     col='blue'):
            # store initial position
            self.xy0 = xy0
            # agent position
            self.agentpos, = ax.plot(xy0[0], xy0[1],
                                            marker='o',
                                            markersize=10,
                                            markeredgecolor='blue',
                                            markerfacecolor=col,
                                            zorder=5)
            

            self.target_pos, = ax.plot([],
                                            [],
                                            marker="x",
                                            markersize=10,
                                            markeredgecolor="red",
                                            markerfacecolor="red",
                                            zorder=4)
            # agent track 

            # intended
            self.track_int_plt, = ax.plot(xy0[0], xy0[1], 
                                                ':', 
                                                color = 'lightgrey', 
                                                zorder=3)
            # history
            self.track_hist_plt,  = ax.plot(xy0[0], xy0[1], 
                                                    ':',
                                                    color='darkgrey',
                                                    zorder=3)

            if not speed == []:
                self.txtlbl = ax.annotate(f'Speed:{speed:.2f} \nCourse:{course:.0f}', 
                                        xy0)
            
        def updateagent(self, xy):            
            x, y = xy[0], xy[1]
            self.agentpos.set_data([x], [y])

        def updatetrackhist(self, xy_hist):
            self.track_hist_plt.set_data(xy_hist[:,0],
                                         xy_hist[:,1])

        def updatetarget(self, xy, xy0):            
            x, y = xy[0], xy[1]
            self.target_pos.set_data([x], [y])
            self.track_int_plt.set_data((xy0[0],xy[0]), 
                                        (xy0[1],xy[1]))

        def updatecourse(self, xy0, course):
            course_rad = np.deg2rad(90-course)
            xy1 = (xy0[0]+100000*np.cos(course_rad),
                xy0[1]+100000*np.sin(course_rad))
            self.track_int_plt.set_data((xy0[0],xy1[0]), 
                                        (xy0[1],xy1[1]))
            self.agentpos.set_marker((3,0,-course))
            
        def addspeedandcourse(self, xy, speed, course):
            self.txtlbl.set_position(xy)
            self.txtlbl.set_text(f'Speed:{speed:.2f} \nCourse:{course:.0f}')          

class SurveyPlotter(Plotter):
    
    def __init__(self,
                 map_lims,
                 scan_lims,
                 map_img,
                 xy0,
                 time_lim, 
                 leadinleadout,
                 min_scan_l,
                 scan_width, 
                 min_scan_ang,
                 n_angles, 
                 n_looks):
        
        self.ml = map_lims
        self.sl = scan_lims
        self.mi = map_img
        self.xy0 = xy0 
        self.tl = time_lim
        self.ll = leadinleadout
        self.ms = min_scan_l
        self.sw = scan_width
        self.msa = min_scan_ang
        self.na = n_angles
        self.nl = n_looks

        self.setup()
    
    def setup(self):
    
        # setup things
        self.setup_plot(self.ml)
        self.setup_map(self.mi)

        self.agent_plt = self.AgentPlot(ax = self.ax,
                                    xy0 = self.xy0)
        self.setup_contacts()
        self.setup_covmap(self.sl, 
                          self.msa,
                          self.na, 
                          self.nl)
        self.setup_scantemp(self.ll,
                            self.ms,
                            self.sw)
        self.updatetime(self.tl, self.tl)
        self.draw()
    
    def update_plots(self,
                     map_stack,
                     detections,
                     agent_xy,
                     agent_xy_hist, 
                     time_remaining):
        # plotting
        if map_stack:
            self.updatecovmap(map_stack)
        self.updatecontacts(detections)
        self.agent_plt.updateagent(agent_xy)
        if len(agent_xy_hist) > 1:
            self.agent_plt.updatetrackhist(agent_xy_hist)
        self.updatetime(time_remaining,
                        time_remaining)
        # self.plotter.remove_temp()
        self.draw()

    def reset(self):
        self.agent_plt.updateagent(self.xy0)
        self.agent.track_hist_plt.set_data((self.xy0[0],
                                            self.xy0[1]))

        self.updatecontacts([])
        self.cov_plt.set_data(np.zeros((self.sl[3]-self.sl[2],
                                    self.sl[1]-self.sl[0],
                                    3)))
        self.det_cls_plt.set_data([], [])
        self.det_grp_plt.set_data([], [])
        self.targ_plt.set_data([], [])
        self.updatetime(self.tl, self.tl)
        
class SEASPlotter(Plotter):

    def __init__(self, 
                 map_lims,
                 xy_start,
                 xy_start_vessels):
        self.setup_plot(map_lims)
        self.agent = self.AgentPlot(ax = self.ax,
                                    xy0 = xy_start, 
                                    speed = 0, 
                                    course = 0)
        self.vessels = []
        for xy in xy_start_vessels:
            self.vessels.append(self.AgentPlot(self.ax,
                                               xy,
                                               col='green', 
                                                speed = 0, 
                                                course = 0))
        self.updateps(1)
        self.ax.figure.canvas.draw()

class AgentViz:
    """
    """

    def __init__(self,
                 map_dims):
        
        im_init = np.zeros((map_dims[3]-map_dims[2],
                            map_dims[1]-map_dims[0]))

        self.fig, axs = plt.subplots(2,2, layout='constrained')
        # Agent pos
        axs[0][0].set_title('Agent position')
        self.plt_ag = axs[0][0].imshow(im_init,
                                       vmin=0,
                                       vmax=1,
                                       cmap='hot',
                                       origin='lower')
        # 
        axs[0][1].set_title('Occupancy grid')
        self.plt_occ = axs[0][1].imshow(im_init,
                                        vmin=0,
                                        vmax=1,
                                        cmap='hot',
                                        origin='lower')

        # 
        axs[1][0].set_title('Coverage map')
        self.plt_cov = axs[1][0].imshow(im_init,
                                        vmin=0,
                                        vmax=5,
                                        cmap='hot',
                                        origin='lower')

        #
        axs[1][1].set_title('Contacts')
        self.plt_cts = axs[1][1].imshow(im_init,
                                        vmin=0,
                                        vmax=3,
                                        cmap='hot',
                                        origin='lower')
        plt.ion()
        plt.show()

    def update(self, 
               ag_im,
               occ_im,
               cov_im,
               con_im):
        
        self.plt_ag.set_data(ag_im)
        self.plt_occ.set_data(occ_im)
        self.plt_cov.set_data(cov_im)
        self.plt_cts.set_data(con_im)
        self.fig.canvas.draw_idle()
        plt.pause(0.0001)
