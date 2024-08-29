import numpy as np
import math
from dataclasses import dataclass

def is_point_in_polygon(point, polygon_corners):
    """ 
    Determines whether a point is in a polygon
    
    Args:
        point: A tuple (x, y) 
        polygon_corners: A list of tuples representing the polygon vertices

    Returns: 
        True if point is inside the polygon, False otherwise.
    """
    x, y = point
    n = len(polygon_corners)
    inside = False
    for i in range(n):
        x1, y1 = polygon_corners[i]
        x2, y2 = polygon_corners[(i + 1) % n]
        
        if y > min(y1, y2) \
           and y <= max(y1, y2) \
           and x <= max(x1, x2) \
           and y1 != y2:
            x_inters = (y - y1)*(x2 - x1)/(y2 - y1) + x1
            if x1 == x2 or x < x_inters:
                inside = not inside
    return inside 


class CoverageMap:
    """_summary_
    """

    def __init__(self, 
                 scan_lims, 
                 leadinleadout,
                 min_scan_l,
                 scan_width,
                 nadir_width):
        # reset coverage map variable
        self.map_stack = []
        # unpack relevant parameters
        self.sa = scan_lims
        self.scan_area = [int(x) for x in self.sa]
        self.leadinleadout = leadinleadout
        self.min_scan_l = min_scan_l
        self.scan_width = scan_width
        self.nadir_width = nadir_width

    def add_scan(self, xy1, xy2):
        # cov_temp: (cov_l,
        #            cov_r,
        #            cov_corners_l,
        #            cov_corners_r,
        #            orientation)
        x1 = xy1[0]
        y1 = xy1[1]
        x2 = xy2[0]
        y2 = xy2[1]

        cov_temp = self.pixelrect(x1 - self.scan_area[0],
                                    x2 - self.scan_area[0],
                                    y1 - self.scan_area[2],
                                    y2 - self.scan_area[2])

        # Compute scan angle
        ang_L_deg = np.rad2deg(cov_temp[4]-np.pi/2+np.pi/2)
        ang_R_deg = np.rad2deg(cov_temp[4]-np.pi/2-np.pi/2)
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
        if len(np.unique(cm))>1:
            self.map_stack.append(cm)
        
        # output the corners and the scan angles
        rect_corners = [cov_temp[2], cov_temp[3]]

        return rect_corners, angs

    def pixelrect(self, x1, x2, y1, y2):
        # compute length and rotation of rectangle
        rec_l = math.dist((x2, y2),
                            (x1, y1)) - 2*self.leadinleadout
        rec_rot = np.arctan2(y2 - y1,
                                x2 - x1)

        dy_sa = self.scan_area[3]-self.scan_area[2]
        dx_sa = self.scan_area[1]-self.scan_area[0]

        # Check that the length of scan is above the minimum scan length
        if rec_l > self.min_scan_l and self.scan_width >0:
            # account for the leadinleadout length
            x1 += np.cos(rec_rot)*self.leadinleadout
            y1 += np.sin(rec_rot)*self.leadinleadout
            x2 -= np.cos(rec_rot)*self.leadinleadout
            y2 -= np.sin(rec_rot)*self.leadinleadout
            # compute scan
            dy = np.sin((np.pi/2) + rec_rot)*self.scan_width/2
            dx = np.cos((np.pi/2) + rec_rot)*self.scan_width/2
            # compute nadir
            dy_nad = np.sin((np.pi/2) + rec_rot)*self.nadir_width/2
            dx_nad = np.cos((np.pi/2) + rec_rot)*self.nadir_width/2
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
            rect_temp = np.flip(self.rect_to_mask(rect_corners_1,
                                          [dx_sa, dy_sa]), 0)
            
            rect_temp2 = np.flip(self.rect_to_mask(rect_corners_2,
                                          [dx_sa, dy_sa]), 0)

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

    def rect_to_mask(self, 
                     rectangle_corners,
                     map_xy) -> np.array:
        width, height = map_xy
        mask = np.zeros((height, width))

        min_x = min(rectangle_corners[0][0], 
                    rectangle_corners[1][0],
                    rectangle_corners[2][0],
                    rectangle_corners[3][0])
        max_x = max(rectangle_corners[0][0], 
                    rectangle_corners[1][0],
                    rectangle_corners[2][0],
                    rectangle_corners[3][0])
        min_y = min(rectangle_corners[0][1], 
                    rectangle_corners[1][1],
                    rectangle_corners[2][1],
                    rectangle_corners[3][1])
        max_y = max(rectangle_corners[0][1], 
                    rectangle_corners[1][1],
                    rectangle_corners[2][1],
                    rectangle_corners[3][1])
        
        for y in range(max(min_y,0), min(max_y+1, height)):
            for x in range(max(min_x,0), min(max_x+1, width)):
                if is_point_in_polygon((x, y), rectangle_corners):
                    mask[y, x] = 1

        return mask

    def reset(self):
        self.map_stack = []

class ContactDetections:
    """_summary_
    """

    def __init__(self, 
                 loc_uncertainty, 
                 scan_lims,
                 n_targets, 
                 det_probs, 
                 clutter_density,
                 det_probs_clutter,
                 clutter_ori_mean, 
                 clutter_ori_std):
        
        # unpack relevant params
        self.loc_uncertainty = loc_uncertainty
        self.sa = scan_lims
        self.n_targets = n_targets
        self.det_probs = det_probs
        self.clutter_density = clutter_density
        self.det_probs_clutter = det_probs_clutter
        self.clutter_ori_mean = clutter_ori_mean
        self.clutter_ori_std = clutter_ori_std

        # initialise vars
        self.detections = []
        self.det_grp = []
        self.det_n = 0
        self.scan_n = 0
        self.group_n = 0

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

    def add_dets(self, rect_corners, scan_angles):
        # output to recorder
        obs_str = []
        if np.array(rect_corners).any():
            # for each side of new scan
            for n in [0, 1]:
                # check if inside scan area and detected
                rect_c_tmp = rect_corners[n] + [self.sa[0],
                                                self.sa[2]]
                x_out, y_out = self.inside_rect(self.truth,
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

        for xy in list(contacts):
            # add some uncertainty to the location of the contact
            xy_temp = xy.location + np.random.normal(0,
                                                    self.loc_uncertainty,
                                                    size=(1, 2))[0]

            # if all signs the same, point is in rectangle
            if is_point_in_polygon(xy_temp, rect_corners) and rect_corners.any():
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
    
    def generate_targets(self, map_occ):
        # Get coordinates of unoccupied spaces on the map
        nz_i = np.where(map_occ == 0)
        
        self.truth = []
        # generate target contacts
        for n in np.arange(self.n_targets):
            n_rnd = np.random.randint(0,len(nz_i[0]))
            self.truth.append(TargetObject(n,
                                            'Target',
                                            (nz_i[1][n_rnd],
                                            nz_i[0][n_rnd]),
                                            round(np.random.uniform(0, 360)),
                                            self.det_probs))
        # generate non-target contacts
        # compute how many targets need to be spawned based on a given clutter density 
        nc = len(nz_i[0])*self.clutter_density
        for n in np.arange(nc):
            n_rnd = np.random.randint(0,len(nz_i[0]))
            self.truth.append(TargetObject(n+self.n_targets,
                                                'Clutter',
                                                (nz_i[1][n_rnd],
                                                 nz_i[0][n_rnd]),
                                                np.random.normal(self.clutter_ori_mean, 
                                                                 self.clutter_ori_std), 
                                                self.det_probs_clutter))

    def reset(self):
        # initialise vars
        self.detections = []
        self.det_grp = []
        self.det_n = 0
        self.scan_n = 0
        self.group_n = 0

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

@dataclass
class TargetObject:
    id: float
    obj_class: str
    location: float
    orientation: float
    det_probs: list
