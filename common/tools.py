
import numpy as np
import pdb
import pickle


class Dimensions:
    def __init__(self, l, w, il, iw, d2ex, d2ey, gs, gd, cmr, hs):
        self.l = int(l)
        self.w = int(w)
        self.il = int(il*hs) # inner length (between LEDs)
        self.iw = int(iw*hs) # inner width (between LEDs)
        self.d2ex = int(d2ex*hs) # distance from LED to edge (x axis)
        self.d2ey = int(d2ey*hs) # distance from LED to edge (y axis)
        # self.ml = int((self.l-self.il)/2)
        # self.mw = int((self.w-self.iw)/2)
        self.fl = 2*self.d2ex+self.il
        self.fw = 2*self.d2ey+self.iw
        self.gs = int(gs*hs) # goal size
        self.gd = int(gd*hs) # goal depth
        self.pr = int(cmr*hs) # puck radius
        self.hs = hs

    def save_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_from_file(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)


def get_dimensions(nat_res_x, nat_res_y, hom_scale):

    dim = Dimensions(nat_res_x, nat_res_y, 288, 174, 76, 56, 90, 10, 20, hom_scale)
    return dim


def get_shapes(dim, vis_scale):

    # pdb.set_trace()

    tl = [(0)*vis_scale, (0)*vis_scale]
    tr = [(dim.fl-1)*vis_scale, (0)*vis_scale]
    bl = [(0)*vis_scale, (dim.fw-1)*vis_scale]
    br = [(dim.fl-1)*vis_scale, (dim.fw-1)*vis_scale]

    field = [bl, tl, tr, br]

    line = [(int(dim.fl/2)*vis_scale, (0)*vis_scale), (int(dim.fl/2)*vis_scale, (dim.fw-1)*vis_scale)]

    c1 = [(dim.d2ex)*vis_scale, (dim.d2ey+dim.iw)*vis_scale]
    c2 = [(dim.d2ex)*vis_scale, (dim.d2ey)*vis_scale]
    c3 = [(dim.d2ex+dim.il)*vis_scale, (dim.d2ey)*vis_scale]
    c4 = [(dim.d2ex+dim.il)*vis_scale, (dim.d2ey+dim.iw)*vis_scale]
    c5 = [(dim.d2ex+int(dim.il/2))*vis_scale, (dim.d2ey+int(dim.iw/2))*vis_scale]

    circles = [c1, c2, c3, c4, c5]
    radius = dim.cmr*vis_scale

    # Coordinates of rectangles that define the goals
    gu = int((dim.fw-dim.gs)/2)*vis_scale
    gb = gu + dim.gs*vis_scale
    gl_l = (0)*vis_scale
    gl_r = (dim.gd)*vis_scale
    gr_l = (dim.fl-dim.gd)*vis_scale
    gr_r = (dim.fl-1)*vis_scale
    goal_1 = [[gl_l,gu], [gl_r, gu], [gl_r, gb], [gl_l, gb]]
    goal_2 = [[gr_l,gu], [gr_r, gu], [gr_r, gb], [gr_l, gb]]
    goals = [goal_1, goal_2]

    # pdb.set_trace()
    return field, line, goals, circles, radius
