
import numpy as np

def add_markers(im_acc, radius, points, color):

    # pdb.set_trace()

    for idx in range(len(points)):
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                im_acc[points[idx][1]+i, points[idx][0]+j] = color
    
    return im_acc 


def get_dimensions():

    l = 288
    w = 174
    dlx = 76
    dly = 56

    # Homography centered within 640x480
    ml = int((640-l)/2)
    mw = int((480-w)/2)

    
    # Homography shows Play Area ONLY
    # ml = 120
    # mw = 80
    return l, w, ml, mw, dlx, dly

def get_shapes(l, w, ml, mw, dlx, dly, scale):

    tl = [(ml-dlx)*scale, (mw-dly)*scale]
    tr = [(ml+l+dlx)*scale, (mw-dly)*scale]
    bl = [(ml-dlx)*scale, (mw+w+dly)*scale]
    br = [(ml+l+dlx)*scale, (mw+w+dly)*scale]

    field = [bl, tl, tr, br]

    line = [((ml+int(l/2))*scale, (mw-dly)*scale),((ml+int(l/2))*scale, (mw+w+dly)*scale)]

    c1 = [(ml)*scale, (mw+w)*scale]
    c2 = [(ml)*scale, (mw)*scale]
    c3 = [(ml+l)*scale, (mw)*scale]
    c4 = [(ml+l)*scale, (mw+w)*scale]
    c5 = [(ml+int(l/2))*scale, (mw+int(w/2))*scale]

    circles = [c1, c2, c3, c4, c5]
    radius = 30*scale

    # Coordinates of rectangles that define the goals
    goal_l = 90
    goal_depth = 20
    gu = int((480-goal_l)/2)*scale
    gb = gu + goal_l*scale
    gl_l = (ml-dlx-goal_depth)*scale
    gl_r = (ml-dlx)*scale
    gr_l = (ml+l+dlx)*scale
    gr_r = (ml+l+dlx+goal_depth)*scale
    goal_1 = [[gl_l,gu], [gl_r, gu], [gl_r, gb], [gl_l, gb]]
    goal_2 = [[gr_l,gu], [gr_r, gu], [gr_r, gb], [gr_l, gb]]
    goals = [goal_1, goal_2]

    return field, line, goals, circles, radius