import numpy as np
import pdb
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle


'''
    This function estimates how much the motors need to rotate so a desired end-effector position is reached
    Inputs: 
        - desired (x,y) coordinate of end-effector 
        - l_0 distance between motors
        - l_1, l_2, l_3, l_4: length of arms
    Outputs: 
        - necessary angles for motors (left and right) 
        - (x,y) coordinates of free joints that result from such angles
'''
def apply_inverse_kinematics(pt, l_0, l_1, l_2, l_3, l_4):
    
    # pdb.set_trace()
    d_left = math.sqrt((pt[0])**2+(pt[1])**2)
    if pt[0] == 0:
        phi_left = math.pi/2
    elif pt[0] > 0:
        phi_left = math.atan(pt[1]/pt[0])
    else: 
        phi_left = math.pi + math.atan(pt[1]/pt[0])
    alpha_left = math.acos((d_left**2 + l_1**2 - l_2**2)/(2*d_left*l_1))
    theta_left = phi_left + alpha_left
    lfj_pt = (l_1*math.cos(theta_left), l_1*math.sin(theta_left))
    
    d_right = math.sqrt((l_0-pt[0])**2+(pt[1])**2)
    if pt[0] == l_0:
        phi_right = math.pi/2
    elif pt[0] < l_0:
        phi_right = math.atan(pt[1]/(l_0-pt[0]))
    else:
        phi_right = math.pi + math.atan(pt[1]/(l_0-pt[0]))
    # pdb.set_trace()
    alpha_right = math.acos((d_right**2 + l_4**2 - l_3**2)/(2*d_right*l_4))
    theta_right = phi_right + alpha_right        
    rfj_pt = (l_0-l_4*math.cos(theta_right), l_4*math.sin(theta_right))

    return theta_left, theta_right, lfj_pt, rfj_pt


'''
    This function calculates the lenght of the arms that allow for a certain set of points of interest to be reached by the end-effector.
    Inputs: 
        - pt_array: array of coordinates (x,y) of points of interest
        - l_0: distance between motors (left and right)
        - k: ratio l_2/l_1 = l_3/l_4 (l_1 and l_4 are arms connected to motors)
    Outputs:
'''
def select_arm_lengths(pt_array, l_0, k):

    rough_l_1 = []
    for pt in pt_array:
        rough_l_1.append(math.sqrt(((pt[0]-l_0)**2 + (pt[1])**2)/(1+k)**2))
    l_1 = round(max(rough_l_1)/10,0)*10
    # pdb.set_trace()
    l_2 = k*l_1
    l_3 = l_2
    l_4 = l_1

    print("l_1 = l_4 = " + str(l_1) + " | l_2 = l_3 = " + str(l_2))

    return l_1, l_2, l_3, l_4

def plot_configuration(coor_frame, lfj_pt, rfj_pt, o_x, o_y, l_0, l_t, w_t):
    
        # All the inverse kinematics are calculated based on left-motor located at the origin
        origin_name = ['table_center', 'table_left_bottom', 'left_motor', 'motor_middle']
        coor_lm = [(-l_0/2, -(o_y+l_t/2)), (o_x, -o_y), (0, 0), (-l_0/2, 0)]
        cf = origin_name.index(coor_frame)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Create the 'pitch'
        rectangle = Rectangle((-o_x+coor_lm[cf][0], o_y+coor_lm[cf][1]), w_t, l_t, edgecolor='k', facecolor='none')
        ax.add_patch(rectangle)        
        plt.plot([-o_x+coor_lm[cf][0],l_0+o_x+coor_lm[cf][0]], [o_y+l_t/2+coor_lm[cf][1], o_y+l_t/2+coor_lm[cf][1]], color='k', linestyle='--') # middle line

        # pdb.set_trace()

        plt.plot([0+coor_lm[cf][0], lfj_pt[0]+coor_lm[cf][0], pt[0]+coor_lm[cf][0]], [0+coor_lm[cf][1],lfj_pt[1]+coor_lm[cf][1], pt[1]+coor_lm[cf][1]], color='g') # left arms
        plt.plot(0+coor_lm[cf][0], 0+coor_lm[cf][1], marker='o', color='g') # motor left
        plt.plot(lfj_pt[0]+coor_lm[cf][0], lfj_pt[1]+coor_lm[cf][1], marker='o', color='g') # 'left free joint'
        c_l_m = Circle((lfj_pt[0]+coor_lm[cf][0], lfj_pt[1]+coor_lm[cf][1]), l_1, edgecolor='g', facecolor='none', alpha=0.5) # left-motor
        ax.add_patch(c_l_m)

        plt.plot([l_0+coor_lm[cf][0], rfj_pt[0]+coor_lm[cf][0], pt[0]+coor_lm[cf][0]], [0+coor_lm[cf][1],rfj_pt[1]+coor_lm[cf][1], pt[1]+coor_lm[cf][1]], color='b') # right arms
        plt.plot(l_0+coor_lm[cf][0], 0+coor_lm[cf][1], marker='o', color='b') # motor right
        plt.plot(rfj_pt[0]+coor_lm[cf][0], rfj_pt[1]+coor_lm[cf][1], marker='o', color='b') # 'right free joint'
        c_r_m = Circle((rfj_pt[0]+coor_lm[cf][0], rfj_pt[1]+coor_lm[cf][1]), l_4, edgecolor='b', facecolor='none', alpha=0.5) # right-motor
        ax.add_patch(c_r_m)

        plt.plot(pt[0]+coor_lm[cf][0], pt[1]+coor_lm[cf][1], marker='o', color='r') # 'end-effector'
             
        ax.set_xlim(-(2*o_x+l_1)+coor_lm[cf][0], (2*o_x+l_1) + l_0+coor_lm[cf][0])
        ax.set_ylim(-(2*o_y+l_1)+coor_lm[cf][1], (2*o_y+l_1) + l_t+coor_lm[cf][1])
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title(pt_names[idx])
        ax.set_aspect('equal')
        plt.grid(True)

        # Save the plot as a PNG file
        plt.savefig('tabletop_'+ pt_names[idx] +'.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    
    # Units: mm and radians

    # Size of play-area
    l_t = 500
    w_t = 300

    # Distance between motors
    l_0 = 60
    k = 1 # ratio l_2/l_1

    # Offsets between motor and bottom-left corner of play-area
    o_x = (w_t - l_0)/2
    o_y = 120

    plot_results = False

    pt_names = ["left_bottom", "right_bottom", "left_top", "right_top", "center_bottom", "center_top"]
    pt_array = [(-o_x, o_y), (l_0 + o_x, o_y), (-o_x, o_y + l_t/2), (l_0 + o_x, o_y + l_t/2), (l_0/2, o_y), (l_0/2, o_y + l_t/2)]
    l_1, l_2, l_3, l_4 = select_arm_lengths(pt_array, l_0, k)

    for idx, pt in enumerate(pt_array):
        
        print(pt_names[idx] + " (i.e. (" +str(pt[0]) + "," + str(pt[1]) + "))")
        # pdb.set_trace()
        theta_left, theta_right, lfj_pt, rfj_pt = apply_inverse_kinematics(pt, l_0, l_1, l_2, l_3, l_4)
        if plot_results:
            print("   Left: ")
            print("      θ: " + str(theta_left*180/math.pi) + " deg")
            print("   Right: ")
            print("      θ: " + str(theta_right*180/math.pi) + " deg")

        plot_configuration('motor_middle', lfj_pt, rfj_pt, o_x, o_y, l_0, l_t, w_t)
    # pdb.set_trace()