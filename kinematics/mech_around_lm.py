import numpy as np
import pdb
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle



# Units: mm and radians

l_t = 500
w_t = 300

l_0 = 60
k = 1
# l_1 = 180
# l_2 = 180
# l_3 = l_2
# l_4 = l_1

o_x = (w_t - l_0)/2
print("o_x : " + str(o_x))
o_y = 60



if __name__ == "__main__":
    
    plot_results = False

    pt_names = ["left_bottom", "right_bottom", "left_top", "right_top", "center_bottom", "center_top"]
    pt_array = [(-o_x, o_y), (l_0 + o_x, o_y), (-o_x, o_y + l_t/2), (l_0 + o_x, o_y + l_t/2), (l_0/2, o_y), (l_0/2, o_y + l_t/2)]

    # Let's say L1 = L2
    rough_l_1 = []
    for pt in pt_array:
        rough_l_1.append(math.sqrt(((pt[0]-l_0)**2 + (pt[1])**2)/(1+k)**2))
    l_1 = round(max(rough_l_1)/100,1)*100
    l_2 = k*l_1
    l_3 = l_2
    l_4 = l_1

    print("l_1 = l_4 = " + str(l_1))
    print("l_2 = l_3 = " + str(l_2))
    
    lfj_array = []
    rfj_array = []
    for idx, pt in enumerate(pt_array):
        
        # Create a figure and axis
        fig, ax = plt.subplots()

        # Create a Rectangle patch
        rectangle = Rectangle((-o_x, o_y), w_t, l_t, edgecolor='k', facecolor='none')

        # Add the rectangle to the plot
        ax.add_patch(rectangle)

        
        plt.plot([-o_x,l_0+o_x], [o_y+l_t/2, o_y+l_t/2], color='k', linestyle='--') # left arms

        d_left = math.sqrt((pt[0])**2+(pt[1])**2)
        if pt[0] >= 0:
            phi_left = math.atan(pt[1]/pt[0])
        else: 
            phi_left = math.pi + math.atan(pt[1]/pt[0])
        alpha_left = math.acos((d_left**2 + l_1**2 - l_2**2)/(2*d_left*l_1))
        theta_left = phi_left + alpha_left

        lfj_pt = (l_1*math.cos(theta_left), l_1*math.sin(theta_left))
        # if theta_left <= math.pi/2:
        #     lfj_pt = (l_1*math.cos(theta_left), l_1*math.sin(theta_left))
        # elif theta_left <= math.pi:
        #     lfj_pt = (-l_1*math.cos(math.pi-theta_left), l_1*math.sin(math.pi-theta_left))
        # else:
        #     lfj_pt = (-l_1*math.cos(theta_left-math.pi), -l_1*math.sin(theta_left-math.pi))
        lfj_array.append(lfj_pt)
        
        d_right = math.sqrt((l_0-pt[0])**2+(pt[1])**2)
        if pt[0] <= l_0:
            phi_right = math.atan(pt[1]/(l_0-pt[0]))
        else:
            phi_right = math.pi + math.atan(pt[1]/(l_0-pt[0]))
        alpha_right = math.acos((d_right**2 + l_4**2 - l_3**2)/(2*d_right*l_4))
        theta_right = phi_right + alpha_right

        
        rfj_pt = (l_0-l_4*math.cos(theta_right), l_4*math.sin(theta_right))
        # if theta_right <= math.pi/2:
        #     rfj_pt = (l_0-l_4*math.cos(theta_right), l_4*math.sin(theta_right))
        # elif theta_right <= math.pi:
        #     rfj_pt = (l_0+l_4*math.cos(math.pi-theta_right), l_4*math.sin(math.pi-theta_right))
        # else:
        #     rfj_pt = (l_0+l_4*math.cos(theta_right-math.pi), -l_4*math.sin(theta_right-math.pi))
        rfj_array.append(rfj_pt)

        if plot_results:
            print("For " + pt_names[idx] + " (i.e. (" +str(pt[0]) + "," + str(pt[1]) + ")):")
            print("   Left: ")
            print("      d: " + str(d_left))
            print("      φ: " + str(phi_left*180/math.pi) + " deg")
            print("      α: " + str(alpha_left*180/math.pi) + " deg")
            print("      θ: " + str(theta_left*180/math.pi) + " deg")
            print("   Right: ")
            print("      d: " + str(d_right))
            print("      φ: " + str(phi_right*180/math.pi) + " deg")
            print("      α: " + str(alpha_right*180/math.pi) + " deg")
            print("      θ: " + str(theta_right*180/math.pi) + " deg")

        # Plot the line connecting A and B
        
        offset = -l_0/2

        plt.plot([0, lfj_pt[0], pt[0]], [0,lfj_pt[1], pt[1]], color='g') # left arms
        plt.plot(0, 0, marker='o', color='g') # motor left
        plt.plot(lfj_pt[0], lfj_pt[1], marker='o', color='g') # 'left free joint'
        c_l_m = Circle((lfj_pt[0], lfj_pt[1]), l_1, edgecolor='g', facecolor='none', alpha=0.5) # left-motor
        ax.add_patch(c_l_m)

        plt.plot([l_0, rfj_pt[0], pt[0]], [0,rfj_pt[1], pt[1]], color='b') # right arms
        plt.plot(l_0, 0, marker='o', color='b') # motor right
        plt.plot(rfj_pt[0], rfj_pt[1], marker='o', color='b') # 'right free joint'
        c_r_m = Circle((rfj_pt[0], rfj_pt[1]), l_4, edgecolor='b', facecolor='none', alpha=0.5) # right-motor
        ax.add_patch(c_r_m)

        plt.plot(pt[0], pt[1], marker='o', color='r') # 'end-effector'
        # c_e_e = Circle((pt[0], pt[1]), l_2, edgecolor='r', facecolor='none', alpha=0.5) # end-effector
        # ax.add_patch(c_e_e)



        
    
        ax.set_xlim(-(2*o_x+l_1), (2*o_x+l_1) + l_0)
        ax.set_ylim(-(2*o_y+l_1), (2*o_y+l_1) + l_t)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title(pt_names[idx])
        ax.set_aspect('equal')
        plt.grid(True)

        # Save the plot as a PNG file
        plt.savefig('tabletop_'+ pt_names[idx] +'.png', dpi=300, bbox_inches='tight')

    pdb.set_trace()