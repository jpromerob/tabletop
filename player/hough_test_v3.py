import numpy as np
import random
import math
import matplotlib.pyplot as plt

max_x = 256
max_y = 165

# limit of hough space
lim_hp = math.sqrt(max_x**2 + max_y**2)

nb_pts = 10

theta = np.linspace(0, 2*np.pi, 64)
theta = theta[1:-1]

my_pts = [[128,10],[128,155], [128,40], [56,40], [56,120], [56,80], [255,100]]

colors = ['r', 'r', 'r', 'b', 'b', 'b', 'k']



for i in range(len(my_pts)):
   
    
    x = my_pts[i][0]
    y = my_pts[i][1]


    # Calculate corresponding r values
    r_float = (x * np.cos(theta) + y * np.sin(theta))/lim_hp*max_y
    r_int = r_float.astype(int)


    # Plot the point in Hough space
    plt.plot(theta, r_float, color='grey')
    plt.scatter(theta, r_int, color=colors[i], s=4)

        
    
plt.xlabel('Theta (radians)')
plt.ylabel('r')
plt.xlim(0, 2*np.pi)
plt.ylim(-max_y*1.2, max_y*1.2)
plt.grid(True)
plt.savefig('HoughTestv3.png')

