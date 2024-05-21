import numpy as np
import random
import math
import matplotlib.pyplot as plt

max_x = 256
max_y = 165

# limit of hough space
lim_hp = math.sqrt(max_x**2 + max_y**2)

nb_pts = 10

theta = np.linspace(0, 2*np.pi, 256)

my_pts = [[128,0],[128,165], [0,0], [256,165], [0,84], [256,84]]

colors = ['k', 'k', 'r', 'r', 'b', 'b']



for i in range(len(my_pts)):
   
    
    x = my_pts[i][0]
    y = my_pts[i][1]


    # Calculate corresponding r values
    r = (x * np.cos(theta) + y * np.sin(theta))/lim_hp*max_y


    # Plot the point in Hough space
    plt.plot(theta, r, color=colors[i])

        
    
plt.xlabel('Theta (radians)')
plt.ylabel('r')
plt.xlim(0, 2*np.pi)
plt.ylim(-max_y*1.2, max_y*1.2)
plt.grid(True)
plt.savefig('HoughTestv2.png')

