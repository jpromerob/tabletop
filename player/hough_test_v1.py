import numpy as np
import random
import math
import matplotlib.pyplot as plt

max_x = 256
max_y = 165

# limit of hough space
lim_hp = math.sqrt(max_x**2 + max_y**2)

nb_pts = 10

theta = np.linspace(0, 2*np.pi, 1000)

for x in range(max_x):
    for y in range(max_y):

   

        # Calculate corresponding r values
        r = x * np.cos(theta) + y * np.sin(theta)
        

        # Plot the point in Hough space
        plt.plot(theta, r, color='k')
        
    
plt.xlabel('Theta (radians)')
plt.ylabel('r')
plt.xlim(0,2*np.pi)
plt.ylim(-lim_hp*1.2, lim_hp*1.2)
plt.grid(True)
plt.savefig('HoughTestv1.png')

