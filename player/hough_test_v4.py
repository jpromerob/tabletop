import numpy as np
import random
import math
import pdb
import matplotlib.pyplot as plt

def estimate_line(pt_1, pt_2, nb_pts, max_x, max_y):
    # Extract coordinates from the given points
    x1, y1 = pt_1
    x2, y2 = pt_2
    
    # Estimate slope (m) and intercept (b)
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    # Generate x values between x1 and x2
    x_values = np.linspace(x1, x2, nb_pts)

    # Create a list of tuples [x, y] ensuring y is positive and less than max_y
    result_list = [[int(x), min(int(m*x + b), max_y)] for x in x_values]

    # Ensure the points of pt_1 and pt_2 are included in the list
    # result_list.append(pt_1)
    # result_list.append(pt_2)

    return result_list

def hough_to_cartesian(r, theta):
    # Convert theta to radians if not already
    theta_rad = theta
    
    # Compute slope m and y-intercept b
    m = -np.cos(theta_rad) / np.sin(theta_rad)
    b = r / np.sin(theta_rad)
    
    return m, b



max_x = 256
max_y = 165

# limit of hough space
lim_hp = math.sqrt(max_x**2 + max_y**2)
theta = np.linspace(0, 2*np.pi, max_x)

theta = theta[1:-1]

cartesian = np.zeros((max_y, max_x))
houghian = np.zeros((max_y, max_x))


my_pts = estimate_line([20,10],[200,80], 10, max_x, max_y)

pdb.set_trace()

# Showing Points in Cartesian Space
for i in range(len(my_pts)):
    cartesian[my_pts[i][1], my_pts[i][0]] = 1
plt.imshow(cartesian, cmap='viridis', interpolation="none")
plt.savefig('HoughTestv4a.png')
plt.clf()


for i in range(len(my_pts)):
   
    
    x = my_pts[i][0]
    y = my_pts[i][1]


    # Calculate corresponding r values
    r_float = (x * np.cos(theta) + y * np.sin(theta))/lim_hp*max_y
    r_int = r_float.astype(int)

    # pdb.set_trace()

    for j in range(len(r_int)):
        cx = int(theta[j]*(max_x-1)/(2*np.pi))
        cy = int((r_int[j]+max_y)/(2))
        
        houghian[cy, cx] += 1
   

    # Plot the point in Hough space
    plt.plot(theta, r_float, color='grey')
    plt.scatter(theta, r_int, color='r', s=4)

        
    
plt.xlabel('Theta (radians)')
plt.ylabel('r')
plt.xlim(0, 2*np.pi)
plt.ylim(-max_y, max_y)
plt.grid(True)
plt.savefig('HoughTestv4b.png')
plt.clf()

plt.imshow(np.flipud(houghian), cmap='viridis', interpolation="none")
plt.savefig('HoughTestv4c.png')
plt.clf()    

houghian_max = np.where(houghian == np.max(houghian))
print("\n\nLines found:")
for r, theta in zip(houghian_max[0], houghian_max[1]):
    print(f"\nradius: {r}, theta: {theta}")
    # pdb.set_trace()
    m, b  = hough_to_cartesian(r/lim_hp*max_y, theta*2*math.pi/max_x)
    
    x_values = np.linspace(0, max_x, 100)
    y_values = m * x_values + b
    plt.plot(x_values, y_values, color='red')  # You can adjust color and other properties as needed
    print(f"m: {m}, b: {b}")


# plt.imshow(houghian, cmap='viridis', interpolation="none")
plt.xlim(0, max_x)
plt.ylim(0, max_y)


x_pts = [pt[0] for pt in my_pts]
y_pts = [pt[1] for pt in my_pts]

# Perform linear regression to estimate slope (m) and intercept (b)
m, b = np.polyfit(x_pts, y_pts, 1)

# Generate points along the line for plotting
x_values = np.linspace(0, max_x, 100)
y_values = m * x_values + b
plt.plot(x_values, y_values, color='blue', linestyle='--')

plt.savefig('HoughTestv4d.png')
plt.clf()    
#     x = my_pts[i][0]
#     y = my_pts[i][1]


#     # Calculate corresponding r values
#     r = (x * np.cos(theta) + y * np.sin(theta))/lim_hp*max_y


#     # Plot the point in Hough space
#     plt.plot(theta, r, color=colors[i])

        
    
# plt.xlabel('Theta (radians)')
# plt.ylabel('r')
# plt.xlim(0, 2*np.pi)
# plt.ylim(-max_y*1.2, max_y*1.2)
# plt.grid(True)
# plt.savefig('HoughTestv3.png')

