import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time

WIDTH = 240
HEIGHT = 180
RADIUS = 15

# Generator function to generate (x, y) coordinates
def generate_coordinates():
    while True:
        t = time.time()
        x = RADIUS + int((WIDTH/2 - 2*RADIUS)/2 * (np.cos(0.7*t) + 1))
        y = RADIUS + int((HEIGHT - 2*RADIUS)/2 * (np.sin(0.5*t) + 1))
        yield x, y

# Function for real-time plotting
def animate(i, coords, line, circle):
    x, y = next(coords)
    line.set_data([x], [y])
    circle.center = (x, y)
    return line, circle


if __name__ == "__main__":
    # Initialize generator
    coords_generator = generate_coordinates()

    # Initialize plot
    fig, ax = plt.subplots()
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    line, = ax.plot([], [], 'ro')
    circle = plt.Circle((0, 0), RADIUS, color='black', fill=False)
    ax.add_artist(circle)

    # Create FuncAnimation object
    ani = FuncAnimation(fig, animate, fargs=(coords_generator, line, circle), interval=50, save_count=60)

    # Show the plot
    plt.show()
