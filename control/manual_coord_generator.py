import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time
import curses
import pdb
import time
import sys
sys.path.append('../common')
from tools import Dimensions, get_shapes

dim = Dimensions.load_from_file('../common/homdim.pkl')

pdb.set_trace()

WIDTH = 240
HEIGHT = 180
RADIUS = 15
OFFSET_X = 0
OFFSET_Y = 2
DELTA_X = 10
DELTA_Y = 10

KEY_REPEAT_DELAY = 0.020  # Delay in seconds between consecutive key presses

# Generator function to generate (x, y) coordinates
def generate_coordinates(stdscr):
    x = WIDTH // 2
    y = HEIGHT // 2
    last_key_press_time = time.time()
    
    while True:
        t = time.time()
        c = stdscr.getch()

        # Check if enough time has elapsed since the last key press
        if t - last_key_press_time >= KEY_REPEAT_DELAY:
            if c == curses.KEY_UP:
                y = min(HEIGHT - RADIUS, y + DELTA_Y)
            elif c == curses.KEY_DOWN:
                y = max(RADIUS, y - DELTA_Y)
            elif c == curses.KEY_RIGHT:
                x = min(WIDTH - RADIUS, x + DELTA_X)
            elif c == curses.KEY_LEFT:
                x = max(RADIUS, x - DELTA_X)
            last_key_press_time = t  # Update the last key press time

        yield x, y

# Function for real-time plotting
def animate(i, coords, line, circle):
    x, y = next(coords)
    line.set_data([x], [y])
    circle.center = (x, y)
    return line, circle


if __name__ == "__main__":
    # Initialize curses
    stdscr = curses.initscr()
    curses.cbreak()
    stdscr.keypad(True)

    # Initialize plot
    fig, ax = plt.subplots()
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    line, = ax.plot([], [], 'ro')
    circle = plt.Circle((0, 0), RADIUS, color='black', fill=False)
    ax.add_artist(circle)

    # Create FuncAnimation object
    coords_generator = generate_coordinates(stdscr)
    ani = FuncAnimation(fig, animate, fargs=(coords_generator, line, circle), interval=50, save_count=60)

    # Show the plot
    plt.show()

    # Clean up curses
    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()
