import pygame
import multiprocessing
import time
import argparse
import sys 
import socket
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
sys.path.append('../common')
from tools import Dimensions, get_shapes

dim = Dimensions.load_from_file('../common/homdim.pkl')

# Function to handle the window and cursor position
def window_process(scale, shared_data):
    pygame.init()
    original_window_size = (255, 164)
    window_size = (original_window_size[0] * scale, original_window_size[1] * scale)
    window = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Window")

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get the unscaled mouse position
        mouse_pos_unscaled = pygame.mouse.get_pos()

        # Scale the mouse position to match the new window size
        mouse_pos_scaled = (int(mouse_pos_unscaled[0]/scale), int(mouse_pos_unscaled[1]/scale))
        shared_data['mouse_pos'] = mouse_pos_scaled

        # Clear the window
        window.fill((255, 255, 255))

        # Draw a red circle outline at the center of the window
        circle_center = (window_size[0] // 2, window_size[1] // 2)
        pygame.draw.circle(window, (255, 0, 0), circle_center, 21 * scale, width=1)

        # Draw a circle at the mouse position
        pygame.draw.circle(window, (0, 0, 255), mouse_pos_unscaled, 5)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    pygame.quit()

# Function to handle printing messages
def send_process(shared_data):

    # IP and port of the receiver
    receiver_ip = "172.16.222.30"
    receiver_port = 5151
    
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    while True:
        
        if 'mouse_pos' in shared_data:
            mouse_pos = shared_data['mouse_pos']

            x = (mouse_pos[0]/dim.fl)*100
            y = (mouse_pos[1]/dim.fw)*100
            # Encode coordinates as bytes
            data = f"{x},{y}".encode()

            # print(f"Sending: {round(x,3)},{round(y,3)}")
            
            # Send data to the receiver
            sock.sendto(data, (receiver_ip, receiver_port))
            

            time.sleep(0.020)  # Adjust sleep time as needed

# Function to plot the cursor positions in real-time
def plot_process(shared_data):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Live Puck X-Y position')


    # Initialize lists to store data for plotting
    x_data = []
    y_data = []
    time_data = []

    def update_plot(frame):
        nonlocal x_data, y_data, time_data

        if 'mouse_pos' in shared_data:
            mouse_pos = shared_data['mouse_pos']
            x = mouse_pos[0] / dim.fl
            y = mouse_pos[1] / dim.fw
            current_time = time.time()

            # Append new data to the lists
            x_data.append(x)
            y_data.append(y)
            time_data.append(current_time)

            # Keep only the last 20 data points
            nb_pts = 100
            x_data = x_data[-nb_pts:]
            y_data = y_data[-nb_pts:]
            time_data = time_data[-nb_pts:]

            # Update the top subplot
            ax1.clear()
            ax1.plot(np.array(time_data) - time_data[-1], x_data, color='green')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('X Position')
            ax1.set_ylim(0, 1)

            # Update the bottom subplot
            ax2.clear()
            ax2.plot(np.array(time_data) - time_data[-1], y_data, color='red')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Y Position')
            ax2.set_ylim(0, 1)

    # Set up the animation
    ani = FuncAnimation(fig, update_plot, interval=100)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--scale', type=int, default=2,
                        help='Scale factor for the window')

    args = parser.parse_args()

    shared_data = multiprocessing.Manager().dict()

    # Start the window process
    window_proc = multiprocessing.Process(target=window_process, args=(args.scale, shared_data))
    window_proc.start()

    # Start the print process
    print_proc = multiprocessing.Process(target=send_process, args=(shared_data,))
    print_proc.start()

    # Start the plot process
    plot_proc = multiprocessing.Process(target=plot_process, args=(shared_data,))
    # plot_proc.start()

    # Wait for all processes to finish
    window_proc.join()
    print_proc.join()
    # plot_proc.join()
