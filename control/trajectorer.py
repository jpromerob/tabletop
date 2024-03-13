import pygame
import multiprocessing
import time
import argparse
import math
import sys 
import socket
import numpy as np
sys.path.append('../common')
from tools import Dimensions, get_shapes

dim = Dimensions.load_from_file('../common/homdim.pkl')

# Function to handle the window and cursor position
def cursor_process(scale, shared_data):
    pygame.init()
    original_window_size = (255, 164)
    window_size = (original_window_size[0] * scale, original_window_size[1] * scale)
    window = pygame.display.set_mode(window_size)
    pygame.display.set_caption("AirHockey (Fake)")

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
    receiver_port = 6161
    
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    while True:
        
        mouse_pos = shared_data['mouse_pos']

        if mouse_pos[0] == 0 and mouse_pos[1] == 0:
            x = 95 #%
            y = 50 #%
        else:
            x = (mouse_pos[0]/dim.fl)*100
            y = (mouse_pos[1]/dim.fw)*100
        # Encode coordinates as bytes
        data = f"{x},{y}".encode()

        # print(f"Sending: {round(x,3)},{round(y,3)}")
        
        # Send data to the receiver
        sock.sendto(data, (receiver_ip, receiver_port))
        

        time.sleep(0.020)  # Adjust sleep time as needed


def circle_process(shared_data):
    theta = 0
    radius = 16 # %
    center_x = 80 # %
    center_y = 50 # %

    delta_deg = 5

    while True:
        x = (center_x + radius * math.cos(theta))*dim.fl/100
        y = (center_y + radius * math.sin(theta))*dim.fw/100
        shared_data['mouse_pos'] = (int(x), int(y))
        theta -= delta_deg*math.pi/180
        time.sleep(0.020)


def parse_args():

    parser = argparse.ArgumentParser(description='Trajectory Generator')
    parser.add_argument('-s', '--scale', type=int, help="Scale", default=2)
    parser.add_argument('-t', '--trajectory', type=str, help="Trajectory: circle, cursor", default="cursor")
    return parser.parse_args()


if __name__ == '__main__':


    args = parse_args()

    shared_data = multiprocessing.Manager().dict()

    shared_data['mouse_pos'] = (0,0)


    if args.trajectory == "cursor":
        p_trajectory = multiprocessing.Process(target=cursor_process, args=(args.scale, shared_data))
    elif args.trajectory == "circle":
        p_trajectory = multiprocessing.Process(target=circle_process, args=(shared_data,))
    else:
        print("Trajectory not supported")
        quit()
    p_trajectory.start()

    # Start the print process
    p_print = multiprocessing.Process(target=send_process, args=(shared_data,))
    p_print.start()


    # Wait for all processes to finish
    p_trajectory.join()
    p_print.join()
