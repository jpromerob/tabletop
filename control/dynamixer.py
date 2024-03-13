from dynapi import *
import time
import math
import multiprocessing
from tkinter import *
import argparse
import numpy as np
import csv
import socket
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

SLEEPER = 0.1


UDP_IP = '172.16.222.30'  
PORT_UDP_PUCK = 7373  
PORT_UDP_TARGET = 6161


'''
This function converts X-Y from pixel percentage to cm
inputs: x and y must be floats from 0 to 1
'''
def from_px_to_cm(x, y):

    new_x = round(((0.5-y/100)*27.6),6)
    new_y = round(((1-x/100)*46),6)+24

    return new_x, new_y


def algorithm(pose, mode):

    x, y = pose
    if mode == 'block':
        y = 26
    return x, y

'''
This process stores tip poses that are received through UDP
The incoming tuples are floats between 0 and 1 
A conversion into coordinates in [cm] is done using 'from_px_to_cm'
'''
def xy_from_outer_source(shared_data):


    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.020)

    # Bind the socket to the receiver's IP and port
    sock.bind((UDP_IP, PORT_UDP_TARGET))

    new_x = 0
    new_y = 26

    while True:
        
        # Receive data from the sender
        try:
            data, sender_address = sock.recvfrom(2048)
            # Decode the received data and split it into x and y
            x, y = map(float, data.decode().split(","))
            # print(f"Got {x}, {y}")
            new_x, new_y = from_px_to_cm(x, y)
            
        except socket.timeout:
            pass
        
        shared_data['des_puck_pose'] =  algorithm((new_x, new_y), 'block')


'''
This function plots the current tip's pose
'''
def plot_process(shared_data):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.suptitle('Live Puck X-Y position')


    # initialize_dynamixel lists to store data for plotting
    x_des_data = []
    y_des_data = []
    x_cur_data = []
    y_cur_data = []
    e_data = []
    time_data = []

    def update_plot(frame):
        nonlocal x_des_data, y_des_data, x_cur_data, y_cur_data, e_data, time_data

        x_des, y_des = shared_data['des_puck_pose']
        x_cur, y_cur = shared_data['cur_puck_pose']
        e  = math.sqrt((x_des-x_cur)**2+(y_des-y_cur)**2)
        current_time = time.time()

        # Append new data to the lists
        x_des_data.append(x_des)
        y_des_data.append(y_des)
        x_cur_data.append(x_cur)
        y_cur_data.append(y_cur)
        e_data.append(e)
        time_data.append(current_time)

        # Keep only the last 20 data points
        nb_pts = 64
        x_des_data = x_des_data[-nb_pts:]
        y_des_data = y_des_data[-nb_pts:]
        x_cur_data = x_cur_data[-nb_pts:]
        y_cur_data = y_cur_data[-nb_pts:]
        e_data = e_data[-nb_pts:]
        time_data = time_data[-nb_pts:]


        # Update the top subplot
        ax1.clear()
        ax1.plot(np.array(time_data) - time_data[-1], x_des_data, color='green')
        ax1.plot(np.array(time_data) - time_data[-1], x_cur_data, color='green', linestyle='--')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('X Position')
        text_x = f'Desired X: {x_des:.2f} | Current X: {x_cur:.2f}'
        ax1.text(0.05, 0.9, text_x, transform=ax1.transAxes, fontsize=10, verticalalignment='top')
        ax1.set_ylim(-24,24)

        # Update the bottom subplot
        ax2.clear()
        ax2.plot(np.array(time_data) - time_data[-1], y_des_data, color='red')
        ax2.plot(np.array(time_data) - time_data[-1], y_cur_data, color='red', linestyle='--')
        ax2.axhline(y=44, color='k', linestyle='--')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Y Position')
        text_y = f'Desired Y: {y_des:.2f} | Current Y: {y_cur:.2f}'
        ax2.text(0.05, 0.9, text_y, transform=ax2.transAxes, fontsize=10, verticalalignment='top')
        ax2.set_ylim(20,84)


        # Update the bottom subplot
        ax3.clear()
        ax3.plot(np.array(time_data) - time_data[-1], e_data, color='black')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Euclidian Error')
        text_e = f'Error: {e:.2f}'
        ax3.text(0.05, 0.9, text_e, transform=ax3.transAxes, fontsize=10, verticalalignment='top')
        ax3.set_ylim(0,10)

    # Set up the animation
    ani = FuncAnimation(fig, update_plot, interval=100)

    plt.show()


def param_process(shared_data):

    max_speed = None
    min_speed = None
    max_delta = None
    min_delta = None
    while(True):
        # Open the file
        with open('params.cfg', 'r') as f:
            # Read lines from the file
            lines = f.readlines()

        # initialize_dynamixel variables to store parsed values

        # Iterate through the lines
        for line in lines:
            # Split each line by colon (assuming the colon separates key and value)
            parts = line.strip().split(':')
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                # Check if the key matches speed or delta
                if key == 'max_speed':
                    if max_speed != int(value):
                        max_speed = int(value)
                        print(f"max_speed updated to {max_speed}")
                if key == 'min_speed':
                    if min_speed != int(value):
                        min_speed = int(value)
                        print(f"min_speed updated to {min_speed}")
                if key == 'max_delta':
                    if max_delta != int(value):
                        max_delta = int(value)
                        print(f"max_delta updated to {max_delta}")
                if key == 'min_delta':
                    if min_delta != int(value):
                        min_delta = int(value)
                        print(f"min_delta updated to {min_delta}")

        time.sleep(1)

        shared_data['max_speed'] = max_speed
        shared_data['min_speed'] = min_speed
        shared_data['max_delta'] = max_delta
        shared_data['min_delta'] = min_delta

'''
The speed at which the motors move depends on how far the current/target locations are
Huge distance (between current and target): high speed
'''
def find_speed(new_x, new_y, cur_x, cur_y, shared_data):

    max_d = shared_data['max_delta']
    min_d = shared_data['min_delta']

    delta= math.sqrt((new_x-cur_x)**2+(new_y-cur_y)**2)
    if delta > max_d:
        speed = shared_data['max_speed']
    elif delta < min_d:
        speed = shared_data['min_speed']
    else:
        m = (shared_data['max_speed']-shared_data['min_speed'])/(max_d-min_d)
        b = shared_data['min_speed']-m*(min_d)
        speed = delta*m+b
    
    return speed



'''
This process polls the motors for their current position and returns the tip's pose accordingly
'''
def read_process(shared_data):

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    time.sleep(1)
    counter = 0
    while(True):


        try:
            mutex.acquire()
            (cur_x, cur_y) = read_position()
            mutex.release()
            shared_data['cur_puck_pose'] = (cur_x, cur_y)
            message = f"{round(cur_x,3)},{round(cur_y,3)}"
            sock.sendto(message.encode(), (UDP_IP, PORT_UDP_PUCK))

            if counter >= 10:
                # print(f"Sent {round(cur_x,3)},{round(cur_y,3)}")
                counter = 0
            else:
                counter+=1
        except:
            pass

        time.sleep(0.005)





def move_process(shared_data):

    # Wait sufficient time so eveything is properly initialized
    time.sleep(2)

    approved = True

    new_x = 0
    new_y = 26
    old_x = 0
    old_y = 26
    cur_x = 0
    cur_y = 26
    
    if shared_data['filename']=="trash":
        save_data = False
    else:
        save_data = True        
    filename = f"{shared_data['filename']}.csv"

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        while(approved):

            (cur_x, cur_y) = shared_data['cur_puck_pose']

            old_x = new_x
            old_y = new_y
            
            new_x, new_y = shared_data['des_puck_pose']

            if save_data:
                writer.writerow([cur_x, cur_y, old_x, old_y])


            # Imaginary margin
            if new_x < shared_data['min_x'] :
                new_x = shared_data['min_x']
            if new_x > shared_data['max_x']:
                new_x = shared_data['max_x']

            if new_y < shared_data['min_y'] :
                new_y = shared_data['min_y']
            if new_y > shared_data['max_y']:
                new_y = shared_data['max_y']
                


            speed = find_speed(new_x, new_y, cur_x, cur_y, shared_data)
            # print(f"Moving to {new_x},{new_y} @{speed}")
            if shared_data['action']:
                mutex.acquire()
                move_to_from(new_x, new_y, cur_x, cur_y, speed)
                mutex.release()
            time.sleep(0.005)





def parse_args():

    parser = argparse.ArgumentParser(description='Dynamixel Controller')
    parser.add_argument('-f', '--filename', type=str, help="Filename", default="trash")
    parser.add_argument('-a','--action', action='store_true', help='Motion activated?')


    return parser.parse_args()




def initialize_shared_data():

    shared_data = multiprocessing.Manager().dict()
    shared_data['filename'] = args.filename
    shared_data['action'] = args.action
    shared_data['max_speed'] = 30
    shared_data['min_speed'] = 5
    shared_data['max_delta'] = 5
    shared_data['min_delta'] = 1
    shared_data['max_x'] = 10
    shared_data['max_y'] = 42
    shared_data['min_x'] = -10
    shared_data['min_y'] = 26
    shared_data['des_puck_pose'] = (0,shared_data['min_y'])
    shared_data['cur_puck_pose'] = shared_data['des_puck_pose']

    return shared_data



if __name__ == '__main__':


    args = parse_args()

    initialize_dynamixel(args.action)
    shared_data = initialize_shared_data()
    mutex = multiprocessing.Lock()


    param_proc = multiprocessing.Process(target=param_process, args=(shared_data,))
    param_proc.start()


    read_proc = multiprocessing.Process(target=read_process, args=(shared_data,))
    read_proc.start()


    move_proc = multiprocessing.Process(target=move_process, args=(shared_data,))
    move_proc.start()
    
    recv_process = multiprocessing.Process(target=xy_from_outer_source, args=(shared_data,))
    recv_process.start()

    plot_proc = multiprocessing.Process(target=plot_process, args=(shared_data,))
    plot_proc.start()
    


    param_proc.join()
    read_proc.join()    
    move_proc.join()
    recv_process.join()
    plot_proc.join()





