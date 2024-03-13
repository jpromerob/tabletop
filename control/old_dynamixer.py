from five_bar import FiveBar
import dynamixel_sdk as dx
import time
import math
import multiprocessing
import sys
from tkinter import *
import argparse
import datetime
import numpy as np
import csv
import socket
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


DEG_TO_RAD = math.pi / 180
RAD_TO_DEG = 180 / math.pi

XM = True

if XM:
    # Control table address
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_MIN_POSITION_LIMIT = 52
    ADDR_MAX_POSITION_LIMIT = 48
    ADDR_OPERATING_MODE = 11
    ADDR_PROFILE_VELOCITY = 112
    ADDR_PRESENT_POSITION = 132
    left_trim = 0
    right_trim = 210
else:
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_MIN_POSITION_LIMIT = 52
    ADDR_MAX_POSITION_LIMIT = 48
    ADDR_OPERATING_MODE = 11
    left_trim = 0
    right_trim = 0

# Protocol version
PROTOCOL_VERSION = 2

r1 = 23
r2 = 23
r3 = 23
r4 = 23
r5 = 10

left_id = 1
right_id = 2


center = 4095 // 2

def degree_to_dx(angle):
    # minimum = -1_048_575
    maximum = 4095
    can_move = 2 * math.pi
    degress_per_step = can_move / maximum
    return int(angle/degress_per_step)

def initialize():
    global port
    global handler
    port = dx.PortHandler("/dev/ttyACM0")

    # Initialize PacketHandler Structs
    handler = dx.PacketHandler(2)

    # Open port
    port.openPort()

    # Set port baudrate
    port.setBaudRate(1000000)

    handler.reboot(port, right_id)
    handler.reboot(port, left_id)
    time.sleep(1)

    handler.write4ByteTxRx(port, right_id, ADDR_OPERATING_MODE, 4)
    handler.write4ByteTxRx(port, left_id, ADDR_OPERATING_MODE, 4)


    handler.write1ByteTxRx(port, left_id, ADDR_TORQUE_ENABLE, 1)
    handler.write1ByteTxRx(port, right_id, ADDR_TORQUE_ENABLE, 1)
    handler.write4ByteTxRx(port, right_id, ADDR_OPERATING_MODE, 4)
    handler.write4ByteTxRx(port, left_id, ADDR_OPERATING_MODE, 4)
    handler.write1ByteTxRx(port, left_id, ADDR_TORQUE_ENABLE, 1)
    handler.write1ByteTxRx(port, right_id, ADDR_TORQUE_ENABLE, 1)


def get_angles_from_xy(x, y):

    
    x += r5 / 2
    linkage = FiveBar(r1, r2, r3, r4, r5)
    linkage.inverse(x, y)
    if math.isnan(linkage.get_a11()) or math.isnan(linkage.get_a11()):
        return False
    left_angle  = -(math.pi - linkage.get_a11() - math.pi / 2)
    right_angle = -(math.pi / 2 - linkage.get_a42())


    return left_angle, right_angle

def move_to_from(new_x, new_y, cur_x, cur_y, velocity_rpm = 3):
    # print("in", x, y)
    new_x = new_x - 0.5 # to correct for weird offset
    global port
    global handler

    new_left_angle, new_right_angle = get_angles_from_xy(new_x, new_y)
    cur_left_angle, cur_right_angle = get_angles_from_xy(cur_x, cur_y)

    left_discrete = center + degree_to_dx(new_left_angle) + left_trim
    right_discrete = center + degree_to_dx(new_right_angle) + right_trim


    velocity_discrete = int(velocity_rpm / 0.299)


    if 0 < left_discrete < 4095 and 0 < right_discrete < 4095:
        handler.write4ByteTxRx(port, right_id, ADDR_PROFILE_VELOCITY, velocity_discrete)
        handler.write4ByteTxRx(port, left_id, ADDR_PROFILE_VELOCITY, velocity_discrete)

        handler.write4ByteTxRx(port, right_id, ADDR_GOAL_POSITION, right_discrete)
        handler.write4ByteTxRx(port, left_id, ADDR_GOAL_POSITION, left_discrete)
        return True
    return False


def read_position():
    right_raw = handler.read4ByteTxRx(port, right_id, ADDR_PRESENT_POSITION)
    left_raw = handler.read4ByteTxRx(port, left_id, ADDR_PRESENT_POSITION)

    right_raw = right_raw[0]
    left_raw = left_raw[0]

    right_angle = (right_raw - right_trim - center) / 4095 * 2 * math.pi
    left_angle = (left_raw - left_trim - center) / 4095 * 2 * math.pi

    right_adjusted = right_angle + math.pi / 2
    left_adjusted = left_angle + math.pi / 2

    # print("out raw angle", left_angle, right_angle)
    # print("out adjusted angle", left_adjusted, right_adjusted)

    linkage = FiveBar(r1, r2, r3, r4, r5)
    linkage.forward(
        left_angle,
        right_angle
    )
    (x, y) = linkage.calculate_position(left_adjusted, right_adjusted)
    x -= r5 / 2 # center
    return (x, y)

def points_on_circle(R, x_0, y_0, ang_increment):
    angle = 0
    while True:
        x = x_0 + R * math.cos(angle)
        y = y_0 + R * math.sin(angle)
        yield x, y
        angle += ang_increment*math.pi/180  # Adjust the angle increment as needed

def points_along_segment(x_a, y_a, x_b, y_b, distance):
    # Calculate the vector components from A to B
    delta_x = x_b - x_a
    delta_y = y_b - y_a
    
    # Calculate the total distance between A and B
    total_distance = math.sqrt(delta_x**2 + delta_y**2)
    
    # Calculate the number of steps needed to cover the segment with the given distance
    num_steps = int(total_distance / distance)
    
    # Calculate the step size for x and y
    step_x = delta_x / num_steps
    step_y = delta_y / num_steps

    direction = 1 # towards B
    
    # Start at point A
    current_x, current_y = x_a, y_a
    
    # Yield points along the segment from A to B
    while(True):
        yield current_x, current_y
        # print(f'direction: {direction}, current_x: {current_x}, current_y: {current_y}')
        if direction == 1 and round(current_x) == x_b and round(current_y) == y_b:
            direction = -1
        if direction == -1 and round(current_x) == x_a and round(current_y) == y_a:
            direction = 1
        current_x += step_x*direction
        current_y += step_y*direction


def from_px_to_cm(x, y):

    new_x = round(((0.5-y/100)*27.6),6)
    new_y = round(((1-x/100)*46),6)+24

    return new_x, new_y

def xy_from_outer_source():

    # IP and port to listen on
    receiver_ip = "172.16.222.30"
    receiver_port = 6161

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.020)

    # Bind the socket to the receiver's IP and port
    sock.bind((receiver_ip, receiver_port))

    new_x = 0
    new_y = 26

    yield new_x, new_y    

    while True:
        
        # Receive data from the sender
        try:
            data, sender_address = sock.recvfrom(2048)
            # Decode the received data and split it into x and y
            x, y = map(float, data.decode().split(","))

            new_x, new_y = from_px_to_cm(x, y)
            
        except socket.timeout:
            pass
        
        time.sleep(0.001)


        yield new_x, new_y


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

        x, y = shared_data['tip_pose']
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
        ax1.text(0.05, 0.9, f'Current X: {x:.2f}', transform=ax1.transAxes, fontsize=10, verticalalignment='top')
        ax1.set_ylim(-12,12)

        # Update the bottom subplot
        ax2.clear()
        ax2.plot(np.array(time_data) - time_data[-1], y_data, color='red')
        ax2.axhline(y=44, color='k', linestyle='--')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Y Position')
        ax2.text(0.05, 0.9, f'Current Y: {y:.2f}', transform=ax2.transAxes, fontsize=10, verticalalignment='top')
        ax2.set_ylim(20,68)

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

        # Initialize variables to store parsed values

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

def move_process(shared_data):

    time.sleep(1)

   

    initialize()
    sleeper = 0


    generator = xy_from_outer_source()
    approved = True


    new_x = 0
    new_y = 26
    old_x = 0
    old_y = 26
    cur_x = 0
    cur_y = 26


    
    if shared_data['saver']:

        filename = f"{shared_data['filename']}.csv"
    else:
        filename = "trash.csv"
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        while(approved):


            (cur_x, cur_y) = read_position()
            old_x = new_x
            old_y = new_y
            
            new_x, new_y = next(generator)
            shared_data['tip_pose'] = (new_x, new_y)

            if shared_data['saver']:
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
                move_to_from(new_x, new_y, cur_x, cur_y, speed)
            time.sleep(sleeper)






def parse_args():

    parser = argparse.ArgumentParser(description='Dynamixel Controller')
    parser.add_argument('-f', '--filename', type=str, help="Filename", default="none")
    parser.add_argument('-a','--action', action='store_true', help='Motion activated?')


    return parser.parse_args()


if __name__ == '__main__':


    args = parse_args()



    shared_data = multiprocessing.Manager().dict()
    shared_data['filename'] = args.filename
    if args.filename  == "none":
        shared_data['saver'] = False
    else:
        shared_data['saver'] = True
    shared_data['action'] = args.action
    shared_data['max_speed'] = 30
    shared_data['min_speed'] = 5
    shared_data['max_delta'] = 5
    shared_data['min_delta'] = 1
    shared_data['max_x'] = 10
    shared_data['max_y'] = 42
    shared_data['min_x'] = -10
    shared_data['min_y'] = 26
    shared_data['tip_pose'] = (0,shared_data['min_y'])



    param_proc = multiprocessing.Process(target=param_process, args=(shared_data,))
    param_proc.start()



    move_proc = multiprocessing.Process(target=move_process, args=(shared_data,))
    move_proc.start()
    

    plot_proc = multiprocessing.Process(target=plot_process, args=(shared_data,))
    plot_proc.start()
    
    param_proc.join()
    move_proc.join()
    plot_proc.join()





