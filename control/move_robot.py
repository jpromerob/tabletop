from five_bar import FiveBar
import dynamixel_sdk as dx
import time
import math
import sys
from tkinter import *
import argparse
import socket


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

def move_to(x, y):
    global port
    global handler
    x += 4.4
    linkage = FiveBar(r1, r2, r3, r4, r5)
    linkage.inverse(x, y)
    if math.isnan(linkage.get_a11()) or math.isnan(linkage.get_a11()):
        return False
    left_angle  = -(math.pi - linkage.get_a11() - math.pi / 2)
    right_angle = -(math.pi / 2 - linkage.get_a42())

    left_discrete = center + degree_to_dx(left_angle) + left_trim
    right_discrete = center + degree_to_dx(right_angle) + right_trim

    if 0 < left_discrete < 4095 and 0 < right_discrete < 4095:
        handler.write4ByteTxRx(port, right_id, ADDR_GOAL_POSITION, right_discrete)
        handler.write4ByteTxRx(port, left_id, ADDR_GOAL_POSITION, left_discrete)
        return True
    return False


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


def points_from_spinnaker():

    # IP and port to listen on
    receiver_ip = "172.16.222.30"
    receiver_port = 5151

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to the receiver's IP and port
    sock.bind((receiver_ip, receiver_port))

    while True:
        # Receive data from the sender
        data, sender_address = sock.recvfrom(1024)
        
        # Decode the received data and split it into x and y
        x, y = map(float, data.decode().split(","))
        
        y = 28
        print(f"({x},{y})")
        yield x, y

    

def parse_args():

    parser = argparse.ArgumentParser(description='SpiNNaker-SPIF Simulation with Artificial Data')

    parser.add_argument('-t', '--trajectory', type= str, help="Trajectroy: linear, circular, spinnaker", default="linear")
    parser.add_argument('-s', '--speed', type=int, help="Speed: 1 to 100", default=10)
    parser.add_argument('-l', '--line', type=int, help="Line: 0 is back, 18 is front", default=0)
    return parser.parse_args()

if __name__ == '__main__':


    args = parse_args()
    sleeper = 1/(10*args.speed)

    
    initialize()

    y = 24
    x = 0
    time.sleep(1)

    if args.trajectory == 'circular':
        # Circular Motion
        radius = 9
        center_x = 0
        center_y = 33
        ang_increment = 3    
        generator = points_on_circle(radius, center_x, center_y, ang_increment)
        approved = True

    elif args.trajectory == 'linear':
        # Linear Motion
        x_l = int(args.line)+24
        if x_l >=24 and x_l <= 42:
            x_a, y_a = 10, x_l
            x_b, y_b = -10, x_l
            distance_between_points = 0.25
            generator = points_along_segment(x_a, y_a, x_b, y_b, distance_between_points)
            approved = True
        else:
            print("Line not allowed")
            approved = False

    elif args.trajectory == 'spinnaker':

        generator = points_from_spinnaker()
        approved = True

    else:
        approved = False
        print("Unknown Trajectory")


    while(approved):
        x, y = next(generator)

        # Imaginary margin
        if abs(x) <= 10 and 24<= y and 42>=y:

            # print(f"({int(x*10)/10}, {int(y*10)/10})")
            if args.trajectory != "spinnaker":
                move_to(x,y)
            
            time.sleep(sleeper)
        else:
            # print("Error")
            time.sleep(sleeper)




