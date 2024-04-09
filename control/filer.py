from dynapi import *
import time
import math
import multiprocessing
from tkinter import *
import argparse
import numpy as np
import csv
import os
from datetime import datetime
import socket
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


SLEEPER = 0.1


UDP_IP = '172.16.222.30'  
PORT_UDP_PADDLE_CONSOLIDATED = 6464
PORT_UDP_PADDLE_CURRENT = 6363  
PORT_UDP_PADDLE_DESIRED = 6262 
PORT_UDP_PUCK_CURRENT = 6161 


TABLE_LENGHT_X = 27.6
TABLE_LENGHT_Y = 46
MOTOR_OFFSET_Y = 22

LIM_LOW = 25
LIM_HIGH = 36*2-LIM_LOW

'''
This function converts X-Y from pixel percentage to cm
inputs: x and y must be floats from 0 to 1
'''
def from_norm_to_cm(x, y):

    new_x = round(((0.5-y/100)*TABLE_LENGHT_X),6)
    new_y = round(((1-x/100)*TABLE_LENGHT_Y),6)+MOTOR_OFFSET_Y

    return new_x, new_y

def from_cm_to_norm(x, y):

    new_x = -100*((y-MOTOR_OFFSET_Y)/(TABLE_LENGHT_Y)-1)
    new_y = -100*((x)/(TABLE_LENGHT_X)-0.5)

    return new_x, new_y


'''
This process stores desired paddle poses that are received through UDP
The incoming tuples are floats between 0 and 1 
A conversion into coordinates in [cm] is done using 'from_norm_to_cm'
'''
def receive_paddle_xy(shared_data):


    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.020)

    # Bind the socket to the receiver's IP and port
    sock.bind((UDP_IP, PORT_UDP_PADDLE_DESIRED))


    new_x = 0
    new_y = LIM_LOW

    while True:
        
        # Receive data from the sender
        try:
            data, sender_address = sock.recvfrom(2048)
            # Decode the received data and split it into x and y
            x, y = map(float, data.decode().split(","))
            # print(f"Got {x}, {y}")
            new_x, new_y = from_norm_to_cm(x, y)
            
        except socket.timeout:
            pass
        
        shared_data['des_paddle_pose'] =  (round(new_x,3), round(new_y,3))

'''
This process stores puck poses that are received through UDP
The incoming tuples are floats between 0 and 1 
A conversion into coordinates in [cm] is done using 'from_norm_to_cm'
'''
def receive_puck_xy(shared_data):


    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.020)

    # Bind the socket to the receiver's IP and port
    sock.bind((UDP_IP, PORT_UDP_PUCK_CURRENT))


    new_x = 0
    new_y = LIM_LOW

    while True:
        
        # Receive data from the sender
        try:
            data, sender_address = sock.recvfrom(2048)
            # Decode the received data and split it into x and y
            x, y = map(float, data.decode().split(","))
            # print(f"Got {x}, {y}")
            new_x, new_y = from_norm_to_cm(x, y)
            
        except socket.timeout:
            pass
        
        shared_data['cur_puck_pose'] = (round(new_x,3), round(new_y,3))






def file_process(shared_data):
    

    # Get current date and time
    current_time = datetime.now()

    # Format the current time as YYMMDD_HHMM
    formatted_time = current_time.strftime('%y%m%d_%H%M')

    # Print the formatted time
    print(formatted_time)


    filename = f"{shared_data['filename']}_{formatted_time}.csv"

    timestamp = 0
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        while(shared_data["Continue"]):
            puck_x, puck_y = shared_data['cur_puck_pose']            
            paddle_x, paddle_y = shared_data['des_paddle_pose']
            writer.writerow([timestamp, puck_x, puck_y, paddle_x, paddle_y])
            timestamp += 1
            time.sleep(0.001)



def parse_args():

    parser = argparse.ArgumentParser(description='Dynamixel Controller')
    parser.add_argument('-f', '--filename', type=str, help="Filename", default="trash")


    return parser.parse_args()




def initialize_shared_data():

    shared_data = multiprocessing.Manager().dict()
    shared_data["Continue"] = True
    shared_data['filename'] = args.filename
    shared_data['max_speed'] = 30
    shared_data['min_speed'] = 5
    shared_data['max_delta'] = 5
    shared_data['min_delta'] = 1

    shared_data['max_x'] = 10
    shared_data['max_y'] = 42
    shared_data['min_x'] = -10
    shared_data['min_y'] = LIM_LOW

    shared_data['des_paddle_pose'] = (0,shared_data['min_y']) # desired paddle position (from outer source)
    shared_data['cur_puck_pose'] = (0,shared_data['min_y']) # current puck position (from SpiNNaker's SCNN)
    shared_data['cur_paddle_pose'] = (0,shared_data['min_y']) # current paddle position (from Dynamixle readings)

    shared_data['alg_paddle_pose'] = (0,shared_data['min_y']) # applied paddle position (algorithm[des_paddle_pose])
    

    return shared_data


def trigger_process(shared_data):
    cmd = "/opt/aestream/build/src/aestream "
    cmd += "resolution 1280 720 undistortion ~/tabletop/calibration/luts/cam_lut_homography_prophesee.csv "
    cmd += "output udp 172.16.222.30 3330 172.16.223.122 3333 "
    cmd += "input file ~/tabletop/recordings/short_xy.aedat4"
    print(cmd)
    os.system(cmd)
    shared_data["Continue"] = False


if __name__ == '__main__':


    args = parse_args()

    shared_data = initialize_shared_data()
    mutex = multiprocessing.Lock()


    trig_proc = multiprocessing.Process(target=trigger_process, args=(shared_data,))
    recv_paddle_proc = multiprocessing.Process(target=receive_paddle_xy, args=(shared_data,))    
    recv_puck_proc = multiprocessing.Process(target=receive_puck_xy, args=(shared_data,))
    file_proc = multiprocessing.Process(target=file_process, args=(shared_data,))


    trig_proc.start()
    time.sleep(2)
    recv_paddle_proc.start()
    recv_puck_proc.start()
    file_proc.start()

    


    recv_paddle_proc.join()
    recv_puck_proc.join()
    file_proc.join()
    trig_proc.join()





