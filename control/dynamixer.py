from dynapi import *
import time
import aestream
import math
import multiprocessing
from tkinter import *
import argparse
import numpy as np
import csv
import sys
import socket
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
sys.path.append('../common')
from tools import Dimensions, get_shapes

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


    if not shared_data['gpu']:
        print("Receiving X,Y from SpiNNaker")

        in_port = shared_data['board']*100+87

        dim = Dimensions.load_from_file('../common/homdim.pkl')

        x_coord = int(dim.fl/2)
        y_coord = int(dim.fw)
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        with aestream.UDPInput((dim.fl, dim.fw), device = 'cpu', port=in_port) as stream1:
                    
            while True:

                reading = stream1.read().numpy() 
                idx_x = np.where(reading[:,0]>0.5)
                idx_y = np.where(reading[:,1]>0.5)

                try:
                    if len(idx_x[0])>0 and len(idx_y[0]) > 0:
                        x_coord = int(np.mean(idx_x))
                        y_coord = int(np.mean(idx_y))
                except:
                    pass       
                x_norm = x_coord/dim.fl*100
                y_norm = y_coord/dim.fw*100

                new_x, new_y = from_norm_to_cm(x_norm, y_norm)
                
                shared_data['des_paddle_pose'] =  (new_x, new_y)
    else:
        print("Receiving X,Y from GPU")

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
            
            shared_data['des_paddle_pose'] =  (new_x, new_y)
            

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
        
        shared_data['cur_puck_pose'] =  (new_x, new_y)


'''

'''
def forward_paddle_xy(shared_data):




    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    

    while True:

        x, y = shared_data['des_paddle_pose']

        norm_x, norm_y = from_cm_to_norm(x, y)
        message = f"{round(norm_x,3)},{round(norm_y,3)}"
        sock.sendto(message.encode(), (UDP_IP, PORT_UDP_PADDLE_CONSOLIDATED))


        time.sleep(0.005)

'''
This function plots the current tip's pose
'''
def plot_process(shared_data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6.55))
    fig.suptitle('Live Puck X-Y position')


    # initialize_dynamixel lists to store data for plotting
    x_des_paddle_data = []
    y_des_paddle_data = []
    x_cur_paddle_data = []
    y_cur_paddle_data = []
    x_cur_puck_data = []
    y_cur_puck_data = []
    e_data = []
    time_data = []

    def update_plot(frame):
        nonlocal x_des_paddle_data, y_des_paddle_data
        nonlocal x_cur_paddle_data, y_cur_paddle_data
        nonlocal x_cur_puck_data, y_cur_puck_data
        nonlocal e_data, time_data

        x_des_paddle, y_des_paddle = shared_data['des_paddle_pose']
        x_cur_paddle, y_cur_paddle = shared_data['cur_paddle_pose']
        x_cur_puck, y_cur_puck = shared_data['cur_puck_pose']
        e  = math.sqrt((x_des_paddle-x_cur_paddle)**2+(y_des_paddle-y_cur_paddle)**2)
        current_time = time.time()

        # Append new data to the lists
        x_des_paddle_data.append(x_des_paddle)
        y_des_paddle_data.append(y_des_paddle)
        x_cur_paddle_data.append(x_cur_paddle)
        y_cur_paddle_data.append(y_cur_paddle)
        x_cur_puck_data.append(x_cur_puck)
        y_cur_puck_data.append(y_cur_puck)
        e_data.append(e)
        time_data.append(current_time)

        # Keep only the last 20 data points
        nb_pts = 64
        x_des_paddle_data = x_des_paddle_data[-nb_pts:]
        y_des_paddle_data = y_des_paddle_data[-nb_pts:]
        x_cur_paddle_data = x_cur_paddle_data[-nb_pts:]
        y_cur_paddle_data = y_cur_paddle_data[-nb_pts:]
        x_cur_puck_data = x_cur_puck_data[-nb_pts:]
        y_cur_puck_data = y_cur_puck_data[-nb_pts:]
        e_data = e_data[-nb_pts:]
        time_data = time_data[-nb_pts:]


        # Update the top subplot
        ax1.clear()
        ax1.plot(np.array(time_data) - time_data[-1], x_cur_puck_data, color='red', label='Puck Position')
        ax1.plot(np.array(time_data) - time_data[-1], x_des_paddle_data, color='orange', label='Desired Paddle Position')
        ax1.plot(np.array(time_data) - time_data[-1], x_cur_paddle_data, color='blue', label='Current Paddle Position', linestyle=(0, (3, 10, 1, 10, 1, 10)), linewidth=2)
        ax1.legend(ncol=3, loc='lower center')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('X Position')
        text_x = f'Desired X: {x_des_paddle:.2f} | Current X: {x_cur_paddle:.2f}'
        ax1.text(0.05, 0.9, text_x, transform=ax1.transAxes, fontsize=10, verticalalignment='top')
        ax1.set_ylim(-24,24)
        # Update the bottom subplot
        ax2.clear()
        ax2.axhspan(LIM_LOW, LIM_HIGH, color='black', alpha=0.2)
        ax2.plot(np.array(time_data) - time_data[-1], y_cur_puck_data, color='red', label='Puck Position')
        ax2.plot(np.array(time_data) - time_data[-1], y_des_paddle_data, color='orange',  label='Desired Paddle Position')
        ax2.plot(np.array(time_data) - time_data[-1], y_cur_paddle_data, color='blue', label='Current Paddle Position', linestyle=(0, (3, 10, 1, 10, 1, 10)), linewidth=2)
        ax2.legend(ncol=3, loc='lower center')
        # ax2.axhline(y=LIM_HIGH, color='red', linestyle='--')
        # ax2.axhline(y=LIM_LOW, color='red', linestyle='--')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Y Position')
        text_y = f'Desired Y: {y_des_paddle:.2f} | Current Y: {y_cur_paddle:.2f}'
        ax2.text(0.05, 0.9, text_y, transform=ax2.transAxes, fontsize=10, verticalalignment='top')
        ax2.set_ylim(10,84)


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
            shared_data['cur_paddle_pose'] = (cur_x, cur_y)
            norm_x, norm_y = from_cm_to_norm(cur_x, cur_y)
            message = f"{round(norm_x,3)},{round(norm_y,3)}"
            sock.sendto(message.encode(), (UDP_IP, PORT_UDP_PADDLE_CURRENT))

            if counter >= 10:
                # print(f"Sent {round(norm_x,3)},{round(norm_y,3)}")
                counter = 0
            else:
                counter+=1
        except:
            pass

        time.sleep(0.005)


def file_process(shared_data):
    
    filename = f"{shared_data['filename']}.csv"

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        while(True):

            (cur_x, cur_y) = shared_data['cur_puck_pose']            
            new_x, new_y = shared_data['des_paddle_pose']
            writer.writerow([cur_x, cur_y, new_x, new_y])

def move_process(shared_data):

    # Wait sufficient time so eveything is properly initialized
    time.sleep(2)


    new_x = 0
    new_y = LIM_LOW
    cur_x = 0
    cur_y = LIM_LOW
    

    while(True):

        (cur_x, cur_y) = shared_data['cur_paddle_pose']        
        new_x, new_y = shared_data['des_paddle_pose']



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
    parser.add_argument('-g','--gpu', action='store_true', help='Run on GPU!')
    parser.add_argument('-b', '--board', type= int, help="Board sending events", default=43)


    return parser.parse_args()




def initialize_shared_data():

    shared_data = multiprocessing.Manager().dict()
    shared_data['filename'] = args.filename
    shared_data['action'] = args.action
    shared_data['gpu'] = args.gpu
    shared_data['board'] = args.board
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
    
    recv_paddle_proc = multiprocessing.Process(target=receive_paddle_xy, args=(shared_data,))
    recv_paddle_proc.start()
    
    recv_puck_proc = multiprocessing.Process(target=receive_puck_xy, args=(shared_data,))
    recv_puck_proc.start()

    forwarder_proc = multiprocessing.Process(target=forward_paddle_xy, args=(shared_data,))
    forwarder_proc.start()

    plot_proc = multiprocessing.Process(target=plot_process, args=(shared_data,))
    plot_proc.start()
    
    file_proc = multiprocessing.Process(target=file_process, args=(shared_data,))
    file_proc.start()
    


    param_proc.join()
    read_proc.join()    
    move_proc.join()
    recv_paddle_proc.join()
    recv_puck_proc.join()
    forwarder_proc.join()
    plot_proc.join()
    file_proc.join()





