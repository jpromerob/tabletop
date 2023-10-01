import torch
import aestream
import time
import cv2
import pdb
import numpy as np
import math
import argparse
import sys
import socket
sys.path.append('../common')
from tools import Dimensions, get_shapes

# pdb.set_trace()

# IP and port of the receiver (Computer B)
controller_ip = "172.16.222.30"
controller_port = 5151
plotter_ip = controller_ip
# plotter_ip = "172.16.222.199"
plotter_port = 4000

def from_px_to_cm(dim, px_x, px_y):

    a_x = -17.2/dim.iw
    b_x = 8.6 - a_x*dim.d2ey

    a_y = -28.8/dim.il
    b_y = 29.7-a_y*(dim.d2ex+dim.il)

    x = round(a_x*px_y+b_x,2)
    y = round(a_y*px_x+b_y,2)


    return x, y

def get_high_idx(data):

    # Find the indices where elements are greater than 0
    indices = np.where(data > 0)[0]

    highest_index = 0
    if indices.size > 0:
        highest_index = np.max(indices)

    return highest_index

parser = argparse.ArgumentParser(description='Visualizer')
parser.add_argument('-p1', '--port1', type= int, help="Port for events", default=1987)
parser.add_argument('-p2', '--port2', type= int, help="Port for events", default=1988)
parser.add_argument('-s', '--scale', type= float, help="Image scale", default=1)
parser.add_argument('-x', '--width', type= int, help="X res = width", default=64)
parser.add_argument('-y', '--height', type= int, help="Y res = height", default=64)

args = parser.parse_args()

dim = Dimensions.load_from_file('../common/homdim.pkl')

e_port_1 = args.port1
e_port_2 = args.port2
width = args.width
height = args.height


cv2.namedWindow('Airhocket Display')
# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Stream events from UDP port 3333 (default)
black = np.zeros((width+1,height+1,3))
frame = np.zeros((width+1,height+1,3))

mean_x = int(width/2)
mean_y = int(height/2)
new_robot_x, new_robot_y = from_px_to_cm(dim, mean_x, mean_y)
with aestream.UDPInput((width, height), device = 'cpu', port=e_port_1) as stream1:
    with aestream.UDPInput((width+height,1), device = 'cpu', port=e_port_2) as stream2:
            
        while True:
                
                aux = stream1.read().numpy()
                xy_aux = stream2.read().numpy()

                frame[0:width+1,0:height+1,:] = black

                frame[1:width+1,1:height+1,1] = aux # Provides a (width, height) tensor

                x_array = xy_aux[0:width,0]
                y_array = np.transpose(xy_aux[width:,0])

                frame[0:width,0,2] = x_array
                frame[0,0:height,2] = y_array

                x_idx = np.where(x_array > 0)
                y_idx = np.where(y_array > 0)
                if np.sum(x_idx)>0 and np.sum(y_idx)>0:
                    mean_x = np.mean(x_idx) # x2 since WTA works with downsampled values
                    mean_y = np.mean(y_idx) # x2 since WTA works with downsampled values
                
                new_robot_x, new_robot_y = from_px_to_cm(dim, mean_x, mean_y)
                message = f"{new_robot_x},{new_robot_y}"
                sock.sendto(message.encode(), (controller_ip, controller_port))
                sock.sendto(message.encode(), (plotter_ip, plotter_port))

                nx = args.scale*args.width/width
                ny = args.scale*args.height/height

                image = cv2.resize(frame.transpose(1,0,2), (math.ceil(width*nx),math.ceil(height*ny)), interpolation = cv2.INTER_AREA)
                # pdb.set_trace()
                cv2.circle(image, (int(mean_x*nx), int(mean_y*ny)), int(3*args.scale), color=(255,0,255), thickness=-2)
                
                cv2.imshow('Durin On Board', image)
                cv2.waitKey(1)

    


        
