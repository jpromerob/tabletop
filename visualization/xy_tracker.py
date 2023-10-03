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
parser.add_argument('-p1', '--port1', type= int, help="Port for events", default=3331)
parser.add_argument('-p2', '--port2', type= int, help="Port for events", default=3334)
parser.add_argument('-s', '--scale', type= float, help="Image scale", default=1)

args = parser.parse_args()

kernel = np.load("../common/kernel.npy")
k_sz = len(kernel)
dim = Dimensions.load_from_file('../common/homdim.pkl')


e_port_1 = args.port1
e_port_2 = args.port2
width = dim.fl
height = dim.fw
width_cnn = (width-k_sz+1)
height_cnn = (height-k_sz+1)
k_margin = int(k_sz/2)


cv2.namedWindow('Airhockey Display')
# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Stream events from UDP port 3333 (default)
black = np.zeros((width+1,height+1,3))
frame = np.zeros((width+1,height+1,3))

mean_x = int(width/2)
mean_y = int(height/2)
new_robot_x, new_robot_y = from_px_to_cm(dim, mean_x, mean_y)
with aestream.UDPInput((width, height), device = 'cpu', port=e_port_1) as stream1:
    with aestream.UDPInput((width_cnn+height_cnn,1), device = 'cpu', port=e_port_2) as stream2:
            
        while True:
                
                aux = stream1.read().numpy()
                xy_aux = stream2.read().numpy()

                frame[0:width+1,0:height+1,:] = black

                frame[1:width+1,1:height+1,1] = aux # Provides a (width, height) tensor

                x_array = xy_aux[0:width_cnn,0]
                y_array = np.transpose(xy_aux[width_cnn:,0])

                frame[k_margin:width_cnn+k_margin,0,2] = x_array
                frame[0,k_margin:height_cnn+k_margin,2] = y_array

                x_idx = np.where(x_array > 0)
                y_idx = np.where(y_array > 0)
                if np.sum(x_idx)>0 and np.sum(y_idx)>0:
                    mean_x = np.mean(x_idx)+k_margin # x2 since WTA works with downsampled values
                    mean_y = np.mean(y_idx)+k_margin # x2 since WTA works with downsampled values
                
                new_robot_x, new_robot_y = from_px_to_cm(dim, mean_x, mean_y)
                message = f"{new_robot_x},{new_robot_y}"
                sock.sendto(message.encode(), (controller_ip, controller_port))
                sock.sendto(message.encode(), (plotter_ip, plotter_port))


                image = cv2.resize(frame.transpose(1,0,2), (math.ceil(width*args.scale),math.ceil(height*args.scale)), interpolation = cv2.INTER_AREA)
                # pdb.set_trace()
                cv2.circle(image, (int(mean_x*args.scale), int(mean_y*args.scale)), int(3*args.scale), color=(255,0,255), thickness=-2)
                
                cv2.imshow('Airhockey Display', image)
                cv2.waitKey(1)

    


        
