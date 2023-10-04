import torch
import torch.nn as nn
import random
import numpy as np
import math
import pdb
import aestream
import cv2
import sys
sys.path.append('../common')
from tools import Dimensions, get_shapes
import argparse
import matplotlib.pyplot as plt
import socket
import random

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

def split_and_sum(original_array, block_size):
    n, m = original_array.shape
    result_array = original_array.reshape(n // block_size, block_size, m // block_size, block_size).sum(axis=(1, 3))
    return result_array

def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-p0', '--port-0', type= int, help="Port for events", default=3330)
    parser.add_argument('-p1', '--port-1', type= int, help="Port for events", default=3334)
    parser.add_argument('-vs', '--vis-scale', type=int, help="Visualization scale", default=1)
    parser.add_argument('-md', '--mode', type= str, help="Mode: cnn|xyp", default="xyp")

    return parser.parse_args()

if __name__ == '__main__':


    args = parse_args()

    if args.mode == "xyp":
        full_cnn_xy = True 
    elif args.mode == "cnn":
        full_cnn_xy = False 
    else:
        print("Wrong Mode")
        quit()
        
    # Load the Dimensions object from the file
    dim = Dimensions.load_from_file('../common/homdim.pkl')
    kernel = np.load("../common/kernel.npy")

    
    # pdb.set_trace()

    cv2.namedWindow('Airhocket Display')

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


    black = np.zeros((dim.fl,dim.fw,3))
    frame = black


    field, line, goals, circles, radius = get_shapes(dim, args.vis_scale)
    red = (0, 0, 255)

    # pdb.set_trace()

    original_img = torch.zeros(1, 1, dim.fl, dim.fw)

    

    new_avg_row_idx = int(dim.fl/2)
    new_avg_col_idx = int(dim.fw/2)
    old_avg_row_idx = new_avg_row_idx
    old_avg_col_idx = new_avg_col_idx
    new_robot_x, new_robot_y = from_px_to_cm(dim, new_avg_row_idx, new_avg_col_idx)

    k_sz = len(kernel)
    width_cnn = (dim.fl-k_sz+1)
    height_cnn = (dim.fw-k_sz+1)
    k_margin = int(k_sz/2)
    print(f"Kernel {k_sz} px")

    x_k = int(dim.fl/2) - int(k_sz/2)
    y_k = int(dim.fw/2) - int(k_sz/2)
    flag_update_old = False
    with aestream.UDPInput((dim.fl, dim.fw), device = 'cpu', port=args.port_0) as original:
        with aestream.UDPInput((width_cnn+height_cnn,1), device = 'cpu', port=args.port_1) as xy_coords:
                    
            while True:

                # pdb.set_trace()
                original_img[0,0,:,:] = original.read()
                orig_out = np.squeeze(original_img.numpy())
                xy_coords_out = xy_coords.read().numpy()


                frame = black
                frame[x_k:x_k+k_sz,y_k:y_k+k_sz,2] = 100*kernel
                frame[:,:,1] = orig_out
                
                x_array = xy_coords_out[0:width_cnn,0]
                y_array = np.transpose(xy_coords_out[width_cnn:,0])
                x_idx = np.where(x_array > 0)
                y_idx = np.where(y_array > 0)
                if np.sum(x_idx)>0 and np.sum(y_idx)>0:
                    new_avg_row_idx = int(np.mean(x_idx)+k_margin) # x2 since WTA works with downsampled values
                    new_avg_col_idx = int(np.mean(y_idx)+k_margin) # x2 since WTA works with downsampled values
                
                
                image = cv2.resize(frame.transpose(1,0,2), (math.ceil(dim.fl*args.vis_scale),math.ceil(dim.fw*args.vis_scale)), interpolation = cv2.INTER_AREA)
               


                # Send new coordinates only if they are sufficiently different
                if math.sqrt((new_avg_row_idx-old_avg_row_idx)**2+(new_avg_col_idx-old_avg_col_idx)**2)>=1:

                    # Estimate coordinates in robot coordinate frame
                    new_robot_x, new_robot_y = from_px_to_cm(dim, new_avg_row_idx, new_avg_col_idx)
                    message = f"{new_robot_x},{new_robot_y}"
                    sock.sendto(message.encode(), (controller_ip, controller_port))
                    sock.sendto(message.encode(), (plotter_ip, plotter_port))

                    cv2.circle(image, (new_avg_row_idx*args.vis_scale, new_avg_col_idx*args.vis_scale), 3*args.vis_scale, color=(255,0,255), thickness=-2)
                else:
                    cv2.circle(image, (old_avg_row_idx*args.vis_scale, old_avg_col_idx*args.vis_scale), 3*args.vis_scale, color=(255,0,255), thickness=-2)


                if flag_update_old:
                    old_avg_row_idx = new_avg_row_idx
                    old_avg_col_idx = new_avg_col_idx
                    flag_update_old = False

                # Define the four corners of the field
                corners = np.array(field, np.int32)
                cv2.polylines(image, [corners], isClosed=True, color=red, thickness=1)

                for goal in goals:
                    corners = np.array(goal, np.int32)
                    cv2.polylines(image, [corners], isClosed=True, color=red, thickness=1)

                for cx, cy in circles:
                    cv2.circle(image, (cx, cy), radius, color=red, thickness=1)
                # cv2.line(image, line[0], line[1], color=red, thickness=1)
                
                cv2.imshow('Airhocket Display', image)
                cv2.waitKey(1)

