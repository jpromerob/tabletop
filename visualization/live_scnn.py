import torch
import torch.nn as nn
import random
import numpy as np
import math
import aestream
import cv2
import sys
sys.path.append('../common')
from tools import add_markers, get_dimensions, get_shapes
import argparse
import matplotlib.pyplot as plt
import socket
import random

# IP and port of the receiver (Computer B)
receiver_ip = "172.16.222.30"
receiver_port = 5151


def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-p0', '--port-0', type= int, help="Port for events", default=3330)
    parser.add_argument('-p1', '--port-1', type= int, help="Port for events", default=3331)
    parser.add_argument('-s', '--scale', type=int, help="Image scale", default=1)

    return parser.parse_args()

if __name__ == '__main__':


    args = parse_args()
    cv2.namedWindow('Airhocket Display')

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Stream events from UDP port 3333 (default)
    black = np.zeros((640,480,3))
    frame = black


    l, w, ml, mw, dlx, dly = get_dimensions()
    field, line, goals, circles, radius = get_shapes(l, w, ml, mw, dlx, dly, args.scale)
    red = (0, 0, 255)

    # pdb.set_trace()

    original_img = torch.zeros(1, 1, 640, 480)
    convolved_img = torch.zeros(1, 1, 640, 480)

    avg_row_idx = 10
    avg_col_idx = 10
    kernel = np.load("../common/kernel.npy")
    k_sz = len(kernel)
    print(f"Kernel {k_sz} px")
    x_k = int((640-k_sz)/2)+1
    y_k = int((480-k_sz)/2)+1
    with aestream.UDPInput((640, 480), device = 'cpu', port=args.port_0) as original:
        with aestream.UDPInput((640, 480), device = 'cpu', port=args.port_1) as convolved:
                    
            while True:

                # pdb.set_trace()
                original_img[0,0,:,:] = original.read()
                orig_out = np.squeeze(original_img.numpy())

                convolved_img[0,0,:,:] = convolved.read()
                conv_out = np.squeeze(convolved_img.numpy())

                frame = black
                frame[x_k:x_k+k_sz,y_k:y_k+k_sz,2] = 100*kernel
                frame[:,:,1] = orig_out
                frame[int(k_sz/2):640-int(k_sz/2),int(k_sz/2):480-int(k_sz/2),0] = conv_out[0:640-k_sz+1,0:480-k_sz+1]
                
                row_indices, column_indices = np.where(conv_out > 0.1)
                if np.sum(conv_out) > 5:
                    
                    if len(row_indices)>0 and len(column_indices)>0:
                        avg_row_idx = int(np.mean(row_indices))+int(k_sz/2)     
                        avg_col_idx = int(np.mean(column_indices))+int(k_sz/2)
                        # print(f"({avg_row_idx},{avg_col_idx})")
                
                
                image = cv2.resize(frame.transpose(1,0,2), (math.ceil(640*args.scale),math.ceil(480*args.scale)), interpolation = cv2.INTER_AREA)
                # pdb.set_trace()
                

                cv2.circle(image, (avg_row_idx*args.scale, avg_col_idx*args.scale), 5*args.scale, color=(255,0,255), thickness=-2)

                # Send Coordinates
                message = f"{avg_row_idx},{avg_col_idx}"
                sock.sendto(message.encode(), (receiver_ip, receiver_port))

                # Define the four corners of the field
                corners = np.array(field, np.int32)
                cv2.polylines(image, [corners], isClosed=True, color=red, thickness=1)

                for goal in goals:
                    corners = np.array(goal, np.int32)
                    cv2.polylines(image, [corners], isClosed=True, color=red, thickness=1)

                for cx, cy in circles:
                    cv2.circle(image, (cx, cy), radius, color=red, thickness=1)
                cv2.line(image, line[0], line[1], color=red, thickness=1)
                

                cv2.imshow('Airhocket Display', image)
                cv2.waitKey(1)

