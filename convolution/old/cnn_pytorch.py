import torch
import torch.nn as nn
import random
import numpy as np
import math
import time
import aestream
import cv2
import pdb
from tools import add_markers, get_dimensions, get_shapes
import argparse
import csv
import os
import matplotlib.pyplot as plt



import socket
import time
import random

# IP and port of the receiver (Computer B)
receiver_ip = "172.16.222.30"
receiver_port = 5151



def make_kernel_circle(r, k_sz,weight, kernel):
    # pdb.set_trace()
    var = int((k_sz+1)/2-1)
    a = np.arange(0, 2 * math.pi, 0.01)
    dx = np.round(r * np.sin(a)).astype("uint32")
    dy = np.round(r * np.cos(a)).astype("uint32")
    kernel[var + dx, var + dy] = weight

def make_whole_kernel(k_sz):

    # The one in the video (using original recordings)
    scaler = 0.01
    max_puck_d = 38
    pos_w = 1
    neg_w = -pos_w * 0.50
    gen_w = neg_w * 0.35
    print(k_sz)
    th = 2

    kernel = gen_w*scaler*np.ones((k_sz, k_sz), dtype=np.float32)

    pos_radi = [21,12,5]
    for r in pos_radi:
        for i in np.arange(r-th+1, r+1):
            make_kernel_circle(i, k_sz, pos_w*scaler, kernel) # 38px


    # neg_radi = [17,9,3]
    # for r in neg_radi:
    #     for i in np.arange(r-th+1, r+1):
    #         make_kernel_circle(i, k_sz, neg_w*scaler, kernel) # 38px

    print(f"sum kernel: {np.sum(kernel)}")

    custom_kernel = torch.from_numpy(kernel)



    cmap = plt.cm.get_cmap('viridis')
    plt.imshow(kernel, cmap=cmap, interpolation='nearest')
    colorbar = plt.colorbar()
    colorbar.set_label('Color Scale')
    plt.savefig("kernel.png")

    return custom_kernel

class CustomCNN(nn.Module):
    def __init__(self, custom_kernel):
        super(CustomCNN, self).__init__()
        self.custom_kernel = nn.Parameter(custom_kernel, requires_grad=False)  # Make the kernel a learnable parameter

    def forward(self, x):
        x = nn.functional.conv2d(x, self.custom_kernel.unsqueeze(0).unsqueeze(0))  # Apply convolution
        return x

    

# output_image = custom_cnn(input_image)


def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-p', '--port', type= int, help="Port for events", default=5151)
    parser.add_argument('-s', '--scale', type=int, help="Image scale", default=1)
    parser.add_argument('-ks', '--kernel-size', type=int, help="Kernel Size", default=61)

    return parser.parse_args()

if __name__ == '__main__':


    args = parse_args()
    k_sz = args.kernel_size
    custom_kernel = make_whole_kernel(k_sz)
    custom_cnn = CustomCNN(custom_kernel)
    cv2.namedWindow('Airhockey Display')

    # pdb.set_trace()

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Stream events from UDP port 3333 (default)
    black = np.zeros((640,480,3))
    frame = black


    l, w, ml, mw, dlx, dly = get_dimensions()
    field, line, goals, circles, radius = get_shapes(l, w, ml, mw, dlx, dly, args.scale)
    red = (0, 0, 255)

    # pdb.set_trace()

    input_image = torch.zeros(1, 1, 640, 480)

    avg_row_idx = 10
    avg_col_idx = 10
    x_k = int((640-k_sz)/2)+1
    y_k = int((480-k_sz)/2)+1
    with aestream.UDPInput((640, 480), device = 'cpu', port=args.port) as stream1:
                
        while True:

            input_image[0,0,:,:] = stream1.read()
            output_image = custom_cnn(input_image)

            output_np = np.squeeze(output_image.numpy())
            # pdb.set_trace()
            frame = black
            frame[0:640,0:480,1] = input_image.numpy()
            frame[x_k:x_k+k_sz,y_k:y_k+k_sz,2] = 100*custom_kernel.numpy()
            frame[int(k_sz/2):640-int(k_sz/2),int(k_sz/2):480-int(k_sz/2),0] = output_np
            
            row_indices, column_indices = np.where(output_np > 0.8)
            if np.sum(output_np)> 100:
                
                if len(row_indices)>0 and len(column_indices)>0:
                    avg_row_idx = int(np.mean(row_indices))+int(k_sz/2)
                    avg_col_idx = int(np.mean(column_indices))+int(k_sz/2)
                    
                
            
            
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
