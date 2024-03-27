import torch
import aestream
import time
import cv2
import math
import sys
import argparse
import csv
import os

from PIL import Image
import numpy as np
import time
import pdb

import socket


# Define constants for UDP IP and port
UDP_IP = '172.16.222.30'
PORT_UDP_TARGET = 6262  # Choose an appropriate port number

def get_model_size(model):
    num_params = sum(p.numel() for p in model.parameters())
    size_bytes = num_params * 4  # Assuming each parameter is a 32-bit float (4 bytes)
    size_gb = size_bytes / (1024 ** 3)  # Convert bytes to gigabytes
    return size_gb

class ControlNet(torch.nn.Module):
    def __init__(self, N, M): #N: x size, M: y size (from robot's perspective)
        super(ControlNet, self).__init__()
        self.N = N
        self.M = M
        self.input_size = N * M
        self.middle_size = N * M

        # Define the parameters of the network
        self.middle_layer = torch.nn.Linear(self.input_size, self.middle_size)
        self.output_layer_x = torch.nn.Linear(self.middle_size, N)  # Nx1 output layer
        self.output_layer_y = torch.nn.Linear(self.middle_size, M)  # Mx1 output layer

        # Initialize weights
        self.middle_layer.weight.data.zero_()  # Fill middle layer weights with zeros
        self.output_layer_x.weight.data.zero_()  # Fill output layer N weights with zeros
        self.output_layer_y.weight.data.zero_()  # Fill output layer M weights with zeros

        # List to store tuples of connections
        self.in_mid = []

        # Connectivity calculation
        p_gap = 0.3 / 2

        mirror = int(M / 2)
        gap = int(M * p_gap)
        middle_left = mirror - gap
        base_left = mirror - 2 * gap
        middle_right = mirror + gap
        base_right = mirror + 2 * gap

        print(f"Connectivity")
        pts = 0
        for i in range(N):  # height
            for j in range(M):  # width

                # Far pitch
                if j <= mirror:
                    self.in_mid.append((j, i, M - 1 - int(j * base_left / mirror), int((i - N / 2) * j / mirror + N / 2)))
                    pts += 1
                elif j < middle_right:
                    self.in_mid.append((j, i, M - 1 - (j - mirror + base_left), i))
                    pts += 1
                # elif j < base_right:
                #     self.in_mid.append((j, i, j, i))
                else:
                    # self.in_mid.append((j, i, j, i))
                    # self.in_mid.append((j, i, 0, int(N/2)))
                    self.in_mid.append((j, i, M-1, int(N/2)))
                    pts += 1

        print(f"We have {pts} non-zero connections")


        for i in range(pts):
            try:
                input_index = self.in_mid[i][1] * M + self.in_mid[i][0]
                middle_index = self.in_mid[i][3] * M + self.in_mid[i][2]
                y_index = self.in_mid[i][0]
                x_index = self.in_mid[i][1]
                self.middle_layer.weight.data[:, input_index] = 0
                self.middle_layer.weight.data[middle_index, input_index] = 4
                self.output_layer_x.weight.data[x_index, input_index] = 1
                self.output_layer_y.weight.data[y_index, input_index] = 1
            except:
                pdb.set_trace()

        print(f"Done with weights")

    def forward(self, x):
        x = x.reshape(-1, self.input_size)
        x = self.middle_layer(x)
        output_N = self.output_layer_x(x)
        output_M = self.output_layer_y(x)
        x = x.view(-1, self.N, self.M)
        return x, output_N, output_M





def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-p', '--port', type= int, help="Port for events", default=5050)
    parser.add_argument('-s', '--scale', type=int, help="Image scale", default=1)
    parser.add_argument('-l', '--length', type=int, help="Image length", default=255)
    parser.add_argument('-w', '--width', type=int, help="Image width", default=164)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()


    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to GPU
    model = ControlNet(args.width, args.length).to(device)

    size_gb = get_model_size(model)
    print(f"Model size: {size_gb:.2f} gigabytes")


    # Set model to evaluation mode (no gradients)
    model.eval()


    res_y = args.length
    res_x = args.width

    window_name = 'Airhockey Display'
    cv2.namedWindow(window_name)

    # Stream events from UDP port 3333 (default)
    frame = np.zeros((res_y+1,res_x+1,3))
    black = np.zeros((res_y+1,res_x+1,3))
    new_l = math.ceil((res_y+1)*args.scale)
    new_w = math.ceil((res_x+1)*args.scale)

    k_sz = 16

    x_coord = int(res_x/2)
    y_coord = int(res_y)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    with torch.no_grad():
         
        with aestream.UDPInput((res_y, res_x), device = 'cpu', port=args.port) as stream1:
                    
            while True:

                reading = stream1.read()

                # frame[0:res_y,0:res_x,1] =  reading


                reshaped_data = torch.tensor(np.transpose(reading), dtype=torch.float32).unsqueeze(0)
                device_input_data = reshaped_data.to(device)
                out_m_dev, out_x_dev, out_y_dev = model(device_input_data)
                out_m = out_m_dev.cpu().squeeze().numpy()
                out_x = out_x_dev.cpu().squeeze().numpy()
                out_y = out_y_dev.cpu().squeeze().numpy()

                out_frame = np.transpose(out_m, (1, 0))
                # frame[0+k_sz:res_y,0+k_sz:res_x,1] = out_frame[0:res_y-k_sz, 0:res_x-k_sz]
                frame = black.copy()
                frame[0:res_y,0:res_x,1] = out_frame
                frame[0:res_y,res_x,0] = out_y
                frame[res_y,0:res_x,0] = out_x

                idx_x = np.where(out_x>0.5)
                idx_y = np.where(out_y>0.5)


                try:
                    if len(idx_x[0])>0 and len(idx_y[0]) > 0:
                        x_coord = int(np.mean(idx_x))
                        y_coord = int(np.mean(idx_y))
                except:
                    pass
                

                y_norm = x_coord/res_x*100
                x_norm = y_coord/res_y*100

                # print(f"{x_norm},{y_norm}")

                data = "{},{}".format(x_norm, y_norm).encode()
                sock.sendto(data, (UDP_IP, PORT_UDP_TARGET))

                circ_center =  (x_coord, y_coord)

                cv2.circle(frame, circ_center, 3, (0,0,255), thickness=1)
                image = cv2.resize(frame.transpose(1,0,2), (new_l, new_w), interpolation = cv2.INTER_AREA)
                
                cv2.imshow(window_name, image)
                
                cv2.waitKey(1)
                
    sock.close()
