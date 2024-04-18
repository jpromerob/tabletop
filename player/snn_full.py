import torch
import norse
from norse.torch.module.lif_box import LIFBoxCell, LIFBoxFeedForwardState, LIFBoxParameters
import aestream
import time
import cv2
import math
import sys
import argparse
import csv
import os
import numpy as np
import time
import pdb

import socket
from struct import pack


import sys
sys.path.append('../common')

P_SHIFT = 15
Y_SHIFT = 0
X_SHIFT = 16
NO_TIMESTAMP = 0x80000000



def load_kernel(name):

    scaler = 0.2

    np_kernel = np.load(f"../common/{name}_kernel.npy")*scaler
    k_sz = len(np_kernel)

    # Convert the NumPy array to a PyTorch tensor
    torch_kernel = torch.tensor(np_kernel)

    return torch_kernel.reshape(1, 1, k_sz, k_sz)


# Define constants for UDP IP and port
UDP_IP = '172.16.222.30'
PORT_UDP_TIP_DESIRED = 6262  # Choose an appropriate port number
PORT_UDP_PUCK_CURRENT = 3337  # Choose an appropriate port number

def get_model_size(model):
    num_params = sum(p.numel() for p in model.parameters())
    size_bytes = num_params * 4  # Assuming each parameter is a 32-bit float (4 bytes)
    size_gb = size_bytes / (1024 ** 3)  # Convert bytes to gigabytes
    return size_gb


# Timestep and Sim Duration in seconds
def setParams(tau_mem):

    return LIFBoxParameters(
        tau_mem_inv=torch.tensor(1000/tau_mem),
        v_leak=torch.tensor(-65.0),
        v_th=torch.tensor(-50.0),
        v_reset=torch.tensor(-65.0),
        method='super',
        alpha=torch.tensor(0.)
    )


def get_weight_list_from_conv_to_mapper(mode, Xsz, Ysz, weight):

        p_gap=0.3 / 2
        mirror = int(Ysz / 2)
        gap = int(Ysz * p_gap)
        middle_left = mirror - gap
        base_left = mirror - 2 * gap
        middle_right = mirror + gap
        base_right = mirror + 2 * gap

        weight_list = []

        print(f"Connectivity from Consolidator to Mapper")
        pts = 0
        for i in range(Xsz):  # height
            for j in range(Ysz):  # width

                if mode == "game":
                    # Far pitch
                    if j <= mirror:
                        weight_list.append((j, i, Ysz - 1 - int(j * base_left / mirror), int((i - Xsz / 2) * j / mirror + Xsz / 2), weight))
                    # Just Mirror
                    elif j < middle_right:
                        weight_list.append((j, i, Ysz - 1 - (j - mirror + base_left), i, weight))
                    # Panic! Go Back Home
                    else:
                        weight_list.append((j, i, Ysz-1, int(Xsz/2), weight))
                elif mode == "test":
                    weight_list.append((j, i, j, i, weight))
                else:
                    pass
                pts += 1

        print(f"We have {pts} non-zero connections")

        # pdb.set_trace()

        return weight_list


def get_weight_list_from_input_to_conv(Xsz, Ysz, kernel):



    pass



class ControlNet(torch.nn.Module):
    def __init__(self, in_x_sz, in_y_sz, mode): #x size, y size (from robot's perspective)
        super(ControlNet, self).__init__()

        self.dt = 0.001
        self.mode = mode

        # We have 7 layers:
        #   → Input
        #       ⤷ F-Conv
        #       ⤷ M-Conv
        #       ⤷ S-Conv
        #           ⤷ Mapper
        #               ⤷ X-Projection
        #               ⤷ Y-Projection

        # Input Layer

        self.input_x_sz = in_x_sz
        self.input_y_sz = in_y_sz

        # Convolutional Layers
        self.f_kernel = load_kernel("fast")
        self.m_kernel = load_kernel("medium")
        self.s_kernel = load_kernel("slow")
        self.conv_x_sz = self.input_x_sz
        self.conv_y_sz = self.input_y_sz

        # Mapper Layer
        self.mapper_x_sz = self.conv_x_sz 
        self.mapper_y_sz = self.conv_y_sz 

        # Input Layer
        self.x_proj_sz = self.mapper_x_sz

        # Input Layer
        self.y_proj_sz = self.mapper_y_sz

        # Population sizes as 1D arrays
        self.input_size = self.input_x_sz * self.input_y_sz
        self.conv_size = self.conv_x_sz * self.conv_y_sz
        self.mapper_size = self.conv_size
        self.consolidator_size = self.conv_size
        self.x_proj_size = self.x_proj_sz
        self.y_proj_size = self.y_proj_sz

        # Convolutional Layer: definition of architecture, cell parameters and states
        self.f_conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=self.f_kernel.shape[-1])
        self.f_conv_cell = LIFBoxCell(p=setParams(1), dt=self.dt)
        self.v_f_conv = None

        self.m_conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=self.m_kernel.shape[-1])
        self.m_conv_cell = LIFBoxCell(p=setParams(8), dt=self.dt)
        self.v_m_conv = None

        self.s_conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=self.s_kernel.shape[-1])
        self.s_conv_cell = LIFBoxCell(p=setParams(64), dt=self.dt)
        self.v_s_conv = None

        # Mapper Layer: definition of architecture, cell parameters and states
        self.mapper_layer = torch.nn.Linear(self.conv_size, self.mapper_size)
        self.mapper_cell = LIFBoxCell(p=setParams(1), dt=self.dt)
        self.v_mapper = None

        # X-Projection Layer: definition of architecture, cell parameters and states
        self.output_layer_x = torch.nn.Linear(self.mapper_size, self.x_proj_sz)
        self.out_x_cell = LIFBoxCell(p=setParams(1), dt=self.dt)
        self.v_x_proj = None

        # Define Y-Projection Layer
        self.output_layer_y = torch.nn.Linear(self.mapper_size, self.y_proj_sz) 
        self.out_y_cell = LIFBoxCell(p=setParams(1), dt=self.dt)
        self.v_y_proj = None

        # Initialize weights to ZERO everywhere
        self.mapper_layer.weight.data.zero_()
        self.output_layer_x.weight.data.zero_()
        self.output_layer_y.weight.data.zero_()

        # Set Actual Weights
        self.f_conv.weight.data = self.f_kernel.to(device)
        self.m_conv.weight.data = self.m_kernel.to(device)
        self.s_conv.weight.data = self.s_kernel.to(device)

        # Lists to store tuples of connections
        self.conv_map = get_weight_list_from_conv_to_mapper(self.mode, self.conv_x_sz, self.conv_y_sz, weight=120)
        self.map_xy = []


        for i in range(len(self.conv_map)):
            try:
                input_index = self.conv_map[i][1] * self.input_y_sz + self.conv_map[i][0]
                middle_index = self.conv_map[i][3] * self.input_y_sz + self.conv_map[i][2]
                weight = self.conv_map[i][4]

                y_index = self.conv_map[i][0]
                x_index = self.conv_map[i][1]

                # 2D Perception to 2D Actuation mapping
                self.mapper_layer.weight.data[middle_index, input_index] = weight

                # 2D to XY mapping
                self.output_layer_x.weight.data[x_index, input_index] = 200
                self.output_layer_y.weight.data[y_index, input_index] = 200
            except:
                pdb.set_trace()

        print(f"Done with weights")



    def forward(self, cnn_input):

        f_conv_out = self.f_conv(cnn_input)
        f_conv_out, self.v_f_conv = self.f_conv_cell(f_conv_out, self.v_f_conv)

        m_conv_out = self.m_conv(cnn_input)
        m_conv_out, self.v_m_conv = self.m_conv_cell(m_conv_out, self.v_m_conv)

        s_conv_out = self.s_conv(cnn_input)
        s_conv_out, self.v_s_conv = self.s_conv_cell(s_conv_out, self.v_s_conv)

        tracking_out = torch.zeros((1, self.input_x_sz, self.input_y_sz)).to(device)
        off_ksz = int(self.f_kernel.shape[-1]/2)
        tracking_out[0,off_ksz:off_ksz+f_conv_out.shape[1],off_ksz:off_ksz+f_conv_out.shape[2]] = f_conv_out+m_conv_out+s_conv_out

        mapper_input = tracking_out.reshape(-1, self.mapper_size)

        mapper_out = self.mapper_layer(mapper_input)
        mapper_out, self.v_mapper = self.mapper_cell(mapper_out, self.v_mapper)

        output_x = self.output_layer_x(mapper_out)
        output_x, self.v_x_proj = self.out_x_cell(output_x, self.v_x_proj)

        output_y = self.output_layer_y(mapper_out)
        output_y, self.v_y_proj = self.out_y_cell(output_y, self.v_y_proj)
        
        mapper_out = mapper_out.view(-1, self.input_x_sz, self.input_y_sz)

        return tracking_out, output_x, output_y

def send_tracking_data(sock, avg_x, avg_y):
    data = b""
    polarity = 1
    radius = 3
    offset = int(radius/2)
    for i in range(radius):
        for j in range(radius):
            x = avg_x-offset+i
            y = avg_y-offset+j
            packed = (NO_TIMESTAMP + (polarity << P_SHIFT) + (y << Y_SHIFT) + (x << X_SHIFT))
            data += pack("<I", packed)
    sock.sendto(data, (UDP_IP, PORT_UDP_PUCK_CURRENT))


def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-p', '--port', type= int, help="Port for events", default=5050)
    parser.add_argument('-s', '--scale', type=int, help="Image scale", default=1)
    parser.add_argument('-l', '--length', type=int, help="Image length", default=256)
    parser.add_argument('-w', '--width', type=int, help="Image width", default=165)
    parser.add_argument('-v','--visualize', action='store_true', help='Visuale Outputs')
    parser.add_argument('-m', '--mode', type=str, help="Mode ('game' vs 'test')", default="game")

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    if args.visualize:
        window_name = 'Airhockey Display'
        cv2.namedWindow(window_name)


    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_float32_matmul_precision('high')

    # Move model to GPU
    model = ControlNet(args.width, args.length, args.mode).to(device)

    size_gb = get_model_size(model)
    print(f"Model size: {size_gb:.2f} GB")


    # Set model to evaluation mode (no gradients)
    model = torch.compile(model.eval())


    res_y = args.length
    res_x = args.width


    # Stream events from UDP port 3333 (default)
    frame = np.zeros((res_y+1,res_x+1,3))
    black = np.zeros((res_y+1,res_x+1,3))
    new_l = math.ceil((res_y+1)*args.scale)
    new_w = math.ceil((res_x+1)*args.scale)


    x_coord = int(res_x/2)
    y_coord = int(res_y)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)



    with torch.inference_mode():
         
        with aestream.UDPInput((res_y, res_x), device = 'cpu', port=args.port) as stream1:
                    
            while True:

                reading = stream1.read()

                reshaped_data = torch.tensor(np.transpose(reading), dtype=torch.float32).unsqueeze(0)
                device_input_data = reshaped_data.to(device)
                out_t_dev, out_x_dev, out_y_dev = model(device_input_data)
                out_t = out_t_dev.cpu().squeeze().numpy()
                out_x = out_x_dev.cpu().squeeze().numpy()
                out_y = out_y_dev.cpu().squeeze().numpy()


                out_t_frame = np.transpose(out_t, (1, 0))
                
                frame = black.copy()
                frame[0:out_t_frame.shape[0],0:out_t_frame.shape[1],1] = out_t_frame


                nonzero_indices = np.nonzero(frame)
                if len(nonzero_indices[0]) > 0:
                    avg_x = int(np.mean(nonzero_indices[0]))
                    avg_y = int(np.mean(nonzero_indices[1]))
                    send_tracking_data(sock, avg_x, avg_y)


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


                data = "{},{}".format(x_norm, y_norm).encode()
                sock.sendto(data, (UDP_IP, PORT_UDP_TIP_DESIRED))

                if args.visualize:

                    circ_center =  (x_coord, y_coord)

                    cv2.circle(frame, circ_center, 3, (0,0,255), thickness=1)
                    image = cv2.resize(frame.transpose(1,0,2), (new_l, new_w), interpolation = cv2.INTER_AREA)
                    
                    cv2.imshow(window_name, image)
                    
                    cv2.waitKey(1)
                
    sock.close()
