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
import time

# IP and port of the receiver (Computer B)
receiver_ip = "172.16.222.30"
receiver_port = 5151


def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-p0', '--port-0', type= int, help="Port for events", default=3330)
    parser.add_argument('-p1', '--port-1', type= int, help="Port for events", default=3331)
    parser.add_argument('-vs', '--vis-scale', type=int, help="Visualization scale", default=1)
    parser.add_argument('-m', '--mode', type= str, help="Mode", default="both")

    return parser.parse_args()

if __name__ == '__main__':


    args = parse_args()
        
    # Load the Dimensions object from the file
    dim = Dimensions.load_from_file('../common/homdim.pkl')
    

    max_frames = 1000
    orig_img = np.zeros((max_frames,dim.fl, dim.fw))
    conv_img = np.zeros((max_frames,dim.fl, dim.fw))


    frame_counter = 0
    elapsed_time = np.zeros((max_frames,1))
    with aestream.UDPInput((dim.fl, dim.fw), device = 'cpu', port=args.port_0) as stream_0:
        with aestream.UDPInput((dim.fl, dim.fw), device = 'cpu', port=args.port_1) as stream_1:
            
            last_time = time.perf_counter()
            while True:

                orig_img[frame_counter,:,:] = stream_0.read().numpy()
                conv_img[frame_counter,:,:] = stream_1.read().numpy()

                now = time.perf_counter()
                elapsed_time[frame_counter] = (now - last_time)
                last_time = now
                frame_counter +=1 
                if frame_counter >= max_frames:
                    break


    print(f"{np.sum(orig_img)} events")
    print(f"{np.sum(conv_img)} events")
    mean = 1e6*np.mean(elapsed_time)
    std = 1e6*np.std(elapsed_time)

    print(f"Elapsed time: {mean:.6f} [us] +- {std}")
