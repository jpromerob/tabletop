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
    

    max_nb_frames = 1000

    original_img = torch.zeros(1, 1, dim.fl, dim.fw)
    convolved_img = torch.zeros(1, 1, dim.fl, dim.fw)

    orig_out = np.zeros((max_nb_frames,dim.fl, dim.fw))
    conv_out = np.zeros((max_nb_frames,dim.fl, dim.fw))


    frame_counter = 0
    read_counter = 0
    elapsed_time = np.zeros((max_nb_frames*10,1))
    with aestream.UDPInput((dim.fl, dim.fw), device = 'cpu', port=args.port_0) as stream_0:
        with aestream.UDPInput((dim.fl, dim.fw), device = 'cpu', port=args.port_1) as stream_1:
            
            start_time = time.perf_counter()
            last_time = time.perf_counter()
            while True:

                # pdb.set_trace()
                original_img[0,0,:,:] += stream_0.read()
                convolved_img[0,0,:,:] += stream_1.read()
                
                now = time.perf_counter()
                elapsed_time[read_counter] = (now - start_time)

                if elapsed_time[read_counter] > 1e-3:
                    orig_out[frame_counter,:,:] = (original_img * 255.0).clamp(0, 255).to(torch.uint8).squeeze().numpy()
                    conv_out[frame_counter,:,:] = (convolved_img * 255.0).clamp(0, 255).to(torch.uint8).squeeze().numpy()
                    original_img = torch.zeros(1, 1, dim.fl, dim.fw)
                    convolved_img = torch.zeros(1, 1, dim.fl, dim.fw)
                    frame_counter += 1
                    start_time = now
                    
                read_counter +=1 
                last_time = now

                if frame_counter >= max_nb_frames:
                    break


    print(f"{np.sum(orig_out)} events")
    print(f"{np.sum(conv_out)} events")
    mean = 1e6*np.mean(elapsed_time)
    std = 1e6*np.std(elapsed_time)

    print(f"Average reading time: {mean:.6f} [us] +- {std}")


    if args.mode == "both":
        flag_orig = True
        flag_conv = True
    elif args.mode == "orig":
        flag_orig = True
        flag_conv = False
    elif args.mode == "conv":
        flag_orig = False
        flag_conv = True
    else:
        quit()

    kernel = np.load("../common/kernel.npy")
    k_sz = len(kernel)
    # pdb.set_trace()
    black = np.zeros((dim.fl,dim.fw,3), dtype=np.uint8)
    frame = black

    # Define video properties
    output_file = f'output_video_{args.mode}.avi'
    frame_rate = 25  # Frames per second (adjust as needed)
    frame_duration_ms = 40  # Duration of each frame in milliseconds

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, (dim.fl, dim.fw))

    orig_out = orig_out.transpose((1, 2, 0))
    conv_out = conv_out.transpose((1, 2, 0))

    # Iterate through the frames and write to the video
    for i in range(max_nb_frames):
        if flag_orig:
            frame[:,:,1] = orig_out[:, :, i]
        if flag_conv:
            frame[int(k_sz/2):dim.fl-int(k_sz/2),int(k_sz/2):dim.fw-int(k_sz/2),0] = conv_out[0:dim.fl-k_sz+1,0:dim.fw-k_sz+1, i]
                    

        # Write the frame to the video
        for _ in range(frame_duration_ms // (1000 // frame_rate)):
            out.write(frame.transpose(1,0,2))

    # Release the VideoWriter object
    out.release()

    # Optional: Show a message when the video creation is complete
    print(f"Video saved as {output_file}")