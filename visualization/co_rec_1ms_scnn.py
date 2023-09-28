import torch
import torch.nn as nn
import random
import numpy as np
import math
import pdb
import aestream
import cv2
import sys
import os
sys.path.append('../common')
from tools import Dimensions, get_shapes
import argparse
import matplotlib.pyplot as plt
import socket
import random
import time

from datetime import datetime

# IP and port of the receiver (Computer B)
receiver_ip = "172.16.222.30"
receiver_port = 5151


def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-p0', '--port-0', type= int, help="Port for events", default=3330)
    parser.add_argument('-p1', '--port-1', type= int, help="Port for events", default=3331)
    parser.add_argument('-vs', '--vis-scale', type=int, help="Visualization scale", default=1)
    parser.add_argument('-m', '--mode', type= str, help="Mode", default="both")
    parser.add_argument('-at', '--acc-time', type= int, help="Accumulation Time", default=1)
    parser.add_argument('-d', '--duration', type= int, help="Duration", default=1)
    parser.add_argument('-l', '--length', type=int, help="Image length", default=640)
    parser.add_argument('-w', '--width', type=int, help="Image width", default=480)

    return parser.parse_args()

if __name__ == '__main__':


    args = parse_args()
    res_x = args.length
    res_y = args.width
        
    # Load the Dimensions object from the file
    dim = Dimensions.load_from_file('../common/homdim.pkl')
    

    max_nb_frames = int(args.duration*1000/args.acc_time)
    print(f"Max # frames: {max_nb_frames}")
    # pdb.set_trace()

    original_img = torch.zeros(1, 1, res_x, res_y)
    convolved_img = torch.zeros(1, 1, res_x, res_y)

    orig_out = np.zeros((max_nb_frames,res_x, res_y))
    conv_out = np.zeros((max_nb_frames,res_x, res_y))


    frame_counter = 0
    read_counter = 0
    beginning = time.perf_counter()
    elapsed_time = 0
    with aestream.UDPInput((res_x, res_y), device = 'cpu', port=args.port_0) as stream_0:
        with aestream.UDPInput((res_x, res_y), device = 'cpu', port=args.port_1) as stream_1:
            
            start_time = time.perf_counter()
            last_time = time.perf_counter()
            while True:

                try:
                    # pdb.set_trace()
                    original_img[0,0,:,:] += stream_0.read()
                    convolved_img[0,0,:,:] += stream_1.read()
                    
                    now = time.perf_counter()
                    elapsed_time += now - start_time

                    if elapsed_time > args.acc_time/1000:
                        # print(elapsed_time[-1])
                        orig_out[frame_counter,:,:] = (original_img * 255.0).clamp(0, 255).to(torch.uint8).squeeze().numpy()
                        conv_out[frame_counter,:,:] = (convolved_img * 255.0).clamp(0, 255).to(torch.uint8).squeeze().numpy()
                        original_img = torch.zeros(1, 1, res_x, res_y)
                        convolved_img = torch.zeros(1, 1, res_x, res_y)
                        frame_counter += 1
                        start_time = now
                        
                    read_counter +=1 
                    last_time = now
                except:
                    pdb.set_trace()

                if frame_counter >= max_nb_frames:
                    print("C'est fini")
                    break

    whole_duration = time.perf_counter() - beginning
    print(f"Whole recording lasted: {whole_duration:.3f} seconds")

    print(f"{np.sum(orig_out)} events")
    print(f"{np.sum(conv_out)} events")
    mean = 1e6*elapsed_time/read_counter
    print(f"Average reading time: {mean:.6f} [us]")



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
    black = np.zeros((res_x,res_y,3), dtype=np.uint8)
    
    old_frame = black
    new_frame = black



    # Get the current date and time
    current_datetime = datetime.now()
    output_file = f"{current_datetime.year}_{current_datetime.month:02d}_{current_datetime.day:02d}_{current_datetime.hour:02d}_{current_datetime.minute:02d}_{args.acc_time}ms.avi"
    print(f"File name: {output_file}")
    # pdb.set_trace()

    # Define video properties
    frame_rate = 25  # Frames per second (adjust as needed)
    frame_duration_ms = 40  # Duration of each frame in milliseconds

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, (res_x, res_y))

    orig_out = orig_out.transpose((1, 2, 0))
    conv_out = conv_out.transpose((1, 2, 0))

    empty_frame_counter = 0
    for i in range(max_nb_frames):
        if np.sum(orig_out[:,:,i]) == 0:
            empty_frame_counter +=1

    print(f"{empty_frame_counter}/{max_nb_frames} frames are empty")

    # Iterate through the frames and write to the video
    for i in range(max_nb_frames):
        if flag_orig:
            new_frame[:,:,1] = orig_out[:, :, i]
        if flag_conv:
            new_frame[int(k_sz/2):res_x-int(k_sz/2),int(k_sz/2):res_y-int(k_sz/2),0] = conv_out[0:res_x-k_sz+1,0:res_y-k_sz+1, i]
                    
        # pdb.set_trace()
        if np.sum(new_frame)==0:
            if i > 0:
                new_frame = old_frame
        else:        
            old_frame = new_frame

        # Write the frame to the video
        for _ in range(frame_duration_ms // (1000 // frame_rate)):
            out.write(new_frame.transpose(1,0,2))

    # Release the VideoWriter object
    out.release()
    os.system(f"mv {output_file} videos/")

    # Optional: Show a message when the video creation is complete
    print(f"Video saved as {output_file}")