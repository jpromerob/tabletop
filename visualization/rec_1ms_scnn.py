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
    

    original_img = torch.zeros(1, 1, dim.fl, dim.fw)
    convolved_img = torch.zeros(1, 1, dim.fl, dim.fw)

    max_nb_frames = 5000
    orig_acc = np.zeros((dim.fl, dim.fw, max_nb_frames))
    conv_acc = np.zeros((dim.fl, dim.fw, max_nb_frames))


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


    frame_counter = 0
    start_time = time.time()
    with aestream.UDPInput((dim.fl, dim.fw), device = 'cpu', port=args.port_0) as original:
        with aestream.UDPInput((dim.fl, dim.fw), device = 'cpu', port=args.port_1) as convolved:
                    
            while True:

                # pdb.set_trace()
                original_img[0,0,:,:] = original.read()
                convolved_img[0,0,:,:] = convolved.read()
                orig_out = (original_img * 255.0).clamp(0, 255).to(torch.uint8).squeeze().numpy()
                conv_out = (convolved_img * 255.0).clamp(0, 255).to(torch.uint8).squeeze().numpy()
                if np.sum(orig_out) > 5:
                    orig_acc[:,:,frame_counter] = orig_out
                    conv_acc[:,:,frame_counter] = conv_out
                    frame_counter +=1
                    if frame_counter%1000 == 0:
                        print(f"{frame_counter} frames")
                    if frame_counter >= max_nb_frames:
                        break
            

# Measure the end time
end_time = time.time()
elapsed_time = end_time - start_time

print(f"End of Accumulation after {elapsed_time:.2f} seconds")

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

# Iterate through the frames and write to the video
for i in range(max_nb_frames):
    if flag_orig:
        frame[:,:,1] = orig_acc[:, :, i]
    if flag_conv:
        frame[int(k_sz/2):dim.fl-int(k_sz/2),int(k_sz/2):dim.fw-int(k_sz/2),0] = conv_acc[0:dim.fl-k_sz+1,0:dim.fw-k_sz+1, i]
                

    # Write the frame to the video
    for _ in range(frame_duration_ms // (1000 // frame_rate)):
        out.write(frame.transpose(1,0,2))

# Release the VideoWriter object
out.release()

# Optional: Show a message when the video creation is complete
print(f"Video saved as {output_file}")
