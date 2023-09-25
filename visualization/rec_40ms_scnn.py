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
receiver_ip = "172.16.222.30"
receiver_port = 5151


def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-p0', '--port-0', type= int, help="Port for events", default=3330)
    parser.add_argument('-p1', '--port-1', type= int, help="Port for events", default=3331)
    parser.add_argument('-vs', '--vis-scale', type=int, help="Visualization scale", default=1)

    return parser.parse_args()

if __name__ == '__main__':


    args = parse_args()
        
    # Load the Dimensions object from the file
    dim = Dimensions.load_from_file('../common/homdim.pkl')
    
    # pdb.set_trace()

    cv2.namedWindow('Airhocket Display')

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Stream events from UDP port 3333 (default)
    black = np.zeros((dim.fl,dim.fw,3), dtype=np.uint8)
    frame = black


    field, line, goals, circles, radius = get_shapes(dim, args.vis_scale)
    red = (0, 0, 255)

    # pdb.set_trace()

    original_img = torch.zeros(1, 1, dim.fl, dim.fw)
    convolved_img = torch.zeros(1, 1, dim.fl, dim.fw)

    avg_row_idx = 10
    avg_col_idx = 10
    kernel = np.load("../common/kernel.npy")
    k_sz = len(kernel)
    print(f"Kernel {k_sz} px")
    x_k = int((dim.fl-k_sz)/2)+1
    y_k = int((dim.fw-k_sz)/2)+1



    
    # Define the video codec, frames per second (FPS), frame size, and video filename
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
    fps = 60.0  # Frames per second
    frame_size = (dim.fl*args.vis_scale, dim.fw*args.vis_scale)  # Frame size (adjust as needed)
    video_filename = f'z_output_video_x{args.vis_scale}.avi'

    # Create a VideoWriter object to save the video
    out = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)


    with aestream.UDPInput((dim.fl, dim.fw), device = 'cpu', port=args.port_0) as original:
        with aestream.UDPInput((dim.fl, dim.fw), device = 'cpu', port=args.port_1) as convolved:
                    
            while True:

                # pdb.set_trace()
                original_img[0,0,:,:] = original.read()
                orig_out = (original_img * 255.0).clamp(0, 255).to(torch.uint8).squeeze().numpy()

                convolved_img[0,0,:,:] = convolved.read()
                conv_out = (convolved_img * 255.0).clamp(0, 255).to(torch.uint8).squeeze().numpy()

                frame = black
                # frame[x_k:x_k+k_sz,y_k:y_k+k_sz,2] = 100*kernel
                frame[:,:,1] = orig_out
                frame[int(k_sz/2):dim.fl-int(k_sz/2),int(k_sz/2):dim.fw-int(k_sz/2),0] = conv_out[0:dim.fl-k_sz+1,0:dim.fw-k_sz+1]
                
                conv_out[0:int(dim.fl/4),:] = np.zeros((int(dim.fl/4),dim.fw))
                row_indices, column_indices = np.where(conv_out > 0.1)
                if np.sum(conv_out) > 5:
                    
                    if len(row_indices)>0 and len(column_indices)>0:
                        avg_row_idx = int(np.mean(row_indices))+int(k_sz/2)     
                        avg_col_idx = int(np.mean(column_indices))+int(k_sz/2)
                        # print(f"({avg_row_idx},{avg_col_idx})")
                
                
                image = cv2.resize(frame.transpose(1,0,2), (math.ceil(dim.fl*args.vis_scale),math.ceil(dim.fw*args.vis_scale)), interpolation = cv2.INTER_AREA)
                # pdb.set_trace()
                

                # Ignore the 1/4 of play field on opponents side
                # if avg_row_idx >= dim.fl*1/4:
                
                # Draw Tracker
                image = cv2.circle(image, (avg_row_idx*args.vis_scale, avg_col_idx*args.vis_scale), 5*args.vis_scale, color=(255,0,255), thickness=-2)

                # Send Coordinates
                message = f"{avg_row_idx},{avg_col_idx}"
                sock.sendto(message.encode(), (receiver_ip, receiver_port))

                # Define the four corners of the field
                corners = np.array(field, np.int32)
                image = cv2.polylines(image, [corners], isClosed=True, color=red, thickness=1)

                for goal in goals:
                    corners = np.array(goal, np.int32)
                    image = cv2.polylines(image, [corners], isClosed=True, color=red, thickness=1)

                for cx, cy in circles:
                    image = cv2.circle(image, (cx, cy), radius, color=red, thickness=1)
                image = cv2.line(image, line[0], line[1], color=red, thickness=1)
                

                cv2.imshow('Airhocket Display', image)
                # pdb.set_trace()
                
                out.write(image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


            # Release the VideoWriter object
            out.release()

            # Release other resources
            cv2.destroyAllWindows()