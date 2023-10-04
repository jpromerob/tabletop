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
import multiprocessing
import time
import socket
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Value  # Import Value for shared variable

# IP and port of the receiver (Computer B)
controller_ip = "172.16.222.30"
controller_port = 5151
plotter_ip = controller_ip
# plotter_ip = "172.16.222.199"
plotter_port = 4000

########################################################################################
#                                    SHARED VARIABLES                                  #
########################################################################################
x_px = multiprocessing.Value('i', 0)
y_px = multiprocessing.Value('i', 0)
x_cm = multiprocessing.Value('d', 0.0)
y_cm = multiprocessing.Value('d', 0.0)


def from_px_to_cm(dim, px_x, px_y):
    '''
    This funciton:
       - converts a px coordinate (camera space) to cm coorinate (robot space)
    '''

    a_x = -17.2/dim.iw
    b_x = 8.6 - a_x*dim.d2ey

    a_y = -28.8/dim.il
    b_y = 29.7-a_y*(dim.d2ex+dim.il)

    x = round(a_x*px_y+b_x,2)
    y = round(a_y*px_x+b_y,2)


    return x, y



def visualize(args, dim, kernel, x_px, y_px):
    '''
    This function:
       - receives data from Camera through AEstream
       - displays raw data
       - draws play area and a marker indicating Puck's current (x, y)

    '''
    
    cv2.namedWindow('Air-Hockey Display')
    black = np.zeros((dim.fl,dim.fw,3))
    frame = black

    field, line, goals, circles, radius = get_shapes(dim, args.vis_scale)
    red = (0, 0, 255)
    draw_kernel = False
    k_sz = len(kernel)
    x_k = int(dim.fl/2) - int(k_sz/2)
    y_k = int(dim.fw/2) - int(k_sz/2)
    
    with aestream.UDPInput((dim.fl, dim.fw), device = 'cpu', port=args.port_raw) as original:
                    
        while True:

            # Read Raw data
            orig_out = np.squeeze(original.read().numpy())

            # Visualize raw data
            frame = black
            if draw_kernel:
                frame[x_k:x_k+k_sz,y_k:y_k+k_sz,2] = 100*kernel
            frame[:,:,1] = orig_out
            image = cv2.resize(frame.transpose(1,0,2), (math.ceil(dim.fl*args.vis_scale),math.ceil(dim.fw*args.vis_scale)), interpolation = cv2.INTER_AREA)
            

            # Super Impose Play Area Features
            corners = np.array(field, np.int32)
            cv2.polylines(image, [corners], isClosed=True, color=red, thickness=1)
            for goal in goals:
                corners = np.array(goal, np.int32)
                cv2.polylines(image, [corners], isClosed=True, color=red, thickness=1)
            for cx, cy in circles:
                cv2.circle(image, (cx, cy), radius, color=red, thickness=1)
            
            # Draw a circle where tracked puck is
            cv2.circle(image, (x_px.value*args.vis_scale, y_px.value*args.vis_scale), 3*args.vis_scale, color=(255,0,255), thickness=-2)
            
            cv2.imshow('Air-Hockey Display', image)
            cv2.waitKey(1)



def xy_from_cnn(args, dim, kernel, x_px, y_px, x_cm, y_cm):
    '''
    This function:
       - receives data from SPIF|SpiNNaker through AEstream
       - estimates puck's (x,y) based on output of SCNN
       - sends (x,y) to controller
    '''

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    convolved_img = torch.zeros(1, 1, dim.fl, dim.fw)
    

    x_px.value = int(dim.fl/2)
    y_px.value = int(dim.fw/2)
    old_x_px = x_px.value
    old_y_px = y_px.value
    x_cm.value, y_cm.value = from_px_to_cm(dim, x_px.value, y_px.value)

    k_sz = len(kernel)


    flag_update_old = False
    with aestream.UDPInput((dim.fl, dim.fw), device = 'cpu', port=args.port_cnn) as convolved:
                
        while True:

            convolved_img[0,0,:,:] = convolved.read()
            conv_out = np.squeeze(convolved_img.numpy())
            row_indices, column_indices = np.where(conv_out > 0.9)
            if np.sum(conv_out) > 10:
                
                if len(row_indices)>0 and len(column_indices)>0:
                    flag_update_old = True
                    x_px.value = int(np.mean(row_indices))+int(k_sz/2)     
                    y_px.value = int(np.mean(column_indices))+int(k_sz/2)
            
            # Send new coordinates only if they are sufficiently different
            if math.sqrt((x_px.value-old_x_px)**2+(y_px.value-old_y_px)**2)>=1:

                # Estimate coordinates in robot coordinate frame
                x_cm.value, y_cm.value = from_px_to_cm(dim, x_px.value, y_px.value)
                message = f"{x_cm.value},{y_cm.value}"
                sock.sendto(message.encode(), (controller_ip, controller_port))
                sock.sendto(message.encode(), (plotter_ip, plotter_port))

            if flag_update_old:
                old_x_px = x_px.value
                old_y_px = y_px.value
                flag_update_old = False



def xy_from_xyp(args, dim, kernel, x_px, y_px, x_cm, y_cm):
    '''
    This function:
       - receives data from SPIF|SpiNNaker through AEstream
       - estimates puck's (x,y) based on SCNN projections onto 1D XY array
       - sends (x,y) to controller
    '''

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    x_px.value = int(dim.fl/2)
    y_px.value = int(dim.fw/2)
    old_x_px = x_px.value
    old_y_px = y_px.value
    x_cm.value, y_cm.value = from_px_to_cm(dim, x_px.value, y_px.value)

    k_sz = len(kernel)
    width_cnn = (dim.fl-k_sz+1)
    height_cnn = (dim.fw-k_sz+1)
    k_margin = int(k_sz/2)

    with aestream.UDPInput((width_cnn+height_cnn,1), device = 'cpu', port=args.port_xyp) as xy_coords:
                    
        while True:

            xy_coords_out = xy_coords.read().numpy()
            
            x_array = xy_coords_out[0:width_cnn,0]
            y_array = np.transpose(xy_coords_out[width_cnn:,0])
            x_idx = np.where(x_array > 0)
            y_idx = np.where(y_array > 0)
            if np.sum(x_idx)>0 and np.sum(y_idx)>0:
                x_px.value = int(np.mean(x_idx)+k_margin)
                y_px.value = int(np.mean(y_idx)+k_margin)
            
            # Send new coordinates only if they are sufficiently different
            if math.sqrt((x_px.value-old_x_px)**2+(y_px.value-old_y_px)**2)>=1:

                # Estimate coordinates in robot coordinate frame
                x_cm.value, y_cm.value = from_px_to_cm(dim, x_px.value, y_px.value)
                message = f"{x_cm.value},{y_cm.value}"
                sock.sendto(message.encode(), (controller_ip, controller_port))
                sock.sendto(message.encode(), (plotter_ip, plotter_port))

            old_x_px = x_px.value
            old_y_px = y_px.value


def plot_live():
    '''
    This function:
       - plots in 'real-time' the current Puck's (x,y) in robot coordinate frame

    '''

    ax1 = None
    ax2 = None

    # Global lists for x_data and y_data
    x_data = []
    y_data = []

    fig, (ax1, ax2) = plt.subplots(2, 1)

    # Define a function to update the plots
    def animate(i):

        global x_cm, y_cm

        x_data.append(x_cm.value)
        y_data.append(y_cm.value)

        # Keep only the latest 'max_data_points' data points
        if len(x_data) > 40:
            x_data.pop(0)
            y_data.pop(0)

        # Update the x and y subplots
        ax1.clear()
        ax1.plot(x_data, color = 'g')
        ax1.set_title('Robot X')
        ax1.set_ylim(-20,20)  # Set Y-axis limits for the first subplot
        ax2.clear()
        ax2.plot(y_data, color = 'b')
        ax2.set_title('Robot Y')
        ax2.set_ylim(20,80)  # Set Y-axis limits for the second subplot
        
    ani = animation.FuncAnimation(fig, animate, interval=1, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-pr', '--port-raw', type= int, help="Port for events", default=3330)
    parser.add_argument('-pc', '--port-cnn', type= int, help="Port for events", default=3331)
    parser.add_argument('-pp', '--port-xyp', type= int, help="Port for events", default=3334)
    parser.add_argument('-vs', '--vis-scale', type=int, help="Visualization scale", default=1)
    parser.add_argument('-md', '--mode', type= str, help="Mode: cnn|xyp", default="xyp")
    parser.add_argument('-lp', '--live-plot', action='store_true', help="Live Plot")


    return parser.parse_args()

if __name__ == '__main__':


    args = parse_args()
        
    dim = Dimensions.load_from_file('../common/homdim.pkl')
    kernel = np.load("../common/kernel.npy")


    # Create two separate processes, passing the parameters as arguments
    if args.mode == "xyp":
        consolidation = multiprocessing.Process(target=xy_from_xyp, args=(args, dim, kernel,x_px,y_px,x_cm, y_cm,))
    elif args.mode == "cnn":
        consolidation = multiprocessing.Process(target=xy_from_cnn, args=(args, dim, kernel,x_px,y_px,x_cm, y_cm,))
    else:
        print("Wrong Mode")
        quit()
    consolidation.daemon = True

    visualization = multiprocessing.Process(target=visualize, args=(args, dim, kernel,x_px,y_px,))
    visualization.daemon = True

    if args.live_plot:
        print("We are plotting in Real Time")
        live_plotting = multiprocessing.Process(target=plot_live)
        live_plotting.daemon = True
        live_plotting.start()


    consolidation.start()
    visualization.start()

    while True:
        time.sleep(10)
    