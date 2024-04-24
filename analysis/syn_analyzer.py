import aestream
import numpy as np
import time
import torch
import torch.nn as nn
import math
import sys
import socket
from datetime import datetime
import multiprocessing
from matplotlib.patches import Circle
import argparse
import pdb
import os
import csv

import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
import matplotlib.pyplot as plt



sys.path.append('../common')
from tools import Dimensions


UDP_IP = '172.16.222.30'  
PORT_UDP_INTIME_DATA = 2626 
PORT_UDP_DELAYED_DATA = 6262  

PLOT_CURVES = True
PLOT_FRAMES = False
MARKER_SIZE = 2
HIGH_DPI = 500

def count_repeated_elements(arr):
    count = 0
    chain_len = max(50,int(len(arr)*1/100))
    for i in range(chain_len, len(arr)):
        repeated = True
        for j in range(1, chain_len):
            if arr[i][0] != arr[i - j][0] or arr[i][1] != arr[i - j][1]:
                repeated = False
        if repeated:
            count+=1
    print(f"{count}/{len(arr)} (i.e. {count/len(arr)*100}%) repeated elements (with chain len = {chain_len})")
    return count/len(arr)

def moving_average(data, window_size):
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

def clean_fname(fname):

    # Split the string from the right side at the last slash
    # and take the last part of the split result
    fname = fname.rsplit('/', 1)[-1]

    # Split the filename at the dot and take the first part
    fname = fname.split('.')[0]

    print(fname)

    return fname

def analyze_speed(fname, delayed_coordinates, window_size):


    # Smoothing the signal using a moving average filter
    on_x = delayed_coordinates[:,1]
    on_y = delayed_coordinates[:,0]
    smooth_on_x = moving_average(on_x, window_size)
    smooth_on_y = moving_average(on_y, window_size)

    on_x = on_x[window_size//2:-window_size//2+1]
    on_y = on_y[window_size//2:-window_size//2+1]


    speed_x = smooth_on_x[1:]-smooth_on_x[:-1]
    speed_y = smooth_on_y[1:]-smooth_on_y[:-1]

    # full_speed = moving_average(np.sqrt(speed_x**2+speed_y**2), window_size)
    full_speed = np.sqrt(speed_x**2+speed_y**2)


    if PLOT_CURVES:

        # Create a figure and three subplots
        fig, axs = plt.subplots(3, 1, figsize=(8, 10))

        axis_scaler = 1.2

        # Plot speed along X axis
        axs[0].scatter(np.arange(len(on_x)), on_x, color='k', s=MARKER_SIZE)
        axs[0].scatter(np.arange(len(smooth_on_x)),smooth_on_x, color='r', s=MARKER_SIZE)
        axs[0].set_xlabel('Time [ms]')
        axs[0].set_ylabel('X Position (Pixel Space)')
        axs[0].set_ylim(0, int(256*axis_scaler))

        # Plot speed along Y axis
        axs[1].scatter(np.arange(len(on_y)), on_y, color='k', s=MARKER_SIZE)
        axs[1].scatter(np.arange(len(smooth_on_y)), smooth_on_y, color='g', s=MARKER_SIZE)
        axs[1].set_xlabel('Time [ms]')
        axs[1].set_ylabel('Y Position (Pixel Space)')
        axs[1].set_ylim(0, int(165*axis_scaler))

        # Plot overall speed
        # axs[2].scatter(np.arange(len(speed_x)), speed_x, color='r', s=MARKER_SIZE)
        # axs[2].scatter(np.arange(len(speed_y)), speed_y, color='g', s=MARKER_SIZE)
        # axs[2].scatter(np.arange(len(full_speed)), full_speed, color='b', s=MARKER_SIZE)

        axs[2].plot(speed_x, color='r')
        axs[2].plot(speed_y, color='g')
        axs[2].plot(full_speed, color='b')

        axs[2].set_xlabel('Time [ms]')
        axs[2].set_ylabel('Speed [m/s]')

        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.savefig(f'images/{fname}_speed_profiles.png', format='png', dpi=HIGH_DPI)

        plt.clf()



    # Compute histogram
    hist, bins = np.histogram(full_speed, bins=20)


    if PLOT_CURVES:

        # Plot histogram
        plt.bar(bins[:-1], hist, width=np.diff(bins), color='blue', alpha=0.7)
        plt.title('Histogram of Speed')
        plt.xlabel('Speed in [m/s]')
        plt.ylabel('# Occurrences')
        plt.xlim(0,5) 
        plt.ylim(0,1000) 
        plt.grid(True)
        plt.savefig(f'images/{fname}_speed_histogram.png', format='png', dpi=HIGH_DPI)
        plt.clf()


    max_speed = round(np.max(full_speed),3)
    mean_speed = round(np.mean(full_speed),3)
    mode_value = round(bins[np.argmax(hist)],3)

    print(f"Max: {max_speed} | Mean: {mean_speed} | Mode: {mode_value}")

    return max_speed, mean_speed, mode_value


def evaluate_latency(fname, delayed_x, delayed_y, intime_x, intime_y, t, plot_flag):

    if t > 0:
        new_intime_x = intime_x[0:-t]
        new_intime_y = intime_y[0:-t]
    else:
        new_intime_x = intime_x
        new_intime_y = intime_y

    new_delayed_x = delayed_x[t:]
    new_delayed_y = delayed_y[t:]


    e_x_array = np.sqrt((new_delayed_x-new_intime_x)**2)
    e_y_array = np.sqrt((new_delayed_y-new_intime_y)**2)

    e_x = np.mean(e_x_array)
    e_y = np.mean(e_y_array)
    std_x = np.std(e_x_array)
    std_y = np.std(e_y_array)


    # clean_x = e_x_array[(e_x_array<e_x+std_x) & (e_x_array>e_x-std_x)]
    # clean_y = e_y_array[(e_y_array<e_y+std_y) & (e_y_array>e_y-std_y)]

    clean_error_x = np.mean(e_x_array)
    clean_error_y = np.mean(e_y_array)


    error = math.sqrt(clean_error_x**2 + clean_error_y**2)


    if PLOT_CURVES and plot_flag:

        # Create figure and subplots
        fig, axs = plt.subplots(2,  figsize=(8, 8))

        axis_scaler = 1.2

        # Subplot 1
        axs[0].scatter(np.arange(len(new_intime_x)), new_intime_x, label='Intime', s=MARKER_SIZE)
        axs[0].scatter(np.arange(len(new_delayed_x)), new_delayed_x, label='Delayed', s=MARKER_SIZE)
        axs[0].set_ylim(0, int(256*axis_scaler))  # Set y-axis limit
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('X (Pixel Space)')
        axs[0].legend()

        # Subplot 2
        axs[1].scatter(np.arange(len(new_intime_y)), new_intime_y, label='Intime', s=MARKER_SIZE)
        axs[1].scatter(np.arange(len(new_delayed_y)), new_delayed_y, label='Delayed', s=MARKER_SIZE)
        axs[1].set_ylim(0, int(165*axis_scaler))  # Set y-axis limit
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Y (Pixel Space)')
        axs[1].legend()

        plt.suptitle(f'Delta t = {t} [ms] --> Error = {round(error,1)} [mm]')  # Add super title

        # Adjust layout
        plt.tight_layout()

        # Show plot

        if t == 0:
            shift_label = "original"
        else:
            shift_label = "best"
        plt.savefig(f'images/{fname}_x_y_comparison_{shift_label}.png', format='png', dpi=HIGH_DPI)
        plt.clf()
        

        plt.close(fig)
    
    return error


def find_latency_and_error(delayed_coordinates, intime_coordinates, nb_shifts, fname):


    last_element = int(len(delayed_coordinates)*0.95)

    delayed_x = delayed_coordinates[1:last_element, 0]
    intime_x = intime_coordinates[1:last_element, 1]
    delayed_y = delayed_coordinates[1:last_element, 1]
    intime_y = intime_coordinates[1:last_element, 0]

    
    error = np.zeros(nb_shifts)
    t_shift = np.zeros(nb_shifts)

    for t in range(nb_shifts-1, -1, -1):     

        t_shift[t] = t
        error[t] = evaluate_latency(fname, delayed_x, delayed_y, intime_x, intime_y, t, False)

    latency = np.argmin(error)
    evaluate_latency(fname, delayed_x, delayed_y, intime_x, intime_y, latency, True)
    evaluate_latency(fname, delayed_x, delayed_y, intime_x, intime_y, 0, True)


    return latency, error, t_shift


def write_to_csv(csv_fname, fname, pipeline, latency, error, min_error, max_speed, mean_speed, dlydreps):
    # Check if the file exists
    file_exists = os.path.isfile(csv_fname)

    # Open the file in append mode
    with open(csv_fname, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header if the file is newly created
        if not file_exists:
            writer.writerow(['Recording', 'Pipeline', 'Latency', 'Error', 'MinError', 'MaxSpeed', 'MeanSpeed', 'DlydReps'])

        # Write the new line
        writer.writerow([fname, pipeline, latency, error, min_error, max_speed, mean_speed, dlydreps])


def intime_process(shared_data):

    dim = Dimensions.load_from_file('../common/homdim.pkl')

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # sock.settimeout(0.001)

    # Bind the socket to the receiver's IP and port
    sock.bind((UDP_IP, PORT_UDP_INTIME_DATA))


    while True:
        
        # Receive data from the sender
        try:
            data, sender_address = sock.recvfrom(2048)
            # Decode the received data and split it into x and y
            x_norm, y_norm = map(float, data.decode().split(","))
            x_coord = int(x_norm*dim.fl/100)
            y_coord = int(y_norm*dim.fw/100)

            # print(f"In Time: Got {x_coord},{y_coord}")
            shared_data['intime_pose'] = (x_coord, y_coord)

            
        except socket.timeout:
            pass
        
        if shared_data['done_storing_data']:
            break
    
    print("Stopped receiving coordinates from Event Generator")


def delayed_process(shared_data):
    
    dim = Dimensions.load_from_file('../common/homdim.pkl')

    x_coord = 0
    y_coord = 0

    if not shared_data['gpu']:
        print("Receiving X,Y from SpiNNaker")

        in_port = shared_data['board']*100+87
       

        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        with aestream.UDPInput((dim.fl, dim.fw), device = 'cpu', port=in_port) as stream1:
                    
            while True:

                reading = stream1.read().numpy() 
                idx_x = np.where(reading[:,0]>0.5)
                idx_y = np.where(reading[:,1]>0.5)

                try:
                    if len(idx_x[0])>0 and len(idx_y[0]) > 0:
                        x_coord = int(np.mean(idx_x))
                        y_coord = int(np.mean(idx_y))
                except:
                    pass       
                                
                shared_data['delayed_pose'] =  (x_coord, y_coord)

                if shared_data['done_storing_data']:
                    break
    else:
        print("Receiving X,Y from GPU")

        # Create a UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.020)

        # Bind the socket to the receiver's IP and port
        sock.bind((UDP_IP, PORT_UDP_DELAYED_DATA))



        while True:
            
            # Receive data from the sender
            try:
                data, sender_address = sock.recvfrom(2048)

                # Decode the received data and split it into x and y
                x_norm, y_norm = map(float, data.decode().split(","))
                x_coord = int(x_norm*dim.fl/100)
                y_coord = int(y_norm*dim.fw/100)

                # print(f"In Time: Got {x_coord},{y_coord}")
                shared_data['delayed_pose'] = (x_coord, y_coord)
                
            except socket.timeout:
                pass
            
            
            if shared_data['done_storing_data']:
                break
    
    print("Stopped receiving coordinates from GPU/SpiNNaker")    

def gen_ev_process(shared_data):

    cmd =""
    cmd += f"python3 ~/tabletop/generation/puck_generator.py "
    cmd += f"-s {shared_data['sparsity']} -d {shared_data['delta']} "
    cmd += f"-ox {shared_data['offx']} -oy {shared_data['offy']} "
    cmd += f" -m {shared_data['gmode']}"
    if shared_data['gpu']:
        print("Launch event generation for GPU")
        cmd += f" -g"
    else:
        print("Launch event generation for SpiNNaker")
        cmd += f""
    
    os.system(cmd)




def ground_truth_process(shared_data):

    time.sleep(5)

    nb_pts_forgotten = 20
    nb_pts = shared_data['nb_frames']+nb_pts_forgotten
    delta_t = 0.001

    intime_coordinates = np.zeros((nb_pts,2))
    delayed_coordinates = np.zeros((nb_pts,2))

    pt_counter = 0

    start_t = time.time()
    while pt_counter < nb_pts:

        cycle_t_i = time.time()
        delayed_coordinates[pt_counter, 0] = shared_data['delayed_pose'][0]
        delayed_coordinates[pt_counter, 1] = shared_data['delayed_pose'][1]
        intime_coordinates[pt_counter, 0] = shared_data['intime_pose'][0]
        intime_coordinates[pt_counter, 1] = shared_data['intime_pose'][1]
        pt_counter += 1
        
        sleeper = max(0, delta_t-(time.time()-cycle_t_i))
        time.sleep(sleeper)
    
    print(f"Done creating coordinate arrays after {round(time.time()-start_t,3)} [s] ")
    shared_data['done_storing_data'] = True
    os.system("pkill -f puck_generator.py")

    print("Ground Truth Process Starts NOW!")

    if shared_data['gpu']:
        pipeline = 'gpu'
    else:
        pipeline = 'spinnaker'

    fname = clean_fname(shared_data['fname'])
    iname = f"{fname}_{pipeline}"


    delta_t = 9 # bin size in [ms]
    window_size = 40
    nb_shifts = 100

    intime_coordinates = intime_coordinates[nb_pts_forgotten:,:]
    delayed_coordinates = delayed_coordinates[nb_pts_forgotten:,:]

    # intime_reps = count_repeated_elements(intime_coordinates)
    delayed_reps = count_repeated_elements(delayed_coordinates)

    max_speed, mean_speed, mode_value = analyze_speed(iname, intime_coordinates, window_size)
    latency, error, t_shift = find_latency_and_error(delayed_coordinates, intime_coordinates, nb_shifts, iname)
    
    if PLOT_CURVES:

        # Find the index of the minimum error
        min_error_index = np.argmin(error)

        plt.scatter(t_shift, error, color='k', s=40)
        plt.scatter(t_shift[min_error_index], error[min_error_index], marker='x', color='r', s=100)
        plt.text(t_shift[min_error_index], error[min_error_index]*0.6, f"Latency = {t_shift[min_error_index]} [ms]", fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))

        
        plt.title('Finding Latency: Time Shift vs Error')
        plt.xlabel('Time Shift in [ms]')
        plt.ylabel('Error inn [mm]')

        # Setting limits
        plt.xlim(0, t_shift[-1]+1)
        plt.ylim(0, 10)

        plt.savefig(f'images/{iname}_t_shift_vs_error.png', format='png', dpi=HIGH_DPI)
        plt.clf()

    real_error = round(error[0],3)
    best_error = round(error[latency],3)

    if delayed_reps > 10/100:
        print(f"Delayed reps: {delayed_reps} --> KO")
        write_to_csv("syn_ko_summary.csv", fname, pipeline, latency, real_error, best_error, max_speed, mean_speed, delayed_reps)
    else:
        print(f"Delayed reps: {delayed_reps} --> OK")
        write_to_csv("syn_ok_summary.csv", fname, pipeline, latency, real_error, best_error, max_speed, mean_speed, delayed_reps)

    real_error_mm = int(real_error/shared_data['hs'])
    best_error_mm = int(best_error/shared_data['hs'])
    print(f"Latency: {latency} ms | Error[t=0]: {real_error_mm} [mm]")



def initialize_shared_data(args):

    shared_data = multiprocessing.Manager().dict()

    dim = Dimensions.load_from_file('../common/homdim.pkl')
    shared_data['res_x'] = dim.fl
    shared_data['res_y'] = dim.fw
    shared_data['hs'] = dim.hs

    shared_data['board'] = args.board

    shared_data['fname'] = args.fname 

    shared_data['offx'] = args.offx
    shared_data['offy'] = args.offy

    shared_data['delta'] = args.delta
    shared_data['sparsity'] = args.sparsity
    shared_data['gpu'] = args.gpu
    shared_data['nb_frames'] = args.nb_frames
    shared_data['gmode'] = args.gmode

    shared_data['done_storing_data'] = False
    shared_data['intime_pose'] = (0,0)
    shared_data['delayed_pose'] = (0,0)

    return shared_data


def parse_args():

    parser = argparse.ArgumentParser(description='Display From AEstream')

    parser.add_argument('-n', '--nb-frames', type= int, help="Max number of frames", default=10000)
    parser.add_argument('-f', '--fname', type= str, help="File Name", default="synthetic")
    parser.add_argument('-m', '--gmode', type= str, help="Generation Mode", default="circle")
    parser.add_argument('-g','--gpu', action='store_true', help='Run on GPU!')
    parser.add_argument('-b', '--board', type= int, help="Board sending events", default=43)

    parser.add_argument('-s', '--sparsity', type= float, help="Sparsity", default=0.6)
    parser.add_argument('-d', '--delta', type= float, help="Delta (puck speed)", default=3.0)
    parser.add_argument('-ox', '--offx', type=float, help="Offset X (percentage)", default=0)
    parser.add_argument('-oy', '--offy', type=float, help="Offset Y (percentage)", default=0)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    shared_data = initialize_shared_data(args)

    delayed_proc = multiprocessing.Process(target=delayed_process, args=(shared_data,))
    delayed_proc.start()

    intime_proc = multiprocessing.Process(target=intime_process, args=(shared_data,))
    intime_proc.start()

    genev_proc = multiprocessing.Process(target=gen_ev_process, args=(shared_data,))
    genev_proc.start()


    # gt_proc = multiprocessing.Process(target=ground_truth_process, args=(shared_data,))
    # gt_proc.start()

    ground_truth_process(shared_data)

    # gt_proc.join()
    delayed_proc.join()  
    intime_proc.join()   
    genev_proc.join()  
 

