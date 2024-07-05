import aestream
import numpy as np
import os
import math
import socket
import csv
import time
import pdb

import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
import matplotlib.pyplot as plt
import sys

sys.path.append('../common')
from tools import Dimensions

UDP_IP = '172.16.222.30'  
PORT_UDP_INTIME_DATA = 2626 
PORT_UDP_DELAYED_DATA = 6262  

PLOT_CURVES = True
PLOT_FRAMES = False
MARKER_SIZE = 2
HIGH_DPI = 200
EXTRA_PTS = 200
MAX_ACCEPTABLE_DLYREPS = 50/100 # 10%
MAX_NB_REPEATED_VAL = 50


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        self.kernel = np.load(f"../common/fast_kernel.npy")
        self.ksz = self.kernel.shape[0]
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=self.ksz, bias=False)
        self.conv.weight.data = torch.FloatTensor(self.kernel).unsqueeze(0).unsqueeze(0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x) 
        return x

def count_repeated_elements(arr):
    count = 0
    chain_len = max(MAX_NB_REPEATED_VAL,int(len(arr)*1/100))
    for i in range(chain_len, len(arr)):
        repeated = True
        for j in range(1, chain_len):
            if arr[i][0] != arr[i - j][0] or arr[i][1] != arr[i - j][1]:
                repeated = False
        if repeated:
            count+=1
    print(f"{count}/{len(arr)} (i.e. {round(count/len(arr)*100,2)}%) repeated elements (with chain len = {chain_len})")
    return count/len(arr)

def clean_fname(fname):

    # Split the string from the right side at the last slash
    # and take the last part of the split result
    fname = fname.rsplit('/', 1)[-1]

    # Split the filename at the dot and take the first part
    fname = fname.split('.')[0]

    print(fname)

    return fname

def moving_average(data, window_size):
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

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
        plt.tight_layout()
        plt.savefig(f'images/{fname}_speed_histogram.png', format='png', dpi=HIGH_DPI)
        plt.clf()


    max_speed = round(np.max(full_speed),3)
    mean_speed = round(np.mean(full_speed),3)
    mode_value = round(bins[np.argmax(hist)],3)

    print(f"Max: {max_speed} | Mean: {mean_speed} | Mode: {mode_value}")

    return max_speed, mean_speed, mode_value



def delayed_process(shared_data):
    

    dim_fl = shared_data['res_x']
    dim_fw = shared_data['res_y']

    x_coord = 0
    y_coord = 0

    if not shared_data['gpu']:
        print("Receiving X,Y from SpiNNaker")

        in_port = shared_data['board']*100+87
        print(f"\n\n\n\n\n{in_port} !!!!!!!\n\n\n\n\n")
       

        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        with aestream.UDPInput((dim_fl, dim_fw), device = 'cpu', port=in_port) as stream1:
                    
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
                x_coord = int(x_norm*dim_fl/100)
                y_coord = int(y_norm*dim_fw/100)

                # print(f"In Time: Got {x_coord},{y_coord}")
                shared_data['delayed_pose'] = (x_coord, y_coord)
                
            except socket.timeout:
                pass
            
            
            if shared_data['done_storing_data']:
                break
    
    print("Stopped receiving coordinates from GPU/SpiNNaker")    


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

    clean_error_x = np.mean(e_x_array)
    clean_error_y = np.mean(e_y_array)


    error = math.sqrt(clean_error_x**2 + clean_error_y**2)


    if PLOT_CURVES and plot_flag:

        # Create figure and subplots
        fig, axs = plt.subplots(2,  figsize=(8, 8))

        axis_scaler = 1.2


        # Subplot 1
        # axs[0].scatter(synthetic_t, smooth_intime_x, label='Smooth Intime', s=MARKER_SIZE)
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
    
    
        # pdb.set_trace()

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

    print(f"Latency: {latency} [ms]")

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

        plt.tight_layout()

        plt.savefig(f'images/{fname}_t_shift_vs_error.png', format='png', dpi=HIGH_DPI)
        plt.clf()


    return latency, error, t_shift


def write_to_csv(csv_fname, fname, pipeline, latency, error, min_error, max_speed, mean_speed, dlydreps):


    if dlydreps > MAX_ACCEPTABLE_DLYREPS:
        quality = "ko"
    else:
        quality = "ok"
    csv_fname = csv_fname+f"_summary_{quality}.csv"


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

def get_delayed_coordinates(shared_data):

    nb_pts_forgotten = 200
    nb_frames = int(shared_data['nb_frames']*1.2)+EXTRA_PTS+nb_pts_forgotten

    res_x = shared_data['res_x']
    res_y = shared_data['res_y']
    port = shared_data['port']

    print(f"Resolution: {res_x}x{res_y}")

    # Stream events from UDP port 3333 (default)
    frame_array = np.zeros((nb_frames, res_x,res_y))
    time_array = np.zeros((nb_frames))
    delayed_coordinates = np.zeros((nb_frames,2))


    stimuli_on = False
    keep_running = True

    frame_counter = 0
    start_time = time.time()
    sleeper = 0

    with aestream.UDPInput((res_x, res_y), device = 'cpu', port=port) as stream1:
                
        while keep_running:

            cycle_t_i = time.time()

            reading = stream1.read().numpy() 


            if not stimuli_on:
                ev_count = np.sum(reading)
                if ev_count > 0:                    
                    t_first_event = time.time()
                    stimuli_on = True
            else:            
                frame_array[frame_counter, :,:] = reading
                current_time = time.time()
                time_array[frame_counter] = (current_time - start_time) * 1000
                delayed_coordinates[frame_counter, 0] = shared_data['delayed_pose'][0]
                delayed_coordinates[frame_counter, 1] = shared_data['delayed_pose'][1]
                frame_counter += 1
                if frame_counter%1000 == 0:
                    print(f"Storing Frame | Timestamp | Coordinates #{frame_counter}")
                if frame_counter == nb_frames:
                    print(f"Reached {frame_counter} frames")
                    keep_running = False

            sleeper = max(0, 0.001-(time.time()-cycle_t_i))
            time.sleep(sleeper)
    
    print(f"Finish recording after {time.time()-t_first_event} seconds")
    os.system("pkill -f aestream")
    time.sleep(2)


    frame_array = frame_array.transpose(2,1,0)

    # Let's trim the arrays based on number of actual recorded frames
    time_array = time_array[nb_pts_forgotten:frame_counter]
    delayed_coordinates = delayed_coordinates[nb_pts_forgotten:frame_counter]
    frame_array = frame_array[:,:,nb_pts_forgotten:frame_counter]

    shared_data['done_storing_data'] = True

    nb_frame = shared_data['nb_frames'] + EXTRA_PTS

    return frame_array[0:nb_frame], time_array[0:nb_frame], delayed_coordinates[0:nb_frame,:]


def from_frames_to_intime_coordinates(frame_array, time_array, delta_t):


    threshold = 10
    nb_pts = time_array.shape[0]-delta_t

    model = CustomCNN()
    intime_coordinates = np.zeros((nb_pts,2), dtype=int)

    for i in np.linspace(int(delta_t/2), nb_pts-1, nb_pts - int(delta_t/2)).astype(int):
        
        if i%100 == 0:
            print(f"Analyzing frame #{i}")

        sub_frame = frame_array[:,:,i-int(delta_t/2):i+int(delta_t/2)+1]
        in_frame = np.sum(sub_frame, axis=-1)


        if PLOT_FRAMES:
            plt.imsave(f'images/frame_test_input.png', in_frame)
        in_max_index = np.argmax(in_frame)
        in_max_index_2d = np.unravel_index(in_max_index, in_frame.shape)

        out_frame = np.zeros(in_frame.shape)

        in_tensor = torch.tensor(in_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        out_tensor = model(in_tensor)
        out_x = out_tensor.shape[2]
        out_y = out_tensor.shape[3]
        offset = int(model.ksz/2)
        out_frame[offset:offset+out_x, offset:offset+out_y] = out_tensor.squeeze().detach().numpy()

        if PLOT_FRAMES:
            plt.imsave(f'images/frame_test_output.png', out_frame)

        # Find the index of the maximum value
        out_max_index = np.argmax(out_frame)
        out_max_index_2d = np.unravel_index(out_max_index, out_frame.shape)
        
        if out_frame[out_max_index_2d] > threshold:
            intime_coordinates[i,0] = int(out_max_index_2d[0])
            intime_coordinates[i,1] = int(out_max_index_2d[1])
        else:
            intime_coordinates[i,0] = int(intime_coordinates[i-1,0])
            intime_coordinates[i,1] = int(intime_coordinates[i-1,1])


        target_frame = in_frame
        target_frame[intime_coordinates[i,0], intime_coordinates[i,1]] = 3*in_frame[in_max_index_2d]
    
        if PLOT_FRAMES:
            plt.imsave(f'images/frame_test_target.png', target_frame)
        
    return intime_coordinates


def get_coordinates(shared_data, delta_t):

    # This is specific to RECORDINGS
    if shared_data['data_origin'] == "rec":

        frame_array, time_array, delayed_coordinates = get_delayed_coordinates(shared_data)
        intime_coordinates = from_frames_to_intime_coordinates(frame_array, time_array, delta_t)

    else:
        # This is specific to SYNTHETIC DATA
        nb_pts_forgotten = 20
        nb_pts = shared_data['nb_frames']+EXTRA_PTS+nb_pts_forgotten
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
        
        intime_coordinates = intime_coordinates[nb_pts_forgotten:,:]
        delayed_coordinates = delayed_coordinates[nb_pts_forgotten:,:]

        print(f"Done creating coordinate arrays after {round(time.time()-start_t,3)} [s] ")
        shared_data['done_storing_data'] = True
        os.system("pkill -f puck_generator.py")

    
    windowcita = 40
    
    smooth_intime_x = moving_average(intime_coordinates[:,0], windowcita)
    smooth_intime_y = moving_average(intime_coordinates[:,1], windowcita)
    diff = int(windowcita/2)

    final_intime_coordinates = np.zeros((shared_data['nb_frames'],2))
    final_delayed_coordinates = np.zeros((shared_data['nb_frames'],2))

    final_intime_coordinates[:,0] = smooth_intime_x[0:shared_data['nb_frames']]
    final_intime_coordinates[:,1] = smooth_intime_y[0:shared_data['nb_frames']]
    final_delayed_coordinates[:,:] = delayed_coordinates[diff:diff+shared_data['nb_frames'],:]

    print(f"final_intime_coordinates: {final_intime_coordinates.shape}")
    print(f"final_delayed_coordinates: {final_delayed_coordinates.shape}")

    return final_delayed_coordinates, final_intime_coordinates



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


def aestream_process(shared_data):

    time.sleep(2)
    raw_port = shared_data['port']
    if shared_data['gpu']:
        print("Sending Raw events to GPU")
        ip = "172.16.222.28"
        port = 5050
    else:
        print("Sending Raw events to SpiNNaker")
        ip = "172.16.223.122"
        port = 3333

    cmd = ""
    cmd += "/opt/aestream/build/src/aestream "
    cmd += "resolution 1280 720 "
    cmd += "undistortion ~/tabletop/calibration/luts/cam_lut_homography_prophesee.csv "
    cmd += f"output udp 172.16.222.30 {raw_port} {ip} {port} "
    cmd += f"input file {shared_data['fname']}"
    # cmd += f"input prophesee"

    start_time = time.time()
    print(f"Starting Streaming")
    print(cmd)
    os.system(cmd)
    print(f"End of streaming after {time.time()-start_time} seconds")


def ground_truth_process(shared_data):

    time.sleep(0.5)

    if shared_data['gpu']:
        pipeline = 'gpu'
    else:
        pipeline = 'spinnaker'

    fname = clean_fname(shared_data['fname'])
    iname = f"{fname}_{pipeline}"


    delta_t = 9 # bin size in [ms]
    window_size = 40
    nb_shifts = 100
   

    print("Ground Truth Process Starts NOW!")
    delayed_coordinates, intime_coordinates = get_coordinates(shared_data, delta_t)
    delayed_reps = count_repeated_elements(delayed_coordinates)
    max_speed, mean_speed, mode_value = analyze_speed(iname, intime_coordinates, window_size)
    latency, error, t_shift = find_latency_and_error(delayed_coordinates, intime_coordinates, nb_shifts, iname)
    
    real_error = round(error[0],3)
    best_error = round(error[latency],3)


    write_to_csv(shared_data['data_origin'], fname, pipeline, latency, real_error, best_error, max_speed, mean_speed, delayed_reps)

