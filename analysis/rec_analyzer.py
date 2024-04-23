import aestream
import numpy as np
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import sys
import socket
import multiprocessing
import argparse
import pdb
import os
import csv

sys.path.append('../common')
from tools import Dimensions


UDP_IP = '172.16.222.30'  
PORT_UDP_DELAYED_DATA = 6262  

PLOT_CURVES = True
PLOT_FRAMES = False

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


def moving_average(data, window_size):
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')

def analyze_speed(fname, online_coordinates, window_size):


    # Smoothing the signal using a moving average filter
    on_x = online_coordinates[:,0]
    on_y = online_coordinates[:,1]
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

        # Plot speed along X axis
        axs[0].plot(on_x, 'k')
        axs[0].plot(smooth_on_x, 'r')
        axs[0].set_title('X Position')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('X')

        # Plot speed along Y axis
        axs[1].plot(on_y, 'k')
        axs[1].plot(smooth_on_y, 'g')
        axs[1].set_title('Y Position')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Y')

        # Plot overall speed
        axs[2].plot(speed_x, 'r')
        axs[2].plot(speed_y, 'g')
        axs[2].plot(full_speed, 'b')
        axs[2].set_title('Overall Speed')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Speed')

        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.savefig(f'images/{fname}_speed_profiles.png')

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
        plt.savefig(f'images/{fname}_speed_histogram.png')


    max_speed = round(np.max(full_speed),3)
    mean_speed = round(np.mean(full_speed),3)
    mode_value = round(bins[np.argmax(hist)],3)

    print(f"Max: {max_speed} | Mean: {mean_speed} | Mode: {mode_value}")

    return max_speed, mean_speed, mode_value



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

    
def get_frames_timestamps_and_coordinates(shared_data):

    offrame_nb = 20
    nb_frames = int(shared_data['nb_frames']*1.2)+offrame_nb

    res_x = shared_data['res_x']
    res_y = shared_data['res_y']
    port = shared_data['port']

    print(f"Resolution: {res_x}x{res_y}")

    # Stream events from UDP port 3333 (default)
    frame_array = np.zeros((nb_frames, res_x,res_y))
    time_array = np.zeros((nb_frames))
    online_coordinates = np.zeros((nb_frames,2))


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
                online_coordinates[frame_counter, 0] = shared_data['delayed_pose'][0]
                online_coordinates[frame_counter, 1] = shared_data['delayed_pose'][1]
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
    time_array = time_array[0:frame_counter]
    online_coordinates = online_coordinates[0:frame_counter]
    frame_array = frame_array[:,:,0:frame_counter]

    shared_data['done_storing_data'] = True

    nb_frame = shared_data['nb_frames'] + offrame_nb

    return frame_array[0:nb_frame], time_array[0:nb_frame], online_coordinates[0:nb_frame,:]


def get_offline_coordinates(frame_array, time_array, delta_t):


    threshold = 10
    nb_pts = time_array.shape[0]-delta_t

    model = CustomCNN()
    offline_coordinates = np.zeros((nb_pts,2), dtype=int)

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
            offline_coordinates[i,0] = int(out_max_index_2d[0])
            offline_coordinates[i,1] = int(out_max_index_2d[1])
        else:
            offline_coordinates[i,0] = int(offline_coordinates[i-1,0])
            offline_coordinates[i,1] = int(offline_coordinates[i-1,1])


        target_frame = in_frame
        target_frame[offline_coordinates[i,0], offline_coordinates[i,1]] = 3*in_frame[in_max_index_2d]
    
        if PLOT_FRAMES:
            plt.imsave(f'images/frame_test_target.png', target_frame)
        
    # Remove datapoints for which X and Y are zero 
    idx_counter = 0
    # print(idx_counter)
    # while(True):
    #     if offline_coordinates[idx_counter,0] != 0 and offline_coordinates[idx_counter,1] != 0:
    #         break
    #     else:
    #         idx_counter += 1
    

    return offline_coordinates[idx_counter:,:]


def evaluate_latency(fname, online_x, online_y, offline_x, offline_y, t):

    if t > 0:
        new_offline_x = offline_x[0:-t]
        new_offline_y = offline_y[0:-t]
    else:
        new_offline_x = offline_x
        new_offline_y = offline_y

    new_online_x = online_x[t:]
    new_online_y = online_y[t:]


    e_x_array = np.sqrt((new_online_x-new_offline_x)**2)
    e_y_array = np.sqrt((new_online_y-new_offline_y)**2)

    e_x = np.mean(e_x_array)
    e_y = np.mean(e_y_array)
    std_x = np.std(e_x_array)
    std_y = np.std(e_y_array)


    clean_x = e_x_array[(e_x_array<e_x+std_x) & (e_x_array>e_x-std_x)]
    clean_y = e_y_array[(e_y_array<e_y+std_y) & (e_y_array>e_y-std_y)]

    clean_error_x = np.mean(clean_x)
    clean_error_y = np.mean(clean_y)


    error = round(math.sqrt(clean_error_x**2 + clean_error_y**2),3)


    if PLOT_CURVES:

        # Create figure and subplots
        fig, axs = plt.subplots(2,  figsize=(8, 8))

        axis_scaler = 1.2

        # Subplot 1
        axs[0].plot(new_online_x, label='Online')
        axs[0].plot(new_offline_x, label='Offline')
        axs[0].set_ylim(0, int(256*axis_scaler))  # Set y-axis limit
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('X (Pixel Space)')
        axs[0].legend()

        # Subplot 2
        axs[1].plot(new_online_y, label='Online')
        axs[1].plot(new_offline_y, label='Offline')
        axs[1].set_ylim(0, int(165*axis_scaler))  # Set y-axis limit
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Y (Pixel Space)')
        axs[1].legend()

        plt.suptitle(f'Delta t = {t} [ms] --> Error = {error} [mm]')  # Add super title

        # Adjust layout
        plt.tight_layout()

        # Show plot

        if t == 0:
            plt.savefig(f'images/{fname}_x_y_on_off_original.png')
        plt.savefig(f'images/{fname}_x_y_on_off_best.png')
        

        plt.close(fig)
    
    return error


def find_latency_and_error(online_coordinates, offline_coordinates, nb_shifts, fname):

    # pdb.set_trace()

    last_element = int(len(online_coordinates)*0.95)

    online_x = online_coordinates[1:last_element, 0]
    offline_x = offline_coordinates[1:last_element, 1]
    online_y = online_coordinates[1:last_element, 1]
    offline_y = offline_coordinates[1:last_element, 0]

    
    error = np.zeros(nb_shifts)
    for t in range(nb_shifts-1, -1, -1):     

        error[t] = evaluate_latency(fname, online_x, online_y, offline_x, offline_y, t)

    latency = np.argmin(error)

    evaluate_latency(fname, online_x, online_y, offline_x, offline_y, latency)


    return latency, error[0], error[latency]


def write_to_csv(csv_fname, fname, pipeline, latency, error, min_error, max_speed, mean_speed, mode_value):
    # Check if the file exists
    file_exists = os.path.isfile(csv_fname)

    # Open the file in append mode
    with open(csv_fname, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header if the file is newly created
        if not file_exists:
            writer.writerow(['Recording', 'Pipeline', 'Latency', 'Error', 'MinError', 'MaxSpeed', 'MeanSpeed', 'ModeSpeed'])

        # Write the new line
        writer.writerow([fname, pipeline, latency, error, min_error, max_speed, mean_speed, mode_value])


def ground_truth_process(shared_data):

    if shared_data['gpu']:
        pipeline = 'gpu'
    else:
        pipeline = 'spinnaker'

    fname = clean_fname(shared_data['fname'])
    iname = f"{fname}_{pipeline}"


    delta_t = 9 # bin size in [ms]
    window_size = 40
    nb_shifts = 100


    frame_array, time_array, online_coordinates = get_frames_timestamps_and_coordinates(shared_data)

    offline_coordinates = get_offline_coordinates(frame_array, time_array, delta_t)


    np.save("online.npy", online_coordinates)
    np.save("offline.npy", offline_coordinates)


    max_speed, mean_speed, mode_value = analyze_speed(fname, offline_coordinates, window_size)
    latency, error, min_error = find_latency_and_error(online_coordinates, offline_coordinates, nb_shifts, iname)

    write_to_csv("rec_summary.csv", fname, pipeline, latency, error, min_error, max_speed, mean_speed, mode_value)

    error_mm = int(error/shared_data['hs'])
    min_error_mm = int(min_error/shared_data['hs'])
    print(f"Latency: {latency} ms | Error[t=0]: {error} [mm]")



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

def clean_fname(fname):

    # Split the string from the right side at the last slash
    # and take the last part of the split result
    fname = fname.rsplit('/', 1)[-1]

    # Split the filename at the dot and take the first part
    fname = fname.split('.')[0]

    print(fname)

    return fname

def initialize_shared_data(args):

    shared_data = multiprocessing.Manager().dict()

    dim = Dimensions.load_from_file('../common/homdim.pkl')
    shared_data['res_x'] = dim.fl
    shared_data['res_y'] = dim.fw
    shared_data['hs'] = dim.hs

    shared_data['fname']=args.fname
    shared_data['gpu'] = args.gpu
    shared_data['port'] = args.port
    shared_data['nb_frames'] = args.nb_frames
    shared_data['board'] = args.board

    shared_data['done_storing_data'] = False
    shared_data['delayed_pose'] = (0,0)

    


    return shared_data

def parse_args():

    parser = argparse.ArgumentParser(description='Display From AEstream')

    parser.add_argument('-n', '--nb-frames', type= int, help="Max number of frames", default=10000)
    parser.add_argument('-f', '--fname', type= str, help="File Name", default="synthetic")
    parser.add_argument('-g','--gpu', action='store_true', help='Run on GPU!')
    parser.add_argument('-b', '--board', type= int, help="Board sending events", default=43)
    parser.add_argument('-p', '--port', type= int, help="Port for events coming from GPU|SpiNNaker", default=5050)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    shared_data = initialize_shared_data(args)

    ae_proc = multiprocessing.Process(target=aestream_process, args=(shared_data,))
    ae_proc.start()

    fl_proc = multiprocessing.Process(target=delayed_process, args=(shared_data,))
    fl_proc.start()

    # gt_proc = multiprocessing.Process(target=ground_truth_process, args=(shared_data,))
    # gt_proc.start()

    ground_truth_process(shared_data)

    # gt_proc.join()
    fl_proc.join()    
    ae_proc.join()

