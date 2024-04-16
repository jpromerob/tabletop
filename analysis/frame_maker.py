import aestream
import numpy as np
import math
import time
import sys
import socket
import multiprocessing
sys.path.append('../common')
from tools import Dimensions
import argparse
import pdb


UDP_IP = '172.16.222.30'  
PORT_UDP_PUCK_CURRENT = 6161 

def ground_truth_process(shared_data):

    nb_frames = shared_data['nb_frames']

    res_x = shared_data['res_x']
    res_y = shared_data['res_y']
    port = shared_data['port']

    print(f"Resolution: {res_x}x{res_y}")

    # Stream events from UDP port 3333 (default)
    frame_array = np.zeros((res_x,res_y,nb_frames))
    time_array = np.zeros((nb_frames))
    coordinate_array = np.zeros((nb_frames,2))


    stimuli_on = False
    keep_running = True
    empty_frame_counter = 0

    frame_counter = 0
    start_time = time.time()

    with aestream.UDPInput((res_x, res_y), device = 'cpu', port=port) as stream1:
                
        while keep_running:

            reading = stream1.read().numpy() 

            ev_count = np.sum(reading)

            if ev_count > 0:
                empty_frame_counter = 0
            elif stimuli_on:
                empty_frame_counter += 1
                if empty_frame_counter == 10:
                    stimuli_on == False
                    print(f"{empty_frame_counter} empty consecutive frames")
                    break

            if not stimuli_on:
                if ev_count > 0:
                    print("New stimuli")
                    stimuli_on = True
                    old_frame = frame_array
            else:            
                frame_array[0:res_x,0:res_y,frame_counter] = reading
                current_time = time.time()
                time_array[frame_counter] = (current_time - start_time) * 1000
                coordinate_array[frame_counter, 0] = shared_data['cur_puck_pose'][0]
                coordinate_array[frame_counter, 1] = shared_data['cur_puck_pose'][1]
                frame_counter += 1
                if frame_counter%100 == 0:
                    print(f"Frame #{frame_counter}")
                if frame_counter == nb_frames:
                    print(f"Reached {frame_counter} frames")
                    keep_running = False

            time.sleep(0.001)
        


    # Plotting the image
    frame_array = frame_array.transpose(1,0,2)


    print(f"Storing Frames ...")

    np.save('frame_array.npy', frame_array[:,:,0:frame_counter])
    np.save('time_array.npy', time_array[0:frame_counter])
    np.save('coordinate_array.npy', coordinate_array[0:frame_counter])

    print(f"{frame_counter} frames stored")

    shared_data['finish'] = True



def full_loop_process(shared_data):

    dim = Dimensions.load_from_file('../common/homdim.pkl')

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.020)

    # Bind the socket to the receiver's IP and port
    sock.bind((UDP_IP, PORT_UDP_PUCK_CURRENT))


    while True:
        
        # Receive data from the sender
        try:
            data, sender_address = sock.recvfrom(2048)
            # Decode the received data and split it into x and y
            x_norm, y_norm = map(float, data.decode().split(","))
            # print(f"Got {x_norm}, {y_norm}")
            x_coord = int(x_norm*dim.fl/100)
            y_coord = int(y_norm*dim.fw/100)
            # print(f"Got {x_coord}, {y_coord}")
            shared_data['cur_puck_pose'] = (x_coord, y_coord)

            
        except socket.timeout:
            pass
        
        if shared_data['finish']:
            break


def initialize_shared_data(args):

    shared_data = multiprocessing.Manager().dict()

    dim = Dimensions.load_from_file('../common/homdim.pkl')
    shared_data['res_x'] = dim.fl
    shared_data['res_y'] = dim.fw
    shared_data['nb_frames'] = 10000
    shared_data['port'] = args.port
    shared_data['finish'] = False
    shared_data['cur_puck_pose'] = (0,0)
    # pdb.set_trace()


    return shared_data

def parse_args():

    parser = argparse.ArgumentParser(description='Display From AEstream')

    parser.add_argument('-p', '--port', type= int, help="Port for events", default=5050)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    shared_data = initialize_shared_data(args)


    gt_proc = multiprocessing.Process(target=ground_truth_process, args=(shared_data,))
    gt_proc.start()


    fl_proc = multiprocessing.Process(target=full_loop_process, args=(shared_data,))
    fl_proc.start()


    gt_proc.join()
    fl_proc.join()    

