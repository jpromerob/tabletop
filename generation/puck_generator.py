import argparse
import sched
import time
import pdb
import socket
import multiprocessing

# Add necessary imports
import numpy as np
from struct import pack

import sys
sys.path.append('../common')


kernel = np.load("../common/kernel.npy")



# label : sparsity : delta(movement)
speed_dict = {
    'slow': (0.04, 0.005),
    'medium': (0.24, 0.024),
    'fast': (0.40, 0.120)
}

P_SHIFT = 15
Y_SHIFT = 0
X_SHIFT = 16
NO_TIMESTAMP = 0x80000000



global sock 
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Define your function here
def ev_generation_process(shared_data):

    time.sleep(0.100)
    indices = shared_data['indices']
    ip = shared_data['ip']
    port = shared_data['port']


    global sock
    while(True):
        data = b""
        offset_x = shared_data['cx'] - int(shared_data['k_sz']/2)
        offset_y = shared_data['cy'] - int(shared_data['k_sz']/2)
        # print(f"Offsets: {offset_x},{offset_y}")
        max_nb_evs = len(indices)
        sparsity = speed_dict[shared_data['speed']][0]
        act_nb_events = int(sparsity*max_nb_evs)
        for i in np.random.choice(np.arange(max_nb_evs), size=act_nb_events, replace=False):      
            x = indices[i][1] + offset_x 
            y = indices[i][0] + offset_y 
            polarity = 1
            packed = (NO_TIMESTAMP + (polarity << P_SHIFT) + (y << Y_SHIFT) + (x << X_SHIFT))
            data += pack("<I", packed)
        sock.sendto(data, (ip, port))
        sock.sendto(data, ("172.16.222.30", 3330))
        time.sleep(0.001)

def trajectory_process(shared_data):

    min_x = shared_data['k_sz']/2
    min_y = shared_data['k_sz']/2
    max_x = shared_data['width']-shared_data['k_sz']/2
    max_y = shared_data['height']-shared_data['k_sz']/2
    cx = int(shared_data['width']/2)
    cy = int(shared_data['height']/4)

    if shared_data['mode'] == 'line_x':
        go_right = True
    if shared_data['mode'] == 'line_y':
        go_down = True

    while(True):

        delta = speed_dict[shared_data['speed']][1]
        if shared_data['mode'] == 'line_x':
            if go_right:
                if cx + delta <= max_x:
                    cx += delta
                else:
                    go_right = False
                    cx -= delta
            if not go_right:
                if cx - delta >= min_x:
                    cx -= delta
                else:
                    go_right = True
                    cx += delta

        if shared_data['mode'] == 'line_y':
            if go_down:
                if cy + delta <= max_y:
                    cy += delta
                else:
                    go_down = False
                    cy -= delta
            if not go_down:
                if cy - delta >= min_y:
                    cy -= delta
                else:
                    go_down = True
                    cy += delta

        else:
            pass

        shared_data['cx'] = int(cx) #int(shared_data['k_sz']/2) # Needs to be higher than k_len/2
        shared_data['cy'] = int(cy) #int(shared_data['k_sz']/2) # Needs to be higher than k_len/2
        time.sleep(0.000100)

    


def parse_args():

    parser = argparse.ArgumentParser(description="Script to call forward_data function every 100 microseconds.")
    parser.add_argument('-i', '--ip', type=str, help="Destination IP address", default="172.16.222.30")
    parser.add_argument('-p', '--port', type=int, help="Destination port number (default: 8080)", default=3331)
    parser.add_argument('-x', '--width', type=int, help="Size X axis", default=256)
    parser.add_argument('-y', '--height', type=int, help="Size Y axis", default=8)
    parser.add_argument('-s', '--speed', type=str, help="Speed", default="slow")
    parser.add_argument('-m', '--mode', type=str, help="Speed", default="line_y")



    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()

    shared_data = multiprocessing.Manager().dict()

    shared_data['cx'] = 0
    shared_data['cy'] = 0
    shared_data['ip'] = args.ip
    shared_data['port'] = args.port
    shared_data['width'] = args.width
    shared_data['height'] = args.height
    shared_data['speed'] = args.speed
    shared_data['mode'] = args.mode

    shared_data['kernel'] = np.load("../common/kernel.npy")
    shared_data['indices'] = np.argwhere(shared_data['kernel']>0)
    shared_data['k_sz'] = len(shared_data['kernel'])



    ev_gen_proc = multiprocessing.Process(target=ev_generation_process, args=(shared_data,))
    ev_gen_proc.start()


    traj_proc = multiprocessing.Process(target=trajectory_process, args=(shared_data,))
    traj_proc.start()



    ev_gen_proc.join()
    traj_proc.join()    
