import argparse
import time
import math
import socket
import random
import multiprocessing

# Add necessary imports
import numpy as np
from struct import pack

import sys
sys.path.append('../common')
from tools import Dimensions

kernel = np.load("../common/kernel.npy")


UDP_IP = '172.16.222.30'  
PORT_UDP_DELAYED_DATA = 6262  

# label : sparsity : delta(movement)

P_SHIFT = 15
Y_SHIFT = 0
X_SHIFT = 16
NO_TIMESTAMP = 0x80000000


def parse_args():

    parser = argparse.ArgumentParser(description="Script to produce puck events and wait for answer")
    parser.add_argument('-s', '--sparsity', type=float, help="Sparsity", default=0.2)
    parser.add_argument('-nr', '--nb-reps', type=int, help="Number of repetitions", default=2)


    return parser.parse_args()


if __name__ == "__main__":
    
    # print(f"Slow Circular Motion: Sparsity 0.1 | Delta 0.1")
    # print(f"Medium Circular Motion: Sparsity 0.4 | Delta 0.5")
    # print(f"Fast Circular Motion: Sparsity 0.7 | Delta 0.9")

    args = parse_args()

    shared_data = multiprocessing.Manager().dict()

    shared_data['sparsity'] = args.sparsity
    shared_data['nb_reps'] = args.nb_reps
    shared_data['kernel'] = np.load("../common/kernel.npy")
    shared_data['indices'] = np.argwhere(shared_data['kernel']>0)
    shared_data['k_sz'] = len(shared_data['kernel'])

    dim = Dimensions.load_from_file('../common/homdim.pkl')

    indices = shared_data['indices']
    sparsity = round(shared_data['sparsity'],3)


    
    sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to the receiver's IP and port
    sock_in.bind((UDP_IP, PORT_UDP_DELAYED_DATA))
    # sock_in.settimeout(0.020)


    while(True):

        cx = random.randint(32, 224)
        cy = random.randint(32, 133)

        t_start = time.time()
        for rep in range(shared_data['nb_reps']):
            data = b""
            offset_x = cx - int(shared_data['k_sz']/2)
            offset_y = cy - int(shared_data['k_sz']/2)
            # print(f"Offsets: {offset_x},{offset_y}")
            max_nb_evs = len(indices)
            act_nb_events = int(sparsity*max_nb_evs)
            for i in np.random.choice(np.arange(max_nb_evs), size=act_nb_events, replace=False):      
                x = indices[i][1] + offset_x 
                y = indices[i][0] + offset_y 
                polarity = 1
                packed = (NO_TIMESTAMP + (polarity << P_SHIFT) + (y << Y_SHIFT) + (x << X_SHIFT))
                data += pack("<I", packed)
            # sock.sendto(data, (ip, port))

            sock_out.sendto(data, ("172.16.222.30", 3330))
            sock_out.sendto(data, ("172.16.222.28", 5050)) # for GPU pipeline

            time.sleep(0.0005)


        
        data, sender_address = sock_in.recvfrom(2048)

        # Decode the received data and split it into x and y
        x_norm, y_norm = map(float, data.decode().split(","))
        x_coord = int(x_norm*dim.fl/100)
        y_coord = int(y_norm*dim.fw/100)

        t_end = time.time()-t_start
        print(f"After {round(t_end*1000,1)}[ms]: Got ({x_coord},{y_coord}) 'cause sent ({cx},{cy})")
        time.sleep(0.5)
            




