import numpy as np
import math
import pyNN.spiNNaker as p
import pdb
import os
import socket
from struct import pack
import matplotlib.pyplot as plt
import paramiko
import socket
import argparse
import math
import time
import sys
import select


spin_spif_map = {"1": "172.16.223.2", 
                 "37": "172.16.223.106", 
                 "43": "172.16.223.98",
                 "13": "172.16.223.10",
                 "121": "172.16.223.122",
                 "129": "172.16.223.130"}

def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-po', '--port-out', type= int, help="Port Out", default=1987)
    parser.add_argument('-ip', '--ip-out', type= str, help="IP out", default="172.16.222.199")
    parser.add_argument('-b', '--board', type=int, help="Board ID", default=13)
    parser.add_argument('-rt', '--runtime', type=int, help="Runtime in [s]", default=600)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    print("Setting machines up ... ")
    CFG_FILE = f"spynnaker_{args.board}.cfg"
    SPIF_IP = spin_spif_map[f"{args.board}"]


    print("Configuring Infrastructure ... ")
    SUB_WIDTH = 16
    SUB_HEIGHT = 8
    WIDTH = 32
    HEIGHT = 32

    pow_2 = [1,2,3,4,5]
    nb_cores = 24 * 48 * 16
    for i in pow_2:
        y = 2**i
        x = y
        if (x*y >= (WIDTH)*(HEIGHT)/nb_cores):
            break
        x = 2*y
        if (x*y >= (WIDTH)*(HEIGHT)/nb_cores):
            break
    
    NPC_X = x
    NPC_Y = y

    MY_PC_IP = args.ip_out
    MY_PC_PORT = args.port_out
    SPIF_PORT = 3332
    POP_LABEL = "target"
    RUN_TIME = 1000*60*args.runtime
    CHIP = (0, 0)


    P_SHIFT = 15
    Y_SHIFT = 0
    X_SHIFT = 16
    NO_TIMESTAMP = 0x80000000


    global sock 
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


    def recv_nid(label, spikes):
        global sock
        data = b""
        np_spikes = np.array(spikes)
        for i in range(np_spikes.shape[0]):    
            # print(i)  
            x = int(np_spikes[i]) % WIDTH
            y = int(int(np_spikes[i]) / WIDTH)
            # print(f"Sending ({x},{y})")
            polarity = 1
            packed = (NO_TIMESTAMP + (polarity << P_SHIFT) + (y << Y_SHIFT) + (x << X_SHIFT))
            data += pack("<I", packed)
        sock.sendto(data, (MY_PC_IP, MY_PC_PORT))


    print("Creating Network ... ")

    cell_params = {'tau_m': 3.0/math.log(2),
                   'tau_syn_E': 1.0,
                   'tau_syn_I': 1.0,
                   'v_rest': -65.0,
                   'v_reset': -65.0,
                   'v_thresh': -60.0,
                   'tau_refrac': 0.0, # 0.1 originally
                   'cm': 1,
                   'i_offset': 0.0
                }


    p.setup(timestep=1.0, n_boards_required=1, cfg_file=CFG_FILE)


    celltype = p.IF_curr_exp
    p.set_number_of_neurons_per_core(celltype, (NPC_X, NPC_Y))

    # Setting up SPIF Input
    p_spif_in = p.external_devices.SPIFRetinaDevice(pipe=0, width=WIDTH, height=HEIGHT, 
                                                    sub_width=SUB_WIDTH, sub_height=SUB_HEIGHT, 
                                                    chip_coords=CHIP)


    IN_POP_LABEL = "input"
    OUT_POP_LABEL = "output"

    # Setting up SPIF Output
    conn_out = p.external_devices.SPIFLiveSpikesConnection([IN_POP_LABEL], SPIF_IP, SPIF_PORT)
    conn_out.add_receive_callback(IN_POP_LABEL, recv_nid)
    p_spif_out = p.external_devices.SPIFOutputDevice(database_notify_port_num=conn_out.local_port, chip_coords=CHIP)


    # Virtual Input Population
    spif_retina = p.Population(WIDTH * HEIGHT, p_spif_in, label=IN_POP_LABEL)
    # target_pop = p.Population(WIDTH + HEIGHT, celltype(**cell_params), label=POP_LABEL)
    # p.Projection(spif_retina, target_pop, convolution, p.Convolution())
    
    spif_output = p.Population(None, p_spif_out, label=OUT_POP_LABEL)
    p.external_devices.activate_live_output_to(spif_retina, spif_output)


    try:
        print("List of parameters:")
        print(f"\tRunning on: 172.16.223.{args.board}")
        print(f"\tSPIF Input: {SPIF_IP} : {3333}")
        print(f"\tSPIF Output: {MY_PC_IP} : {MY_PC_PORT}")
        print(f"\tNPC: {x} x {y}")
        print(f"\tInput {WIDTH} x {HEIGHT} : {WIDTH*HEIGHT}")
        print(f"\tOutput {WIDTH} + {HEIGHT} : {WIDTH+HEIGHT}")
    except KeyboardInterrupt:
        print("\n Simulation cancelled")
        quit()

    print("Waiting for rig-power to end ... ")    
    os.system(f"rig-power 172.16.223.{args.board-1}")
    p.run(RUN_TIME)

    p.end()

