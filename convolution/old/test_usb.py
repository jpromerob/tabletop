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
import csv




spin_spif_map = {"1": "172.16.223.2", 
                 "37": "172.16.223.106", 
                 "43": "172.16.223.98",
                 "13": "172.16.223.10",
                 "121": "172.16.223.122",
                 "129": "172.16.223.130"}


def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-pf', '--port-f-cnn', type= int, help="Port Out", default=3331)
    parser.add_argument('-ip', '--ip-out', type= str, help="IP out", default="172.16.222.30")
    parser.add_argument('-b', '--board', type=int, help="Board ID", default=37)
    parser.add_argument('-pipe', '--pipe', type=int, help="Pipe ID", default=0)
    parser.add_argument('-x', '--x', type=int, help="x-resolution", default=1280)
    parser.add_argument('-y', '--y', type=int, help="y-resolution", default=720)
    parser.add_argument('-rt', '--runtime', type=int, help="Runtime in [m]", default=240)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    

    print("Setting machines up ... ")
    SPIF_IP_F = spin_spif_map[f"{args.board}"] 
    CHIP_F = (0,0) 


    print("Configuring Infrastructure ... ")
    SUB_WIDTH = 16
    SUB_HEIGHT = 8
    WIDTH = args.x
    HEIGHT = args.y

    PIPE_NB = args.pipe

    
    NPC_X = 16
    NPC_Y = 8

    MY_PC_IP = args.ip_out
    MY_PC_PORT_F_CNN = args.port_f_cnn
    SPIF_PORT = 3332
    POP_LABEL = "target"
    RUN_TIME = 1000*60*args.runtime


    P_SHIFT = 15
    Y_SHIFT = 0
    X_SHIFT = 16
    NO_TIMESTAMP = 0x80000000


    global sock 
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def forward_data(spikes, ip, port):
        global sock
        data = b""
        np_spikes = np.array(spikes)
        for i in range(np_spikes.shape[0]):      
            x = int(np_spikes[i]) % WIDTH
            y = int(int(np_spikes[i]) / WIDTH)
            polarity = 1
            packed = (NO_TIMESTAMP + (polarity << P_SHIFT) + (y << Y_SHIFT) + (x << X_SHIFT))
            data += pack("<I", packed)
        sock.sendto(data, (ip, port))

    def forward_f_cnn_data(label, spikes):
        forward_data(spikes, MY_PC_IP, MY_PC_PORT_F_CNN)


    print("Creating Network ... ")

    f_cell_params = {'tau_m': 2.0,
                     'tau_syn_E': 0.1,
                     'tau_syn_I': 0.1,
                     'v_rest': -65.0,
                     'v_reset': -65.0,
                     'v_thresh': -60.0,
                     'tau_refrac': 0.0, # 0.1 originally
                     'cm': 1,
                     'i_offset': 0.0
                     }


    p.setup(timestep=1.0, n_boards_required=24)


    IN_POP_LABEL_A = "input_a"
    F_CNN_POP_LABEL = "f_cnn"

    celltype = p.IF_curr_exp
    p.set_number_of_neurons_per_core(celltype, (NPC_X, NPC_Y))


    # Setting up SPIF Input
    p_spif_virtual_a = p.Population(WIDTH * HEIGHT, p.external_devices.SPIFRetinaDevice(
                                    pipe=PIPE_NB, width=WIDTH, height=HEIGHT,
                                    sub_width=SUB_WIDTH, sub_height=SUB_HEIGHT, 
                                    chip_coords=CHIP_F), label=IN_POP_LABEL_A)

    # Setting up SPIF Outputs
    spif_f_lsc = p.external_devices.SPIFLiveSpikesConnection([IN_POP_LABEL_A], SPIF_IP_F, SPIF_PORT)
    spif_f_lsc.add_receive_callback(IN_POP_LABEL_A, forward_f_cnn_data)
    spif_f_cnn_output = p.Population(None, p.external_devices.SPIFOutputDevice(
        database_notify_port_num=spif_f_lsc.local_port, chip_coords=CHIP_F), label="f_cnn_output")
    p.external_devices.activate_live_output_to(p_spif_virtual_a, spif_f_cnn_output)

   

    try:
        time.sleep(1)
        print("List of parameters:")
        print(f"\tNPC: {NPC_X} x {NPC_Y}")
        print(f"\tInput {WIDTH} x {HEIGHT}")
    except KeyboardInterrupt:
        print("\n Simulation cancelled")
        quit()

    print(f"Waiting for rig-power (172.16.223.{args.board-1}) to end ... ")    
    os.system(f"rig-power 172.16.223.{args.board-1}")
    p.run(RUN_TIME)

    p.end()

