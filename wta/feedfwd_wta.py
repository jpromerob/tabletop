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
import csv
import select

W_DIRECT = 5
W_ACCUMULATION = 0.001*W_DIRECT # 
W_INHIBITION = 0.001*W_DIRECT

def save_tuples_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)

def create_2D_to_1D_conn_list(width, height):
    delay = 1 # [ms]
    weight = 2.5 #2*256/((width+height)/2)
    conn_2D_1D_list = []

    for idx in range(width*height):
        pre_idx = idx
        x = int(idx % width)
        y = int(idx / width)
        post_idx = x
        conn_2D_1D_list.append((pre_idx, post_idx, weight, delay))
        post_idx = width + y
        conn_2D_1D_list.append((pre_idx, post_idx, weight, delay))

        # print(f"({x},{y}): idx:{pre_idx} --> {x} and --> {width + y}")
        # if x == 0 or x == 31:
        #     pdb.set_trace()
    save_tuples_to_csv(conn_2D_1D_list, "my_conn_2D_1D_list.csv")
    return conn_2D_1D_list

def create_xy_inh_conn_list(width, height):
    delay = 1 # [ms]
    weight = W_ACCUMULATION
    conn_xy_inh_list = []

    for idx in range(width):
        pre_idx = idx
        post_idx = 0
        conn_xy_inh_list.append((pre_idx, post_idx, weight, delay))
        
    for idx in range(height):
        pre_idx = idx+width
        post_idx = 1
        conn_xy_inh_list.append((pre_idx, post_idx, weight, delay))

    return conn_xy_inh_list

def create_inh_wta_conn_list(width, height):
    delay = 1 # [ms]
    weight = W_INHIBITION
    conn_inh_xy_list = []
    
    for idx in range(width):
        pre_idx = 0
        post_idx = idx       
        conn_inh_xy_list.append((pre_idx, post_idx, weight, delay))

    for idx in range(height):
        pre_idx = 1
        post_idx = idx + width      
        conn_inh_xy_list.append((pre_idx, post_idx, weight, delay))

    return conn_inh_xy_list

def create_xy_wta_conn_list(width, height):
    delay = 1 # [ms]
    weight = W_DIRECT    
    conn_xy_wta_list = []
    for idx in range(width+height):
        pre_idx = idx
        post_idx = idx       
        conn_xy_wta_list.append((pre_idx, post_idx, weight, delay))

    return conn_xy_wta_list

spin_spif_map = {"1": "172.16.223.2", 
                 "37": "172.16.223.106", 
                 "43": "172.16.223.98",
                 "13": "172.16.223.10",
                 "121": "172.16.223.122",
                 "129": "172.16.223.130"}

def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-po', '--port-out', type= int, help="Port Out", default=1988)
    parser.add_argument('-ip', '--ip-out', type= str, help="IP out", default="172.16.222.199")
    parser.add_argument('-b', '--board', type=int, help="Board ID", default=13)
    parser.add_argument('-ww', '--width', type=int, help="Width", default=32)
    parser.add_argument('-wh', '--height', type=int, help="Height", default=32)
    parser.add_argument('-nx', '--npc-x', type=int, help="Width", default=8)
    parser.add_argument('-ny', '--npc-y', type=int, help="Height", default=4)
    parser.add_argument('-rt', '--runtime', type=int, help="Runtime in [s]", default=600)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    print("Setting machines up ... ")
    CFG_FILE = f"spynnaker_{args.board}.cfg"
    SPIF_IP = spin_spif_map[f"{args.board}"]


    print("Configuring Infrastructure ... ")
    SUB_WIDTH = 16
    SUB_HEIGHT = 16
    WIDTH = args.width
    HEIGHT = args.height
    
    NPC_X = args.npc_x
    NPC_Y = args.npc_y

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
            x = int(np_spikes[i]) % (WIDTH+HEIGHT)
            y = int(int(np_spikes[i]) / (WIDTH+HEIGHT))
            # print(f"Sending idx {np_spikes[i]} : ({x},{y})")
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


    if args.board == 1:
        nbr = 24
    elif args.board == 13:
        nbr = 3
    else:
        nbr = 1
    p.setup(timestep=1.0, n_boards_required=nbr, cfg_file=CFG_FILE)


    celltype = p.IF_curr_exp
    p.set_number_of_neurons_per_core(celltype, (NPC_X, NPC_Y))

    # Setting up SPIF Input
    p_spif_in = p.external_devices.SPIFRetinaDevice(pipe=0, width=WIDTH, height=HEIGHT, 
                                                    sub_width=SUB_WIDTH, sub_height=SUB_HEIGHT, 
                                                    chip_coords=CHIP)


    IN_POP_LABEL = "input"
    OUT_POP_LABEL = "output"
    MID_POP_LABEL = "middle"
    WTA_POP_LABEL = "wta"
    GIN_POP_LABEL = "geninhneu"

    # Setting up SPIF Output
    conn_out = p.external_devices.SPIFLiveSpikesConnection([WTA_POP_LABEL], SPIF_IP, SPIF_PORT)
    conn_out.add_receive_callback(WTA_POP_LABEL, recv_nid)
    p_spif_out = p.external_devices.SPIFOutputDevice(database_notify_port_num=conn_out.local_port, chip_coords=CHIP)


    # Defining all the populations involved
    spif_retina = p.Population(WIDTH * HEIGHT, p_spif_in, label=IN_POP_LABEL)   
    middle_pop = p.Population(WIDTH + HEIGHT, celltype(**cell_params), label=MID_POP_LABEL) 
    inhibitory_pop = p.Population(2, celltype(**cell_params), label=GIN_POP_LABEL)   
    wta_pop = p.Population(WIDTH + HEIGHT, celltype(**cell_params), label=WTA_POP_LABEL)
    spif_output = p.Population(None, p_spif_out, label=OUT_POP_LABEL)

    # Connections between 2D SPIF input and middle_pop
    conn_2D_1D_list = create_2D_to_1D_conn_list(WIDTH, HEIGHT)
    cell_conn_2D_1D = p.FromListConnector(conn_2D_1D_list, safe=True)     
    p.Projection(spif_retina, middle_pop, cell_conn_2D_1D, receptor_type='excitatory')

    # Excitatory connections from middle_pop inhibitory_pop
    conn_xy_inh = create_xy_inh_conn_list(WIDTH, HEIGHT)
    cell_conn_xy_inh = p.FromListConnector(conn_xy_inh, safe=True)     
    p.Projection(middle_pop, inhibitory_pop, cell_conn_xy_inh, receptor_type='excitatory')

    # Inhibitory connections from inhibitory_pop to wta_pop
    conn_inh_wta = create_inh_wta_conn_list(WIDTH, HEIGHT)
    cell_conn_inh_wta = p.FromListConnector(conn_inh_wta, safe=True)     
    p.Projection(inhibitory_pop, wta_pop, cell_conn_inh_wta, receptor_type='inhibitory')
    
    # Excitatory connections from middle_pop to wta_pop
    conn_xy_wta = create_xy_wta_conn_list(WIDTH, HEIGHT)
    cell_conn_xy_wta = p.FromListConnector(conn_xy_wta, safe=True)   
    p.Projection(middle_pop, wta_pop, cell_conn_xy_wta, receptor_type='excitatory')

    p.external_devices.activate_live_output_to(wta_pop, spif_output)


    try:
        print("List of parameters:")
        print(f"\tRunning on: 172.16.223.{args.board}")
        print(f"\tSPIF Input: {SPIF_IP} : {3333}")
        print(f"\tSPIF Output: {MY_PC_IP} : {MY_PC_PORT}")
        print(f"\tNPC: {NPC_X} x {NPC_Y}")
        print(f"\tInput {WIDTH} x {HEIGHT} : {WIDTH*HEIGHT}")
        print(f"\tOutput {WIDTH} + {HEIGHT} : {WIDTH+HEIGHT}")

    except KeyboardInterrupt:
        print("\n Simulation cancelled")
        quit()

    print("Waiting for rig-power to end ... ")    
    os.system(f"rig-power 172.16.223.{args.board-1}")
    p.run(RUN_TIME)

    p.end()

