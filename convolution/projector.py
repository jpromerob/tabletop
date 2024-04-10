import numpy as np
import pyNN.spiNNaker as p
import pdb
import os
import socket
from struct import pack
import socket
import argparse
import time
import math

import sys
sys.path.append('../common')
from tools import Dimensions
dim = Dimensions.load_from_file('../common/homdim.pkl')
from utils import *



spin_spif_map = {"1": "172.16.223.2",       # rack 1   | spif-00
                 "37": "172.16.223.106",    # b-ip 37  | spif-13
                 "43": "172.16.223.98",     # b-ip 43  | spif-12
                 "13": "172.16.223.10",     # 3-b mach | spif-01
                 "121": "172.16.223.122",   # rack 3   | spif-15
                 "129": "172.16.223.130"}   # rack 2   | spif-16


def smallest_power_of_2(x):
    # Calculate the smallest power of 2 greater than x
    power = math.ceil(math.log2(x))
    return 2 ** power


def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-cb', '--current-board', type= int, help="Current Computing Board (172.16.223.XX)", default=43)
    parser.add_argument('-nb', '--next-board', type= int, help="Next Computing Board (172.16.223.XX)", default=-1)
    parser.add_argument('-dp', '--display-pc', type= int, help="Display PC (172.16.222.XX)", default=30)
    parser.add_argument('-rt', '--runtime', type=int, help="Runtime in [m]", default=240)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    

    print("Setting machines up ... ")

    os.system(f"cp ~/.spynnaker_{args.current_board}.cfg ~/.spynnaker.cfg")
    SPIF_IP = spin_spif_map[f"{args.current_board}"]



    print("Configuring Infrastructure ... ")
    SUB_WIDTH = 16
    SUB_HEIGHT = 8


    WIDTH = smallest_power_of_2(dim.fl) 
    HEIGHT = smallest_power_of_2(dim.fw) 
    
    NPC_X = 16  #16
    NPC_Y = 8   #8

    IS_THERE_NEXT_STEP = True
    if args.next_board < 0:
        IS_THERE_NEXT_STEP = False
    else:
        NEXT_BOARD_SPIF_IP = spin_spif_map[f'{args.next_board}']
        NEXT_BOARD_SPIF_PORT = 3333
    DISPLAY_PC_IP = f"172.16.222.{args.display_pc}"
    DISPLAY_PC_PORT = args.current_board*100+87
    CURRENT_SPIF_OUT_PORT = 3332
    RUN_TIME = 1000*60*args.runtime
    CHIP = (0,0) 


    P_SHIFT = 15
    Y_SHIFT = 0
    X_SHIFT = 16
    NO_TIMESTAMP = 0x80000000



    def create_list():

        conn_list = []
        weight = 100
        delay = 0

        for pre_y in range(HEIGHT):
            for pre_x in range(WIDTH):
                pre_idx = pre_y*WIDTH+pre_x

                # Projecting Columns (X)
                post_idx = pre_x
                conn_list.append((pre_idx, post_idx, weight, delay))

                # # Projecting Rows (Y)
                post_idx = pre_y+WIDTH
                conn_list.append((pre_idx, post_idx, weight, delay))


        return conn_list

    global sock 
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def forward_out_data(label, spikes):
        global sock
        data = b""
        np_spikes = np.array(spikes)
        for i in range(np_spikes.shape[0]):      
            x = int(np_spikes[i] % WIDTH)
            y = int(np_spikes[i] / WIDTH)
            # print(f"{np_spikes[i]}: ({x},{y})")
            polarity = 1
            packed = (NO_TIMESTAMP + (polarity << P_SHIFT) + (y << Y_SHIFT) + (x << X_SHIFT))
            data += pack("<I", packed)
        if IS_THERE_NEXT_STEP:
            sock.sendto(data, (NEXT_BOARD_SPIF_IP, NEXT_BOARD_SPIF_PORT))
        sock.sendto(data, (DISPLAY_PC_IP, DISPLAY_PC_PORT))


    print("Creating Network ... ")

    # Define common parameters
    cell_params = {
        'tau_m': 1,
        'tau_syn_E': 0.1,
        'tau_syn_I': 0.1,
        'v_rest': -65.0,
        'v_reset': -65.0,
        'v_thresh': -60.0,
        'tau_refrac': 0.0,
        'cm': 1,
        'i_offset': 0.0
    }



    p.setup(timestep=1.0, n_boards_required=1)


    IN_POP_LABEL = "input"
    MID_POP_LABEL = "output"

    celltype = p.IF_curr_exp
    p.set_number_of_neurons_per_core(celltype, (NPC_X, NPC_Y))



    # Setting up SPIF Input
    p_spif_virtual_a = p.Population(WIDTH * HEIGHT, p.external_devices.SPIFRetinaDevice(
                                    pipe=0, width=WIDTH, height=HEIGHT,
                                    sub_width=SUB_WIDTH, sub_height=SUB_HEIGHT, 
                                    chip_coords=CHIP), label=IN_POP_LABEL)



    
    middle_pop = p.Population(WIDTH*NPC_Y, celltype(**cell_params),
                            structure=p.Grid2D(WIDTH / NPC_Y), label=MID_POP_LABEL)

    # pdb.set_trace()


    conn_list = create_list()
    cell_conn = p.FromListConnector(conn_list, safe=True)      
    con_move = p.Projection(p_spif_virtual_a, middle_pop, cell_conn)
 

    # Setting up SPIF Outputs (lsc: live-spikes-connection)
    spif_lsc = p.external_devices.SPIFLiveSpikesConnection([MID_POP_LABEL], SPIF_IP, CURRENT_SPIF_OUT_PORT)
    spif_lsc.add_receive_callback(MID_POP_LABEL, forward_out_data)
    spif_out_output = p.Population(None, p.external_devices.SPIFOutputDevice(
        database_notify_port_num=spif_lsc.local_port, chip_coords=CHIP), label="spif_output")
    p.external_devices.activate_live_output_to(middle_pop, spif_out_output)


    try:
        time.sleep(1)
        print("\n\n\n")
        print("List of parameters:")
        print(f"Computing Mapping @ Board 172.16.223.{args.current_board}")
        print(f"\tWith SPIF @ {SPIF_IP}")
        if IS_THERE_NEXT_STEP:
            print(f"Sending data for further processing to {NEXT_BOARD_SPIF_IP} ({NEXT_BOARD_SPIF_PORT})")
        print(f"Sending data for visualization {DISPLAY_PC_IP} ({DISPLAY_PC_PORT})")
        print(f"\tInput: {WIDTH} x {HEIGHT}")
        print(f"\tNPC: {NPC_X} x {NPC_Y}")
        user_input = input_with_timeout("Happy?\n ", 10)
        print("\n\n\n")
    except KeyboardInterrupt:
        print("\n Simulation cancelled")
        quit()


    # pdb.set_trace()

    print(f"Waiting for rig-power (172.16.223.{args.current_board-1}) to end ... ")    
    os.system(f"rig-power 172.16.223.{args.current_board-1}")
    
    p.run(RUN_TIME)

    p.end()

