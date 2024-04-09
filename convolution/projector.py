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

    parser.add_argument('-pf', '--port-out', type= int, help="Port Out (fast)", default=6565)
    parser.add_argument('-ip', '--ip-pc', type= str, help="IP PC", default="172.16.222.28")
    parser.add_argument('-ns', '--next-module-ip', type= str, help="IP SPIF", default="172.16.223.10")
    parser.add_argument('-np', '--next-module_port', type= int, help="IP SPIF", default=3333)
    parser.add_argument('-b', '--board', type=int, help="Board ID", default=13)
    parser.add_argument('-rt', '--runtime', type=int, help="Runtime in [m]", default=240)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    
    print(f"{args.port_out}")
    print(f"{args.ip_pc}")

    print("Setting machines up ... ")
    CFG_FILE = f"spynnaker_{args.board}.cfg"
    SPIF_IP = spin_spif_map[f"{args.board}"]



    print("Configuring Infrastructure ... ")
    SUB_WIDTH = 16
    SUB_HEIGHT = 8


    WIDTH = smallest_power_of_2(dim.fl) 
    HEIGHT = smallest_power_of_2(dim.fw) 
    
    NPC_X = 16 #16
    NPC_Y = 8 #8

    MY_PC_IP = args.ip_pc
    MY_PC_PORT_OUT = args.port_out
    NEXT_SPIF_PORT = 3333
    SPIF_PORT = 3332
    RUN_TIME = 1000*60*args.runtime
    CHIP = (0,0) 


    P_SHIFT = 15
    Y_SHIFT = 0
    X_SHIFT = 16
    NO_TIMESTAMP = 0x80000000



    def create_list():

        conn_list = []
        weight = 200
        delay = 0


        # Connectivity calculation
        p_gap = 0.20

        mirror = int(WIDTH / 2)
        gap = int(WIDTH * p_gap)
        middle_left = mirror - gap
        base_left = mirror - 2 * gap
        middle_right = mirror + gap
        base_right = mirror + 2 * gap

        middle_height = int(dim.fw / 2)

        for pre_y in range(HEIGHT):
            for pre_x in range(WIDTH):

                #far pitch:
                if pre_x <= mirror:
                    post_x = WIDTH - 1 - int(pre_x * base_left / mirror)
                    post_y = int((pre_y - middle_height) * pre_x / mirror + middle_height)
                elif pre_x < middle_right:
                    post_x = WIDTH - 1 - (pre_x - mirror + base_left)
                    post_y = pre_y
                else:
                    post_x = WIDTH-2
                    post_y = int(middle_height)
                    
                pre_idx = pre_y*WIDTH+pre_x
                post_idx = post_y*WIDTH+post_x
                conn_list.append((pre_idx, post_idx, weight, delay))
                # print(f"({pre_x},{pre_y}) --> {post_idx}")

        return conn_list

    global sock 
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def forward_data(spikes, ip, port):
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
        sock.sendto(data, (ip, port))
        sock.sendto(data, ("172.16.222.30", 5050))

    def forward_out_data(label, spikes):
        forward_data(spikes, MY_PC_IP, MY_PC_PORT_OUT)

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
    OUT_POP_LABEL = "output"

    celltype = p.IF_curr_exp
    p.set_number_of_neurons_per_core(celltype, (NPC_X, NPC_Y))



    # Setting up SPIF Input
    p_spif_virtual_a = p.Population(WIDTH * HEIGHT, p.external_devices.SPIFRetinaDevice(
                                    pipe=0, width=WIDTH, height=HEIGHT,
                                    sub_width=SUB_WIDTH, sub_height=SUB_HEIGHT, 
                                    chip_coords=CHIP), label=IN_POP_LABEL)



    
    middle_pop = p.Population(WIDTH * HEIGHT, celltype(**cell_params),
                            structure=p.Grid2D(WIDTH / HEIGHT), label=OUT_POP_LABEL)

    # pdb.set_trace()


    conn_list = create_list()
    cell_conn = p.FromListConnector(conn_list, safe=True)      
    con_move = p.Projection(p_spif_virtual_a, middle_pop, cell_conn)
 

    # Setting up SPIF Outputs (lsc: live-spikes-connection)
    spif_lsc = p.external_devices.SPIFLiveSpikesConnection([OUT_POP_LABEL], SPIF_IP, SPIF_PORT)
    spif_lsc.add_receive_callback(OUT_POP_LABEL, forward_out_data)
    spif_out_output = p.Population(None, p.external_devices.SPIFOutputDevice(
        database_notify_port_num=spif_lsc.local_port, chip_coords=CHIP), label="spif_output")
    p.external_devices.activate_live_output_to(middle_pop, spif_out_output)


    try:
        time.sleep(1)
        print("List of parameters:")
        print(f"Board 172.16.223.{args.board} ({CFG_FILE})")
        print(f"SPIF @ {SPIF_IP}")
        print(f"Sending data to {MY_PC_IP} ({MY_PC_PORT_OUT})")
        print(f"\tInput: {WIDTH} x {HEIGHT}")
        print(f"\tNPC: {NPC_X} x {NPC_Y}")
        user_input = input_with_timeout("Happy?\n ", 10)
    except KeyboardInterrupt:
        print("\n Simulation cancelled")
        quit()


    # pdb.set_trace()

    print(f"Waiting for rig-power (172.16.223.{args.board-1}) to end ... ")    
    os.system(f"rig-power 172.16.223.{args.board-1}")
    
    p.run(RUN_TIME)

    p.end()

