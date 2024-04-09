import numpy as np
import pyNN.spiNNaker as p
import pdb
import os
import socket
from struct import pack
import socket
import argparse
import time

import sys
sys.path.append('../../common')
sys.path.append('../')
from tools import Dimensions
dim = Dimensions.load_from_file('../../common/homdim.pkl')
from utils import *



spin_spif_map = {"1": "172.16.223.2",       # rack 1   | spif-00
                 "37": "172.16.223.106",    # b-ip 37  | spif-13
                 "43": "172.16.223.98",     # b-ip 43  | spif-12
                 "13": "172.16.223.10",     # 3-b mach | spif-01
                 "121": "172.16.223.122",   # rack 3   | spif-15
                 "129": "172.16.223.130"}   # rack 2   | spif-16

def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-pf', '--port-out', type= int, help="Port Out (fast)", default=3331)
    parser.add_argument('-ip', '--ip-pc', type= str, help="IP PC", default="172.16.222.30")
    parser.add_argument('-b', '--board', type=int, help="Board ID", default=1)
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
    WIDTH = 32 #dim.fl
    HEIGHT = 32 #dim.fw

    
    NPC_X = 16
    NPC_Y = 8

    MY_PC_IP = args.ip_pc
    MY_PC_PORT_OUT = args.port_out
    NEXT_SPIF_PORT = 3333
    SPIF_PORT = 3332
    RUN_TIME = 1000*60*args.runtime
    CHIP = (0, 0) # since SPIF_IP = "172.16.223.122"


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
            x = int(np_spikes[i] % WIDTH)
            y = int(np_spikes[i] / WIDTH)
            polarity = 1
            packed = (NO_TIMESTAMP + (polarity << P_SHIFT) + (y << Y_SHIFT) + (x << X_SHIFT))
            data += pack("<I", packed)
        sock.sendto(data, (ip, port))
        # sock.sendto(data, (NEXT_SPIF_IP, NEXT_SPIF_PORT))

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



    p.setup(timestep=1.0, n_boards_required=1, cfg_file=CFG_FILE)


    IN_POP_LABEL = "input"
    OUT_POP_LABEL = "output"

    celltype = p.IF_curr_exp
    p.set_number_of_neurons_per_core(celltype, (NPC_X, NPC_Y))



    # Setting up SPIF Input
    p_spif_virtual_a = p.Population(WIDTH * HEIGHT, p.external_devices.SPIFRetinaDevice(
                                    pipe=0, width=WIDTH, height=HEIGHT,
                                    sub_width=SUB_WIDTH, sub_height=SUB_HEIGHT, 
                                    chip_coords=CHIP), label=IN_POP_LABEL)



    k_sz = 1
    k_weight = 10
    kernel = np.ones((k_sz, k_sz))*k_weight
    convolution = p.ConvolutionConnector(kernel_weights=kernel)
    out_width, out_height = convolution.get_post_shape((WIDTH, HEIGHT))
    out_width = int(out_width)
    out_height = int(out_height)
    print(f"{out_width} x {out_height}")
    middle_pop = p.Population(out_width * out_height, p.IF_curr_exp(),
                                    structure=p.Grid2D(out_width / out_height), label=OUT_POP_LABEL)
    
    p.Projection(p_spif_virtual_a, middle_pop, convolution, p.Convolution())


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

