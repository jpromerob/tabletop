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
sys.path.append('../common')
from tools import Dimensions
from utils import *



spin_spif_map = {"1": "172.16.223.2",       # rack 1   | spif-00
                 "37": "172.16.223.106",    # b-ip 37  | spif-13
                 "43": "172.16.223.98",     # b-ip 43  | spif-12
                 "13": "172.16.223.10",     # 3-b mach | spif-01
                 "121": "172.16.223.122",   # rack 3   | spif-15
                 "129": "172.16.223.130"}   # rack 2   | spif-16

def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-pf', '--port-f-cnn', type= int, help="Port Out (fast)", default=3340)
    parser.add_argument('-ip', '--ip-out', type= str, help="IP out", default="172.16.222.30")
    parser.add_argument('-ks', '--ks', type=int, help="Kernel Size", default=45)
    parser.add_argument('-b', '--board', type=int, help="Board ID", default=1)
    parser.add_argument('-ws', '--w-scaler', type=float, help="Weight Scaler", default=0.4) 
    parser.add_argument('-tm', '--tau-m', type=float, help="Tau m", default=3.0) # 
    parser.add_argument('-th', '--thickness', type=int, help="Kernel edge thickness", default=2)
    parser.add_argument('-r', '--ratio', type=float, help="f/s ratio", default=1.0) # 
    parser.add_argument('-rt', '--runtime', type=int, help="Runtime in [m]", default=240)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    

    dim = Dimensions.load_from_file('../common/homdim.pkl')
    print("Setting machines up ... ")
    CFG_FILE = f"spynnaker_{args.board}.cfg"
    SPIF_IP_OUT = spin_spif_map[f"{args.board}"]


    print("Generating Kernels ... \n")
    kernel = make_whole_kernel("fast", args.ip_out, args.ks, dim.hs, args.w_scaler, args.thickness, 2.00*0.9)

    print("Configuring Infrastructure ... ")
    SUB_WIDTH = 16
    SUB_HEIGHT = 8
    WIDTH = dim.fl
    HEIGHT = dim.fw
    OUT_WIDTH = WIDTH-len(kernel)+1
    OUT_HEIGHT = HEIGHT-len(kernel)+1


    NPC_X = 16
    NPC_Y = 16

    MY_PC_IP = args.ip_out
    MY_PC_PORT_OUT = args.port_f_cnn
    SPIF_PORT = 3332
    POP_LABEL = "target"
    RUN_TIME = 1000*60*args.runtime
    CHIP_F = (0, 0)
    CHIP_I = CHIP_F # input through SPIF-15 (172.16.223.122)


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
            x = int(np_spikes[i]) % OUT_WIDTH
            y = int(int(np_spikes[i]) / OUT_WIDTH)
            polarity = 1
            packed = (NO_TIMESTAMP + (polarity << P_SHIFT) + (y << Y_SHIFT) + (x << X_SHIFT))
            data += pack("<I", packed)
        sock.sendto(data, (ip, port))

    def forward_output_data(label, spikes):
        forward_data(spikes, MY_PC_IP, MY_PC_PORT_OUT)



    print("Creating Network ... ")

    # Define common parameters
    common_neuron_params = {
        'tau_syn_E': 0.1,
        'tau_syn_I': 0.1,
        'v_rest': -65.0,
        'v_reset': -65.0,
        'v_thresh': -60.0,
        'tau_refrac': 0.0,
        'cm': 1,
        'i_offset': 0.0
    }

    # Define specific parameters for different cell types
    cell_params = {'tau_m': 1, **common_neuron_params}



    p.setup(timestep=1.0, n_boards_required=1)


    IN_POP_LABEL = "input"
    OUT_POP_LABEL = "output"
    MIDDLE_POP_LABEL = "middle"

    celltype = p.IF_curr_exp
    p.set_number_of_neurons_per_core(celltype, (NPC_X, NPC_Y))



    # Setting up SPIF Input
    p_spif_virtual_a = p.Population(OUT_WIDTH * OUT_HEIGHT, p.external_devices.SPIFRetinaDevice(
                                    pipe=0, width=OUT_WIDTH, height=OUT_HEIGHT,
                                    sub_width=SUB_WIDTH, sub_height=SUB_HEIGHT, 
                                    chip_coords=CHIP_I), label=IN_POP_LABEL)


    k_sz = 1
    k_weight = 100
    kernel = np.ones((k_sz, k_sz))*k_weight
    convolution = p.ConvolutionConnector(kernel_weights=kernel)
    out_width, out_height = convolution.get_post_shape((OUT_WIDTH, OUT_HEIGHT))
    middle_pop = p.Population(out_width * out_height, p.IF_curr_exp(),
                                    structure=p.Grid2D(out_width / out_height), label=MIDDLE_POP_LABEL)
    
    p.Projection(p_spif_virtual_a, middle_pop, convolution, p.Convolution())
   


    spif_f_lsc = p.external_devices.SPIFLiveSpikesConnection([MIDDLE_POP_LABEL], SPIF_IP_OUT, SPIF_PORT)
    spif_f_lsc.add_receive_callback(MIDDLE_POP_LABEL, forward_output_data)
    spif_output = p.Population(None, p.external_devices.SPIFOutputDevice(
        database_notify_port_num=spif_f_lsc.local_port, chip_coords=CHIP_F), label=OUT_POP_LABEL)
    p.external_devices.activate_live_output_to(middle_pop, spif_output)



    # spif_f_lsc = p.external_devices.SPIFLiveSpikesConnection([IN_POP_LABEL], SPIF_IP_OUT, SPIF_PORT)
    # spif_f_lsc.add_receive_callback(IN_POP_LABEL, forward_output_data)
    # spif_output = p.Population(None, p.external_devices.SPIFOutputDevice(
    #     database_notify_port_num=spif_f_lsc.local_port, chip_coords=CHIP_F), label=OUT_POP_LABEL)
    # p.external_devices.activate_live_output_to(p_spif_virtual_a, spif_output)




    try:
        time.sleep(1)
        print("List of parameters:")
        print(f"Board 172.16.223.{args.board} ({CFG_FILE})")
        print(f"SPIF @{SPIF_IP_OUT} --> PC @{MY_PC_IP} on {MY_PC_PORT_OUT}")
        print(f"\tNPC: {NPC_X} x {NPC_Y}")
        print(f"\tOutput {OUT_WIDTH} x {OUT_HEIGHT}")
        print(f"\tKernel Size: {len(kernel)}")
        print(f"\tKernel Sum: {abs(round(np.sum(kernel),3))}")
        print(f"\tWeight Scaler: {args.w_scaler}")
        print(f"\tTau_m: {args.tau_m}")
        user_input = input_with_timeout("Happy?\n ", 10)
    except KeyboardInterrupt:
        print("\n Simulation cancelled")
        quit()



    # pdb.set_trace()

    print(f"Waiting for rig-power (172.16.223.{args.board-1}) to end ... ")    
    os.system(f"rig-power 172.16.223.{args.board-1}")
    
    p.run(RUN_TIME)

    p.end()

