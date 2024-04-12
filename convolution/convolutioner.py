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

    parser.add_argument('-pf', '--port-f-cnn', type= int, help="Port Out (fast)", default=3331)
    parser.add_argument('-pm', '--port-m-cnn', type= int, help="Port Out (mixed)", default=3334)
    parser.add_argument('-ps', '--port-s-cnn', type= int, help="Port Out (slow)", default=3337)
    parser.add_argument('-ip', '--ip-pc', type= str, help="IP PC", default="172.16.222.30")
    parser.add_argument('-ns', '--next-module-ip', type= str, help="IP SPIF", default="172.16.223.10")
    parser.add_argument('-np', '--next-module_port', type= int, help="IP SPIF", default=3333)

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
    
    print(f"{args.port_f_cnn}")
    print(f"{args.port_m_cnn}")
    print(f"{args.port_s_cnn}")
    print(f"{args.ip_pc}")

    dim = Dimensions.load_from_file('../common/homdim.pkl')
    print("Setting machines up ... ")
    
    CFG_FILE = f"spynnaker_{args.board}.cfg"
    SPIF_IP_F = "172.16.223.2"
    SPIF_IP_M = "172.16.223.130" # SPIF-16 CHIP (16,8)
    SPIF_IP_S = "172.16.223.122" # SPIF-15 CHIP (32,16)
    SPIF_IP_I = SPIF_IP_S



    print("Generating Kernels ... \n")
    f_kernel = make_whole_kernel("fast", args.ip_pc, args.ks, dim.hs, args.w_scaler, args.thickness, 2.00*0.9)
    m_kernel = make_whole_kernel("medium", args.ip_pc, args.ks, dim.hs, args.w_scaler, args.thickness, 1.10*0.9)
    s_kernel = make_whole_kernel("slow", args.ip_pc, args.ks, dim.hs, args.w_scaler, args.thickness, 0.75*0.9)



    k_sz = len(f_kernel)

    print("Configuring Infrastructure ... ")
    SUB_WIDTH = 16
    SUB_HEIGHT = 8
    WIDTH = dim.fl
    HEIGHT = dim.fw
    OUT_WIDTH = WIDTH-len(f_kernel)+1
    OUT_HEIGHT = HEIGHT-len(f_kernel)+1


    pow_2 = [1,2,3,4,5]
    nb_cores = 24 * 48 * 16
    for i in pow_2:
        y = 2**i
        x = y
        if (x*y >= 4*(WIDTH-args.ks)*(HEIGHT-args.ks)/nb_cores):
            break
        x = 2*y
        if (x*y >= 4*(WIDTH-args.ks+1)*(HEIGHT-args.ks+1)/nb_cores):
            break
    
    NPC_X = x
    NPC_Y = y

    used_nb_neurons = 3*((WIDTH-len(f_kernel)+1)*(HEIGHT-len(f_kernel)+1))
    used_nb_cores = int(used_nb_neurons/(NPC_X*NPC_Y))
    percentage_use = round(100*used_nb_cores/nb_cores,2)

    print(f"Using {used_nb_neurons} neurons i.e. {used_nb_cores}/{nb_cores} cores ({percentage_use} % of the system)")
    time.sleep(2)
    # pdb.set_trace()

    MY_PC_IP = args.ip_pc
    MY_PC_PORT_F_CNN = args.port_f_cnn
    MY_PC_PORT_S_CNN = args.port_s_cnn
    MY_PC_PORT_M_CNN = args.port_m_cnn
    NEXT_MODULE_IP = args.next_module_ip
    NEXT_MODULE_PORT = args.next_module_port
    SPIF_PORT = 3332
    POP_LABEL = "target"
    RUN_TIME = 1000*60*args.runtime
    CHIP_F = (0, 0)
    CHIP_M = (16, 8)
    CHIP_S = (32, 16)
    CHIP_I = CHIP_S # input through SPIF-15 (172.16.223.122)


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
            x = int(int(np_spikes[i]) % OUT_WIDTH)+int(k_sz/2)
            y = int(int(np_spikes[i]) / OUT_WIDTH)+int(k_sz/2)
            polarity = 1
            packed = (NO_TIMESTAMP + (polarity << P_SHIFT) + (y << Y_SHIFT) + (x << X_SHIFT))
            data += pack("<I", packed)
        sock.sendto(data, (ip, port))
        sock.sendto(data, (NEXT_MODULE_IP, NEXT_MODULE_PORT))

    def forward_f_cnn_data(label, spikes):
        forward_data(spikes, MY_PC_IP, MY_PC_PORT_F_CNN)

    def forward_m_cnn_data(label, spikes):
        forward_data(spikes, MY_PC_IP, MY_PC_PORT_M_CNN)

    def forward_s_cnn_data(label, spikes):
        forward_data(spikes, MY_PC_IP, MY_PC_PORT_S_CNN)

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
    f_cell_params = {'tau_m': 1, **common_neuron_params}
    m_cell_params = {'tau_m': 8, **common_neuron_params}
    s_cell_params = {'tau_m': 64, **common_neuron_params}
    a_cell_params = f_cell_params
    x_cell_params  = f_cell_params



    p.setup(timestep=1.0, n_boards_required=24, cfg_file=CFG_FILE)


    IN_POP_LABEL = "input_a"
    F_CNN_POP_LABEL = "f_cnn"
    M_CNN_POP_LABEL = "m_cnn"
    S_CNN_POP_LABEL = "s_cnn"

    celltype = p.IF_curr_exp
    p.set_number_of_neurons_per_core(celltype, (NPC_X, NPC_Y))



    # Setting up SPIF Input
    p_spif_virtual_a = p.Population(WIDTH * HEIGHT, p.external_devices.SPIFRetinaDevice(
                                    pipe=0, width=WIDTH, height=HEIGHT,
                                    sub_width=SUB_WIDTH, sub_height=SUB_HEIGHT, 
                                    chip_coords=CHIP_I), label=IN_POP_LABEL)

    # Setting up Fast (high-speed) Convolutional Layer
    f_cnn_conn = p.ConvolutionConnector(kernel_weights=f_kernel)
    f_cnn_pop = p.Population(OUT_WIDTH * OUT_HEIGHT, celltype(**f_cell_params),
                            structure=p.Grid2D(OUT_WIDTH / OUT_HEIGHT), label=F_CNN_POP_LABEL)

    # Setting up Mid (medium-speed) Convolutional Layer
    m_cnn_conn = p.ConvolutionConnector(kernel_weights=m_kernel)
    m_cnn_pop = p.Population(OUT_WIDTH * OUT_HEIGHT, celltype(**m_cell_params),
                            structure=p.Grid2D(OUT_WIDTH / OUT_HEIGHT), label=M_CNN_POP_LABEL)

    # Setting up Slow (low-speed) Convolutional Layer
    s_cnn_conn = p.ConvolutionConnector(kernel_weights=s_kernel)
    s_cnn_pop = p.Population(OUT_WIDTH * OUT_HEIGHT, celltype(**s_cell_params),
                            structure=p.Grid2D(OUT_WIDTH / OUT_HEIGHT), label=S_CNN_POP_LABEL)


    # Projection from SPIF virtual to CNN populations
    p.Projection(p_spif_virtual_a, f_cnn_pop, f_cnn_conn, p.Convolution())
    p.Projection(p_spif_virtual_a, m_cnn_pop, m_cnn_conn, p.Convolution())
    p.Projection(p_spif_virtual_a, s_cnn_pop, s_cnn_conn, p.Convolution())


    # Setting up SPIF Outputs (lsc: live-spikes-connection)
    spif_f_lsc = p.external_devices.SPIFLiveSpikesConnection([F_CNN_POP_LABEL], SPIF_IP_F, SPIF_PORT)
    spif_f_lsc.add_receive_callback(F_CNN_POP_LABEL, forward_f_cnn_data)
    spif_f_cnn_output = p.Population(None, p.external_devices.SPIFOutputDevice(
        database_notify_port_num=spif_f_lsc.local_port, chip_coords=CHIP_F), label="f_cnn_output")
    p.external_devices.activate_live_output_to(f_cnn_pop, spif_f_cnn_output)

    spif_m_lsc = p.external_devices.SPIFLiveSpikesConnection([M_CNN_POP_LABEL], SPIF_IP_M, SPIF_PORT)
    spif_m_lsc.add_receive_callback(M_CNN_POP_LABEL, forward_m_cnn_data)
    spif_m_cnn_output = p.Population(None, p.external_devices.SPIFOutputDevice(
        database_notify_port_num=spif_m_lsc.local_port, chip_coords=CHIP_M), label="m_cnn_output")
    p.external_devices.activate_live_output_to(m_cnn_pop, spif_m_cnn_output)
   
    spif_s_lsc = p.external_devices.SPIFLiveSpikesConnection([S_CNN_POP_LABEL], SPIF_IP_S, SPIF_PORT)
    spif_s_lsc.add_receive_callback(S_CNN_POP_LABEL, forward_s_cnn_data)
    spif_s_cnn_output = p.Population(None, p.external_devices.SPIFOutputDevice(
        database_notify_port_num=spif_s_lsc.local_port, chip_coords=CHIP_S), label="s_cnn_output")
    p.external_devices.activate_live_output_to(s_cnn_pop, spif_s_cnn_output)


    try:
        time.sleep(1)
        print("List of parameters:")
        print(f"Board 172.16.223.{args.board} ({CFG_FILE})")
        print(f"SPIF @ {SPIF_IP_I}")
        print(f"\tNPC: {NPC_X} x {NPC_Y}")
        print(f"\tOutput {OUT_WIDTH} x {OUT_HEIGHT}")
        print(f"\tKernel Size: {len(f_kernel)}")
        print(f"\tKernel Sum: {abs(round(np.sum(f_kernel),3))}")
        print(f"\tWeight Scaler: {args.w_scaler}")
        print(f"\tTau_m: {args.tau_m}")
        print(f"\tSending to {NEXT_MODULE_IP} through {NEXT_MODULE_PORT}")


        user_input = input_with_timeout("Happy?\n ", 10)
    except KeyboardInterrupt:
        print("\n Simulation cancelled")
        quit()


    # pdb.set_trace()

    print(f"Waiting for rig-power (172.16.223.{args.board-1}) to end ... ")    
    os.system(f"rig-power 172.16.223.{args.board-1}")
    
    p.run(RUN_TIME)

    p.end()

