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

def parse_cfg():
    # Initialize my_flags array
    my_flags = [False] * 10

    # Open the settings.cfg file for reading
    with open('settings.cfg', 'r') as file:

        index = 0
        # Read each line in the file
        for line in file:
            # Split the line into parts based on ':'
            parts = line.strip().split(':')
            
            # Get the index and values after ':'
            values = parts[1].split(',')

            if index < 3:
                # Update the my_flags array based on the values
                for i, value in enumerate(values):
                    if 'f' in value:
                        my_flags[index*3] = True
                    if 'm' in value:
                        my_flags[index*3 + 1] = True
                    if 's' in value:
                        my_flags[index*3 + 2] = True
            else:
                for i, value in enumerate(values):
                    if '1' in value:
                        my_flags[-1] = True
            index+=1

    return my_flags

def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-b', '--board', type=int, help="Board ID", default=1)
    parser.add_argument('-t', '--runtime', type=int, help="Runtime in [m]", default=240)
    parser.add_argument('-r', '--resolution', type=int, help="Max X,Y resolution", default=64)

    parser.add_argument('-ip', '--ip-out', type= str, help="IP out", default="172.16.222.30")
    parser.add_argument('-pf', '--port-f-cnn', type= int, help="Port Out (fast)", default=3331)
    parser.add_argument('-pm', '--port-m-cnn', type= int, help="Port Out (medium)", default=3334)
    parser.add_argument('-ps', '--port-s-cnn', type= int, help="Port Out (slow)", default=3337)

    parser.add_argument('-ks', '--ks', type=int, help="Kernel Size", default=45)
    parser.add_argument('-ws', '--w-scaler', type=float, help="Weight Scaler", default=0.4) 
    parser.add_argument('-th', '--thickness', type=int, help="Kernel edge thickness", default=2)
    
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    flags = parse_cfg()
    

    dim = Dimensions.load_from_file('../common/homdim.pkl')
    print("Setting machines up ... ")
    CFG_FILE = f"spynnaker_{args.board}.cfg"
    SPIF_IP_F = "172.16.223.2"
    SPIF_IP_M = "172.16.223.130" # SPIF-16 CHIP (16,8)
    SPIF_IP_S = "172.16.223.122" # SPIF-15 CHIP (32,16)


    print("Generating Kernels ... \n")
    f_kernel = make_whole_kernel("fast", args.ip_out, args.ks, dim.hs, args.w_scaler, args.thickness, 2.00*0.9)
    m_kernel = make_whole_kernel("medium", args.ip_out, args.ks, dim.hs, args.w_scaler, args.thickness, 1.10*0.9)
    s_kernel = make_whole_kernel("slow", args.ip_out, args.ks, dim.hs, args.w_scaler, args.thickness, 0.75*0.9)

    # pdb.set_trace()

    print("Configuring Infrastructure ... ")
    SUB_WIDTH = 16
    SUB_HEIGHT = 8
    WIDTH = min(args.resolution, 2**math.ceil(math.log(dim.fl,2)))
    HEIGHT = min(args.resolution,dim.fw)
    OUT_WIDTH = WIDTH-len(f_kernel)+1
    OUT_HEIGHT = HEIGHT-len(f_kernel)+1


    pow_2 = [1,2,3,4,5]
    nb_cores = 24 * 48 * 16
    for i in pow_2:
        y = 2**i
        x = y
        if (x*y >= 4*(WIDTH-args.ks+1)*(HEIGHT-args.ks+1)/nb_cores):
            break
        x = 2*y
        if (x*y >= 4*(WIDTH-args.ks+1)*(HEIGHT-args.ks+1)/nb_cores):
            break
    
    NPC_X = 8
    NPC_Y = 4

    MY_PC_IP = args.ip_out
    MY_PC_PORT_F_CNN = args.port_f_cnn
    MY_PC_PORT_S_CNN = args.port_s_cnn
    MY_PC_PORT_M_CNN = args.port_m_cnn
    SPIF_PORT = 3332
    POP_LABEL = "target"
    RUN_TIME = 1000*60*args.runtime
    CHIP_F = (0, 0)
    CHIP_M = (16, 8)
    CHIP_S = (32, 16)


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

    def forward_f_cnn_data(label, spikes):
        forward_data(spikes, MY_PC_IP, MY_PC_PORT_F_CNN)

    def forward_s_cnn_data(label, spikes):
        forward_data(spikes, MY_PC_IP, MY_PC_PORT_S_CNN)

    def forward_m_cnn_data(label, spikes):
        forward_data(spikes, MY_PC_IP, MY_PC_PORT_M_CNN)


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
    m_cell_params = {'tau_m': 11, **common_neuron_params}
    s_cell_params = {'tau_m': 64, **common_neuron_params}
    a_cell_params = f_cell_params
    x_cell_params  = f_cell_params



    p.setup(timestep=1.0, n_boards_required=24)


    celltype = p.IF_curr_exp
    p.set_number_of_neurons_per_core(celltype, (NPC_X, NPC_Y))


    IN_POP_LABEL = "input"

    # Setting up SPIF Input
    p_spif_virtual_a = p.Population(WIDTH * HEIGHT, p.external_devices.SPIFRetinaDevice(
                                    pipe=0, width=WIDTH, height=HEIGHT,
                                    sub_width=SUB_WIDTH, sub_height=SUB_HEIGHT, 
                                    chip_coords=CHIP_S), label=IN_POP_LABEL)

    F_CNN_POP_LABEL = "f_cnn"
    M_CNN_POP_LABEL = "m_cnn"
    S_CNN_POP_LABEL = "s_cnn"

    
    mux_syn = p.StaticSynapse(weight=100, delay=0) # weights from SCNNs to copies

    # Inhibiting mux-ed populations using activity neurons
    act_scaler=0.5
    exc_syn = p.StaticSynapse(weight=30*act_scaler, delay=0) # Exciting activity neuron
    inh_syn = p.StaticSynapse(weight=50*act_scaler, delay=0) 


    # Give name to all flags
    CONNECT_FAST_SCNN = flags[0]
    CONNECT_MEDIUM_SCNN = flags[1]
    CONNECT_SLOW_SCNN = flags[2]
    CONNECT_FAST_COPY = flags[3]
    CONNECT_MEDIUM_COPY = flags[4]
    CONNECT_SLOW_COPY = flags[5]
    CONNECT_FAST_INHIB = flags[6]
    CONNECT_MEDIUM_INHIB = flags[7]
    CONNECT_SLOW_INHIB = flags[8]
    MERGE_THREE_CHANNELS = flags[9]


    arbo_one_two = [[],[],[]]
    arbo_three_two = ["","",""]

    if CONNECT_FAST_SCNN:

        arbo_one_two[0].append("f_cnn")
        arbo_three_two[0] = "f_cnn"

        print("Creating SCNN for High Speed Motion")

        # Setting up Fast (high-speed) Convolutional Layer
        f_cnn_conn = p.ConvolutionConnector(kernel_weights=f_kernel)
        f_cnn_pop = p.Population(OUT_WIDTH * OUT_HEIGHT, celltype(**f_cell_params),
                                structure=p.Grid2D(OUT_WIDTH / OUT_HEIGHT), label=F_CNN_POP_LABEL)
        
        # Projection from SPIF virtual to CNN populations
        p.Projection(p_spif_virtual_a, f_cnn_pop, f_cnn_conn, p.Convolution())


    if CONNECT_MEDIUM_SCNN:

        arbo_one_two[1].append("m_cnn")
        arbo_three_two[1] = "m_cnn"

        print("Creating SCNN for Medium Speed Motion")
        
        # Setting up Mid (medium-speed) Convolutional Layer
        m_cnn_conn = p.ConvolutionConnector(kernel_weights=m_kernel)
        m_cnn_pop = p.Population(OUT_WIDTH * OUT_HEIGHT, celltype(**m_cell_params),
                                structure=p.Grid2D(OUT_WIDTH / OUT_HEIGHT), label=M_CNN_POP_LABEL)

        # Projection from SPIF virtual to CNN populations
        p.Projection(p_spif_virtual_a, m_cnn_pop, m_cnn_conn, p.Convolution())


    if CONNECT_SLOW_SCNN:

        arbo_one_two[2].append("s_cnn")
        arbo_three_two[2] = "s_cnn"

        print("Creating SCNN for Low Speed Motion")
        # Setting up Slow (low-speed) Convolutional Layer
        s_cnn_conn = p.ConvolutionConnector(kernel_weights=s_kernel)
        s_cnn_pop = p.Population(OUT_WIDTH * OUT_HEIGHT, celltype(**s_cell_params),
                                structure=p.Grid2D(OUT_WIDTH / OUT_HEIGHT), label=S_CNN_POP_LABEL)

        # Projection from SPIF virtual to CNN populations
        p.Projection(p_spif_virtual_a, s_cnn_pop, s_cnn_conn, p.Convolution())


    if CONNECT_FAST_COPY and CONNECT_FAST_SCNN:

        arbo_one_two[0].append(" ++ f_mux")
        arbo_three_two[0] = "f_mux"

        print("Creating Copy (post-SCNN) for High Speed Motion")

        F_MUX_POP_LABEL = "f_mux"

        # # Setting up Mid (medium-speed) Multiplexing Layer
        f_mux_pop = p.Population(OUT_WIDTH * OUT_HEIGHT, celltype(**x_cell_params),
                                structure=p.Grid2D(OUT_WIDTH / OUT_HEIGHT), label=F_MUX_POP_LABEL)

        # Projection from SCNN to mux-ed populations
        p.Projection(f_cnn_pop, f_mux_pop, p.OneToOneConnector(), receptor_type='excitatory', synapse_type=mux_syn)
        


    if CONNECT_MEDIUM_COPY and CONNECT_MEDIUM_SCNN:

        arbo_one_two[1].append(" ++ m_mux")
        arbo_three_two[1] = "m_mux"

        print("Creating Copy (post-SCNN) for Medium Speed Motion")


        M_MUX_POP_LABEL = "m_mux"

        # # Setting up Mid (medium-speed) Multiplexing Layer
        m_mux_pop = p.Population(OUT_WIDTH * OUT_HEIGHT, celltype(**x_cell_params),
                                structure=p.Grid2D(OUT_WIDTH / OUT_HEIGHT), label=M_MUX_POP_LABEL)

        # Projection from SCNN to mux-ed populations
        p.Projection(m_cnn_pop, m_mux_pop, p.OneToOneConnector(), receptor_type='excitatory', synapse_type=mux_syn)
        


    if CONNECT_SLOW_COPY and CONNECT_SLOW_SCNN:

        arbo_one_two[2].append(" ++ s_mux")
        arbo_three_two[2] = "s_mux"

        print("Creating Copy (post-SCNN) for Low Speed Motion")
        S_MUX_POP_LABEL = "s_mux"

        # Setting up Slow (low-speed) Multiplexing Layer
        s_mux_pop = p.Population(OUT_WIDTH * OUT_HEIGHT, celltype(**x_cell_params),
                                structure=p.Grid2D(OUT_WIDTH / OUT_HEIGHT), label=S_MUX_POP_LABEL)

        p.Projection(s_cnn_pop, s_mux_pop, p.OneToOneConnector(), receptor_type='excitatory', synapse_type=mux_syn)


    if CONNECT_FAST_INHIB and CONNECT_MEDIUM_COPY and CONNECT_FAST_SCNN:

        arbo_one_two[0].append(" -- f_act_neuron")

        print("Creating Inhibitor from High Speed Motion")
        f_act_neuron = p.Population(32, celltype(**a_cell_params),
                                structure=p.Grid2D(8 / 4), label="f_act_neuron")

                                
        # Fast SCNN inhibits Medium mux-ed population
        p.Projection(f_cnn_pop, f_act_neuron, p.AllToAllConnector(), receptor_type='excitatory', synapse_type=exc_syn)    
        p.Projection(f_act_neuron, m_mux_pop, p.AllToAllConnector(), receptor_type='inhibitory', synapse_type=inh_syn)
            


    if CONNECT_MEDIUM_INHIB and CONNECT_SLOW_COPY and CONNECT_MEDIUM_SCNN:

        arbo_one_two[1].append(" -- m_act_neuron")

        print("Creating Inhibitor from Medium Speed Motion")

        m_act_neuron = p.Population(32, celltype(**a_cell_params),
                                structure=p.Grid2D(8 / 4), label="m_act_neuron")


        # Medium SCNN inhibits Slow mux-ed population
        p.Projection(m_cnn_pop, m_act_neuron, p.AllToAllConnector(), receptor_type='excitatory', synapse_type=exc_syn)
        p.Projection(m_act_neuron, s_mux_pop, p.AllToAllConnector(), receptor_type='inhibitory', synapse_type=inh_syn)
            

    if CONNECT_SLOW_INHIB:

        arbo_one_two[2].append(" -- s_act_neuron")

        print("Creating Inhibitor from Low Speed Motion")
        

    if MERGE_THREE_CHANNELS:

        print("Merging three channels")
    
        # Merging Stuff
        MERGED_POP_LABEL = "merged"
        
        merged_pop = p.Population(OUT_WIDTH * OUT_HEIGHT, celltype(**x_cell_params),
                                structure=p.Grid2D(OUT_WIDTH / OUT_HEIGHT), label=MERGED_POP_LABEL)

        out_syn = p.StaticSynapse(weight=80, delay=0)

        # Fast SCNN and mux-ed populations converged into one full output layer
        if CONNECT_FAST_COPY:
            p.Projection(f_mux_pop, merged_pop, p.OneToOneConnector(), receptor_type='excitatory', synapse_type=out_syn)
        elif CONNECT_FAST_SCNN: 
            p.Projection(f_cnn_pop, merged_pop, p.OneToOneConnector(), receptor_type='excitatory', synapse_type=out_syn)
        
        if CONNECT_MEDIUM_COPY:
            p.Projection(m_mux_pop, merged_pop, p.OneToOneConnector(), receptor_type='excitatory', synapse_type=out_syn)
        elif CONNECT_MEDIUM_SCNN: 
            p.Projection(m_cnn_pop, merged_pop, p.OneToOneConnector(), receptor_type='excitatory', synapse_type=out_syn)
        
        if CONNECT_SLOW_COPY:
            p.Projection(s_mux_pop, merged_pop, p.OneToOneConnector(), receptor_type='excitatory', synapse_type=out_syn)
        elif CONNECT_SLOW_SCNN: 
            p.Projection(s_cnn_pop, merged_pop, p.OneToOneConnector(), receptor_type='excitatory', synapse_type=out_syn)
    

    print("\n")
    print("NETWORK TOPOLOGY:")
    print("\n")
    print("Input")
    for i in range(len(arbo_one_two)):
        speed_text_list = arbo_one_two[i]
        for j in range(len(speed_text_list)):
            layers_text_list = speed_text_list[j]
            print("  " + layers_text_list)

    print("\n")

    for j in range(len(arbo_three_two)):
        print("  " + arbo_three_two[j])
    print("      Output")

    print("\n\n")



    # Setting up SPIF Outputs (lsc: live-spikes-connection)
    # spif_f_lsc = p.external_devices.SPIFLiveSpikesConnection([F_CNN_POP_LABEL], SPIF_IP_F, SPIF_PORT)
    # spif_f_lsc.add_receive_callback(F_CNN_POP_LABEL, forward_f_cnn_data)
    # spif_f_cnn_output = p.Population(None, p.external_devices.SPIFOutputDevice(
    #     database_notify_port_num=spif_f_lsc.local_port, chip_coords=CHIP_F), label="f_cnn_output")
    # p.external_devices.activate_live_output_to(f_cnn_pop, spif_f_cnn_output)

    # spif_m_lsc = p.external_devices.SPIFLiveSpikesConnection([M_CNN_POP_LABEL], SPIF_IP_M, SPIF_PORT)
    # spif_m_lsc.add_receive_callback(M_CNN_POP_LABEL, forward_m_cnn_data)
    # spif_m_cnn_output = p.Population(None, p.external_devices.SPIFOutputDevice(
    #     database_notify_port_num=spif_m_lsc.local_port, chip_coords=CHIP_M), label="m_cnn_output")
    # p.external_devices.activate_live_output_to(m_cnn_pop, spif_m_cnn_output)
   
    spif_s_lsc = p.external_devices.SPIFLiveSpikesConnection([S_CNN_POP_LABEL], SPIF_IP_S, SPIF_PORT)
    spif_s_lsc.add_receive_callback(S_CNN_POP_LABEL, forward_s_cnn_data)
    spif_s_cnn_output = p.Population(None, p.external_devices.SPIFOutputDevice(
        database_notify_port_num=spif_s_lsc.local_port, chip_coords=CHIP_S), label="s_cnn_output")
    p.external_devices.activate_live_output_to(s_cnn_pop, spif_s_cnn_output)



    # spif_s_lsc = p.external_devices.SPIFLiveSpikesConnection([MERGED_POP_LABEL], SPIF_IP_S, SPIF_PORT)
    # spif_s_lsc.add_receive_callback(MERGED_POP_LABEL, forward_s_cnn_data)
    # spif_merged_output = p.Population(None, p.external_devices.SPIFOutputDevice(
    #     database_notify_port_num=spif_s_lsc.local_port, chip_coords=CHIP_S), label="merged_output")
    # p.external_devices.activate_live_output_to(merged_pop, spif_merged_output)


    try:
        time.sleep(1)
        print("List of parameters:")
        print(f"\tNPC: {NPC_X} x {NPC_Y}")
        print(f"\tInput {WIDTH} x {HEIGHT}")
        print(f"\tOutput {OUT_WIDTH} x {OUT_HEIGHT}")
        print(f"\tKernel Size: {len(f_kernel)}")
        print(f"\tKernel Sum: {abs(round(np.sum(f_kernel),3))}")
        print(f"\tWeight Scaler: {args.w_scaler}")
        user_input = input_with_timeout("Happy?\n ", 10)
    except KeyboardInterrupt:
        print("\n Simulation cancelled")
        quit()


    # pdb.set_trace()

    print(f"Waiting for rig-power (172.16.223.{args.board-1}) to end ... ")    
    os.system(f"rig-power 172.16.223.{args.board-1}")
    
    p.run(RUN_TIME)

    p.end()

