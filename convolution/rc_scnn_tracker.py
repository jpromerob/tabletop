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
sys.path.append('../common')
from tools import Dimensions


def save_tuples_to_csv(data, filename):
    """
    Save a list of tuples to a CSV file.

    Args:
        data (list of tuples): The data to be saved.
        filename (str): The name of the CSV file to create or overwrite.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)

def create_conn_list(width, height):
    delay = 0.1 # [ms]
    weight = 56
    conn_list = []
    for idx in range(width*height):
        pre_idx = idx
        x = int(idx % width)
        y = int(idx / width)

        post_idx = x
        conn_list.append((pre_idx, post_idx, weight, delay))
        post_idx = width + y
        conn_list.append((pre_idx, post_idx, weight, delay))

    save_tuples_to_csv(conn_list, "my_conn_list.csv")
    return conn_list

def create_inner_conn_list(w, h, npc_x, npc_y):
    conn_list = []    
    weight = 56
    delay = 0.1 # 1 [ms]
    nb_col = math.ceil(w/npc_x)
    nb_row = math.ceil(h/npc_y)

    pre_idx = -1
    for h_block in range(nb_row):
        for v_block in range(nb_col):
            for row in range(npc_y):
                for col in range(npc_x):
                    x = v_block*npc_y+col
                    y = h_block*npc_x+row
                    if x<w and y<h:
                        pre_idx += 1                       
                        conn_list.append((pre_idx, x, weight, delay))
                        conn_list.append((pre_idx, w+y, weight, delay))
        
    save_tuples_to_csv(conn_list, "my_conn_list.csv")
    return conn_list

def input_with_timeout(prompt, timeout):
    print(prompt, end='', flush=True)
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    
    if rlist:
        return sys.stdin.readline().strip()
    else:
        return None

def send_kernel():
    # Define the connection parameters
    hostname = '172.16.222.199'  # IP address or hostname of Computer B
    port = 22  # Default SSH port is 22
    username = 'juan'
    password = '@Q9ep427x'  # Replace with your actual password

    # Define the local add remote file paths
    local_file_path = '../common/kernel.npy' 
    remote_file_path = '/home/juan/tabletop/common/kernel.npy'  


    # Create an SSH client
    ssh_client = paramiko.SSHClient()

    # Automatically add the server's host key (this is insecure, see note below)
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Connect to the remote server
    ssh_client.connect(hostname, port, username, password)

    # Use SFTP to copy the file from Computer A to Computer B
    with ssh_client.open_sftp() as sftp:
        sftp.put(local_file_path, remote_file_path)

    # Close the SSH connection
    ssh_client.close()

    print(f"File '{local_file_path}' copied to '{remote_file_path}' on {hostname}")

def make_kernel_circle(r, k_sz, weight, kernel):
    
    var = int((k_sz+1)/2-1)
    a = np.arange(0, 2 * math.pi, 0.01)
    dx = np.round(r * np.sin(a)).astype("uint32")
    dy = np.round(r * np.cos(a)).astype("uint32")
    kernel[var + dx, var + dy] = weight

def make_whole_kernel(k_sz, hs, w_scaler, thickness):

    # Kernel size should be 'odd' (not 'even')
    k_sz = int(k_sz*hs)
    if k_sz%2 == 0:
        k_sz += 1

    # Radius of pattern edges (target)
    pos_radi = np.array([20,9])*hs

    # pos_w is given by the w_scaler
    pos_w = w_scaler

    # Try Kernel WITHOUT negative values 'first'
    kernel = np.zeros((k_sz, k_sz), dtype=np.float32)
    for r in pos_radi:
        for i in np.arange(r-thickness+1, r+1):
            make_kernel_circle(i, k_sz, pos_w, kernel) # 38px

    # Check positive values so as to compensate for them
    total_positive = np.sum(kernel)
    count_positive = np.sum(kernel > 0)

    # neg_w is estimated to compensate for pos_w
    neg_w = -total_positive/((k_sz*k_sz)-count_positive)

    # Try Kernel WITH negative values
    kernel = neg_w*np.ones((k_sz, k_sz), dtype=np.float32)
    for r in pos_radi:
        for i in np.arange(r-thickness+1, r+1):
            make_kernel_circle(i, k_sz, pos_w, kernel) # 38px
            


    plt.imshow(kernel, interpolation='nearest')
    colorbar = plt.colorbar()
    colorbar.set_label('Color Scale')
    plt.savefig(f"kernel.png")


    np.save("../common/kernel.npy", kernel)
    send_kernel()

    print(f"Positive weights: {round(pos_w,6)}")
    print(f"Negative weights: {round(neg_w,6)}")
        
    return kernel


spin_spif_map = {"1": "172.16.223.2", 
                 "37": "172.16.223.106", 
                 "43": "172.16.223.98",
                 "13": "172.16.223.10",
                 "121": "172.16.223.122",
                 "129": "172.16.223.130"}

def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-pc', '--port-cnn', type= int, help="Port Out", default=3331)
    parser.add_argument('-pp', '--port-xyp', type= int, help="Port Out", default=3334)
    parser.add_argument('-ip', '--ip', type= str, help="IP out", default="172.16.222.199")
    parser.add_argument('-ks', '--ks', type=int, help="Kernel Size", default=45)
    parser.add_argument('-b', '--board', type=int, help="Board ID", default=1)
    parser.add_argument('-ws', '--w-scaler', type=float, help="Weight Scaler", default=0.80)
    parser.add_argument('-tm', '--tau-m', type=float, help="Tau m", default=3.0)
    parser.add_argument('-th', '--thickness', type=int, help="Kernel edge thickness", default=2)
    parser.add_argument('-rt', '--runtime', type=int, help="Runtime in [s]", default=240)
    parser.add_argument('-md', '--mode', type=str, help="SPIF Out: cnn or xyp", default="cnn")
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    if args.mode == "cnn":
        showing_cnn = True 
    else:
        showing_cnn = False 
    

    dim = Dimensions.load_from_file('../common/homdim.pkl')
    print("Setting machines up ... ")
    CFG_FILE = f"spynnaker_{args.board}.cfg"
    SPIF_IP_A = "172.16.223.2"
    SPIF_IP_B = "172.16.223.16"


    print("Generating Kernel ... ")
    kernel = make_whole_kernel(args.ks, dim.hs, args.w_scaler, args.thickness)

    print("Configuring Infrastructure ... ")
    SUB_WIDTH = 16
    SUB_HEIGHT = 8
    WIDTH = dim.fl
    HEIGHT = dim.fw
    OUT_WIDTH = WIDTH-len(kernel)+1
    OUT_HEIGHT = HEIGHT-len(kernel)+1


    pow_2 = [1,2,3,4,5]
    nb_cores = 24 * 48 * 16
    for i in pow_2:
        y = 2**i
        x = y
        if (x*y >= (WIDTH-args.ks+1)*(HEIGHT-args.ks+1)/nb_cores):
            break
        x = 2*y
        if (x*y >= (WIDTH-args.ks+1)*(HEIGHT-args.ks+1)/nb_cores):
            break
    
    NPC_X = x*2
    NPC_Y = y*2

    MY_PC_IP = args.ip
    MY_PC_PORT_CNN = args.port_cnn
    MY_PC_PORT_XYP = args.port_xyp
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
            x = int(np_spikes[i]) % OUT_WIDTH
            y = int(int(np_spikes[i]) / OUT_WIDTH)
            polarity = 1
            packed = (NO_TIMESTAMP + (polarity << P_SHIFT) + (y << Y_SHIFT) + (x << X_SHIFT))
            data += pack("<I", packed)
        sock.sendto(data, (MY_PC_IP, MY_PC_PORT_CNN))


    def recv_nid_rc(label, spikes):
        global sock
        data = b""
        np_spikes = np.array(spikes)
        for i in range(np_spikes.shape[0]):    
            x = int(np_spikes[i]) % (OUT_WIDTH+OUT_HEIGHT)
            y = int(int(np_spikes[i]) / (OUT_WIDTH+OUT_HEIGHT))
            # if(x<OUT_WIDTH):
            #     print(f"x:{x}")
            # else:
            #     print(f"\t\ty:{x-OUT_WIDTH}")
            polarity = 1
            packed = (NO_TIMESTAMP + (polarity << P_SHIFT) + (y << Y_SHIFT) + (x << X_SHIFT))
            data += pack("<I", packed)
        sock.sendto(data, (MY_PC_IP, MY_PC_PORT_XYP))

    print("Creating Network ... ")

    cell_params = {'tau_m': args.tau_m/math.log(2),
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


    IN_POP_LABEL = "input"
    CNN_POP_LABEL = "convolution"
    XYP_POP_LABEL = "xyproj"

    celltype = p.IF_curr_exp
    p.set_number_of_neurons_per_core(celltype, (NPC_X, NPC_Y))


    # Setting up SPIF Input
    p_spif_virtual = p.Population(WIDTH * HEIGHT, p.external_devices.SPIFRetinaDevice(
                                  pipe=0, width=WIDTH, height=HEIGHT,
                                  sub_width=SUB_WIDTH, sub_height=SUB_HEIGHT, 
                                  chip_coords=CHIP), label=IN_POP_LABEL)

    # Setting up Convolutional Layer
    convolution = p.ConvolutionConnector(kernel_weights=kernel)
    cnn_pop = p.Population(OUT_WIDTH * OUT_HEIGHT, celltype(**cell_params),
                            structure=p.Grid2D(OUT_WIDTH / OUT_HEIGHT), label=CNN_POP_LABEL)


    # Setting XY Projection Layer
    xyp_pop = p.Population(OUT_WIDTH + OUT_HEIGHT, celltype(**cell_params), label=XYP_POP_LABEL)
    # conn_list = create_conn_list(OUT_WIDTH, OUT_HEIGHT)
    conn_list = create_inner_conn_list(OUT_WIDTH, OUT_HEIGHT, NPC_X, NPC_Y)
    cell_conn = p.FromListConnector(conn_list, safe=True)     

    
    # Projections
    p.Projection(p_spif_virtual, cnn_pop, convolution, p.Convolution())
    p.Projection(cnn_pop, xyp_pop, cell_conn, receptor_type='excitatory')

    # Setting up SPIF Output
    if showing_cnn:
        conn_cnn = p.external_devices.SPIFLiveSpikesConnection([CNN_POP_LABEL], SPIF_IP_A, SPIF_PORT)
        conn_cnn.add_receive_callback(CNN_POP_LABEL, recv_nid)
        spif_cnn_output = p.Population(None, p.external_devices.SPIFOutputDevice(
            database_notify_port_num=conn_cnn.local_port, chip_coords=CHIP), label="cnn_output")
        p.external_devices.activate_live_output_to(cnn_pop, spif_cnn_output)
    else:
        conn_xyp = p.external_devices.SPIFLiveSpikesConnection([XYP_POP_LABEL], SPIF_IP_A, SPIF_PORT)
        conn_xyp.add_receive_callback(XYP_POP_LABEL, recv_nid_rc)
        spif_xyp_output = p.Population(None, p.external_devices.SPIFOutputDevice(
            database_notify_port_num=conn_xyp.local_port, chip_coords=CHIP), label="xyp_output")
        p.external_devices.activate_live_output_to(xyp_pop, spif_xyp_output)


    try:
        time.sleep(1)
        print("List of parameters:")
        print(f"\tNPC: {NPC_X} x {NPC_Y}")
        print(f"\tOutput {OUT_WIDTH} x {OUT_HEIGHT}")
        print(f"\tKernel Size: {len(kernel)}")
        print(f"\tKernel Sum: {abs(round(np.sum(kernel),3))}")
        user_input = input_with_timeout("Happy?\n ", 10)
    except KeyboardInterrupt:
        print("\n Simulation cancelled")
        quit()

    print("Waiting for rig-power to end ... ")    
    os.system(f"rig-power 172.16.223.{args.board-1}")
    p.run(RUN_TIME)

    p.end()

