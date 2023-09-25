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
import time


'''
JPRB 25/09/2023 14:38: I have the impression that:
 - when the puck moves to fast, it's not properly tracked ... 
'''

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

def make_kernel_circle(r, k_sz,weight, kernel):
    # pdb.set_trace()
    var = int((k_sz+1)/2-1)
    a = np.arange(0, 2 * math.pi, 0.01)
    dx = np.round(r * np.sin(a)).astype("uint32")
    dy = np.round(r * np.cos(a)).astype("uint32")
    kernel[var + dx, var + dy] = weight

def make_whole_kernel(k_sz):

    pos_radi = [19,11,5]

    w_scaler = 0.012
    pos_w = 1
    neg_w = -pos_w * 0.50
    gen_w = neg_w * 0.35

    kernel = gen_w*w_scaler*np.ones((k_sz, k_sz), dtype=np.float32)

    thickness = 2
    for r in pos_radi:
        for i in np.arange(r-thickness+1, r+1):
            make_kernel_circle(i, k_sz, pos_w*w_scaler, kernel) # 38px

    np.save("../common/kernel.npy", kernel)
    send_kernel()


    cmap = plt.cm.get_cmap('viridis')
    plt.imshow(kernel, cmap=cmap, interpolation='nearest')
    colorbar = plt.colorbar()
    colorbar.set_label('Color Scale')

    plt.savefig("kernel.png")


    sum_k = np.sum(kernel)
    print(f"Kernel Sum: {sum_k}")
    
    return kernel


spin_spif_map = {"1": "172.16.223.2", 
                 "37": "172.16.223.106", 
                 "43": "172.16.223.98",
                 "13": "172.16.223.10",
                 "121": "172.16.223.122",
                 "129": "172.16.223.130"}

def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-po', '--port-out', type= int, help="Port Out", default=3331)
    parser.add_argument('-ip', '--ip-out', type= str, help="IP out", default="172.16.222.199")
    parser.add_argument('-ks', '--ks', type=int, help="Kernel Size", default=45)
    parser.add_argument('-b', '--board', type=int, help="Board ID", default=1)
    parser.add_argument('-rx', '--res-x', type=int, help="Resolution (X: width)", default=640)
    parser.add_argument('-ry', '--res-y', type=int, help="Resolution (Y: height)", default=480)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    print("Setting machines up ... ")
    CFG_FILE = f"spynnaker_{args.board}.cfg"
    SPIF_IP = spin_spif_map[f"{args.board}"]
    os.system(f"rig-power 172.16.223.{args.board-1}")

    print("Generating Kernel ... ")
    kernel = make_whole_kernel(args.ks)

    print("Creating Network ... ")
    SUB_WIDTH = 16
    SUB_HEIGHT = 8
    WIDTH = args.res_x
    HEIGHT = args.res_y


    print("Calculating number of neurons per core")
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
    
    print(f"NPC: {x} x {y}")

    # pdb.set_trace()

    NPC_X = x
    NPC_Y = y

    MY_PC_IP = args.ip_out
    MY_PC_PORT = args.port_out
    SPIF_PORT = 3332
    POP_LABEL = "target"
    RUN_TIME = 1000*60*240
    CHIP = (0, 0)

    convolution = p.ConvolutionConnector(kernel_weights=kernel)
    out_width, out_height = convolution.get_post_shape((WIDTH, HEIGHT))

    print(f"Output {out_width} x {out_height}")
    time.sleep(5)

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
            x = int(np_spikes[i]) % out_width
            y = int(int(np_spikes[i]) / out_width)
            polarity = 1
            packed = (NO_TIMESTAMP + (polarity << P_SHIFT) + (y << Y_SHIFT) + (x << X_SHIFT))
            data += pack("<I", packed)
        sock.sendto(data, (MY_PC_IP, MY_PC_PORT))



    conn = p.external_devices.SPIFLiveSpikesConnection(
        [POP_LABEL], SPIF_IP, SPIF_PORT)
    conn.add_receive_callback(POP_LABEL, recv_nid)

    cell_params = {'tau_m': 5.0,
                'tau_syn_E': 1.0,
                'tau_syn_I': 1.0,
                'v_rest': -65.0,
                'v_reset': -65.0,
                'v_thresh': -60.0,
                'tau_refrac': 0.0, # 0.1 originally
                'cm': 1,
                'i_offset': 0.0
                }

    p.setup(timestep=1.0, n_boards_required=24,  cfg_file=CFG_FILE)

    celltype = p.IF_curr_exp
    p.set_number_of_neurons_per_core(celltype, (NPC_X, NPC_Y))

    spif_retina = p.Population(
        WIDTH * HEIGHT, p.external_devices.SPIFRetinaDevice(
            pipe=0, width=WIDTH, height=HEIGHT,
            sub_width=SUB_WIDTH, sub_height=SUB_HEIGHT, chip_coords=CHIP),
        label="retina")


    target_pop = p.Population(
        out_width * out_height, celltype(**cell_params),
        structure=p.Grid2D(out_width / out_height), label=POP_LABEL)

    p.Projection(spif_retina, target_pop, convolution, p.Convolution())

    spif_output = p.Population(None, p.external_devices.SPIFOutputDevice(
        database_notify_port_num=conn.local_port, chip_coords=CHIP), label="output")
    p.external_devices.activate_live_output_to(target_pop, spif_output)


    p.run(RUN_TIME)

    p.end()
