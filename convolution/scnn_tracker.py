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
sys.path.append('../common')
from tools import Dimensions


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

    parser.add_argument('-vpo', '--vis-port', type= int, help="Port Out", default=3331)
    parser.add_argument('-vip', '--vis-ip', type= str, help="IP out", default="172.16.222.199")
    parser.add_argument('-wpo', '--wta_port', type= int, help="Port Out", default=3333)
    parser.add_argument('-wip', '--wta_ip', type= str, help="IP out", default="172.16.223.10")
    parser.add_argument('-ks', '--ks', type=int, help="Kernel Size", default=45)
    parser.add_argument('-b', '--board', type=int, help="Board ID", default=1)
    parser.add_argument('-ws', '--w-scaler', type=float, help="Weight Scaler", default=0.10)
    parser.add_argument('-tm', '--tau-m', type=float, help="Tau m", default=3.0)
    parser.add_argument('-th', '--thickness', type=int, help="Kernel edge thickness", default=2)
    parser.add_argument('-rt', '--runtime', type=int, help="Runtime in [s]", default=240)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    dim = Dimensions.load_from_file('../common/homdim.pkl')
    print("Setting machines up ... ")
    CFG_FILE = f"spynnaker_{args.board}.cfg"
    SPIF_IP = spin_spif_map[f"{args.board}"]

    print("Generating Kernel ... ")
    kernel = make_whole_kernel(args.ks, dim.hs, args.w_scaler, args.thickness)

    print("Configuring Infrastructure ... ")
    SUB_WIDTH = 16
    SUB_HEIGHT = 8
    WIDTH = dim.fl
    HEIGHT = dim.fw

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

    VIS_IP = args.vis_ip
    VIS_PORT = args.vis_port
    WTA_IP = args.wta_ip
    WTA_PORT = args.wta_port
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
        vis_data = b""
        wta_data = b""
        np_spikes = np.array(spikes)
        # max_x_wta = 0
        # max_y_wta = 0
        # min_x_wta = 1000
        # min_y_wta = 1000
        for i in range(np_spikes.shape[0]):      
            x_vis = int(np_spikes[i]) % out_width
            y_vis = int(int(np_spikes[i]) / out_width)
            x_wta = int(x_vis/2)
            y_wta = int(y_vis/2)
            # # update maxs
            # if x_wta > max_x_wta:
            #     max_x_wta = x_wta
            # if y_wta > max_y_wta:
            #     max_y_wta = y_wta
            # # update mins
            # if x_wta < min_x_wta:
            #     min_x_wta = x_wta
            # if y_wta < min_y_wta:
            #     min_y_wta = y_wta
            polarity = 1
            packed_vis = (NO_TIMESTAMP + (polarity << P_SHIFT) + (y_vis << Y_SHIFT) + (x_vis << X_SHIFT))
            packed_wta = (NO_TIMESTAMP + (polarity << P_SHIFT) + (y_wta << Y_SHIFT) + (x_wta << X_SHIFT))
            vis_data += pack("<I", packed_vis)
            wta_data += pack("<I", packed_wta)
        sock.sendto(vis_data, (VIS_IP, VIS_PORT))
        sock.sendto(wta_data, (WTA_IP, WTA_PORT))
        # print(f"{max_x_wta},{max_y_wta},{min_x_wta},{min_y_wta}")


    print("Creating Network ... ")

    cell_params = {'tau_m': args.tau_m/math.log(2),
                'tau_syn_E': 1.0,
                'tau_syn_I': 1.0,
                'v_rest': -65.0,
                'v_reset': -65.0,
                'v_thresh': -60.0,
                'tau_refrac': 0.0, # 0.1 originally
                'cm': 1,
                'i_offset': 0.0
                }

    conn = p.external_devices.SPIFLiveSpikesConnection(
        [POP_LABEL], SPIF_IP, SPIF_PORT)
    conn.add_receive_callback(POP_LABEL, recv_nid)


    p.setup(timestep=1.0, n_boards_required=24)

    celltype = p.IF_curr_exp
    p.set_number_of_neurons_per_core(celltype, (NPC_X, NPC_Y))

    spif_retina = p.Population(
        WIDTH * HEIGHT, p.external_devices.SPIFRetinaDevice(
            pipe=0, width=WIDTH, height=HEIGHT,
            sub_width=SUB_WIDTH, sub_height=SUB_HEIGHT, chip_coords=CHIP),
        label="retina")

    convolution = p.ConvolutionConnector(kernel_weights=kernel)
    out_width, out_height = convolution.get_post_shape((WIDTH, HEIGHT))

    target_pop = p.Population(
        out_width * out_height, celltype(**cell_params),
        structure=p.Grid2D(out_width / out_height), label=POP_LABEL)

    p.Projection(spif_retina, target_pop, convolution, p.Convolution())

    spif_output = p.Population(None, p.external_devices.SPIFOutputDevice(
        database_notify_port_num=conn.local_port, chip_coords=CHIP), label="output")
    p.external_devices.activate_live_output_to(target_pop, spif_output)


    try:
        print("List of parameters:")
        print(f"\tNPC: {NPC_X} x {NPC_Y}")
        print(f"\tOutput {out_width} x {out_height}")
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

