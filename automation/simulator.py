import socket
import time
import pdb
import os
import multiprocessing

def get_parameters(input_string):

    # Split the string by commas and remove leading/trailing spaces
    values_list = [x.strip() for x in input_string.split(',')]
    values_list[1] = float(values_list[1])
    values_list[2] = float(values_list[2])
    values_list[3] = int(values_list[3])

    return values_list[0], values_list[1:]


def stop_simulation():
    time.sleep(60*10)
    os.system("pkill -f scnn_tracker.py")
    

IP = '172.16.223.5' 
PORT = 2222  

# Create a UDP socket
synchronizer_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the IP and port
synchronizer_socket.bind((IP, PORT))



# Start the data reception process

while True:
    print("SIMULATOR: Waiting from signal from synchronizer")
    data, addr = synchronizer_socket.recvfrom(1024)
    message, params = get_parameters(data.decode())
    if message == "simulate":

        t_m = params[0]
        w_s = params[1]
        th = params[2]


        force_sim_to_stop = multiprocessing.Process(target=stop_simulation)
        force_sim_to_stop.daemon = True  # Set as a daemon process to exit when the main process exits
        force_sim_to_stop.start()

        print(f"Loading and running simulation with t_m:{t_m}, w_s:{w_s}, th:{th}")
        os.system(f"rm -rf ~/tabletop/*/reports")
        os.system(f"python3 ~/tabletop/convolution/scnn_tracker.py -tm {t_m} -ws {w_s} -th {th} -rt 4")


        force_sim_to_stop.join()
            
        
        print("Simulation ended ...")
        synchronizer_socket.sendto(b'good', addr)



