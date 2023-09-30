

import socket
import pdb
import time
import numpy as np
import os

simulator_ip = '172.16.223.5'
simulator_port = 2222 

streamer_ip = '172.16.222.199'
streamer_port = 2223 

recorder_ip = '172.16.222.199'
recorder_port = 2224 


# Create a UDP socket
simulator_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
streamer_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recorder_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


simulator_socket.settimeout(15*60) # 15 minutes

parameters = [(0.10, 3.0, 2)]
for th in [2,1]:
    for t_m in np.arange(1.0,10.0,1.0):
        for w_s in np.arange(0.16, 0.06, -0.02):
                new_combination = (round(w_s,2), round(t_m,2), th)
                parameters.append(new_combination)

# pdb.set_trace()

loading_time = 60*6 # seconds (i.e 8 minutes)
run_counter = 0
start_time = time.time()
for w_s, t_m, th in parameters:
    
    run_counter += 1

    # Calculate the elapsed time in seconds
    elapsed_time = int((time.time() - start_time)/60)
    print(f"SYNCHRONIZER: {run_counter-1}/{len(parameters)} runs after {elapsed_time} minutes")
    print(f"Start of test with tau_m:{t_m}, w_scaler: {w_s}, thickness {th}")
    # pdb.set_trace()
    
    
    # Trigger Simulator
    message = f"simulate,{t_m},{w_s},{th}"
    simulator_socket.sendto(message.encode(), (simulator_ip, simulator_port))

    print("Waiting for SpiNNaker simulation to be running")
    time.sleep(loading_time)

    message = "stream"
    streamer_socket.sendto(message.encode(), (streamer_ip, streamer_port))
    print("Waiting for AEstream to start streaming")
    time.sleep(3)

    # Trigger Recorder
    message = f"record,{int(t_m)}_{int(w_s*100)}_{th}"
    recorder_socket.sendto(message.encode(), (recorder_ip, recorder_port))   

    # Wait for Recorder
    print("Waiting for Recorder to stop")
    reply, addr = recorder_socket.recvfrom(1024) 
    answer = reply.decode()
    if answer != "good":
        print("Error while recording")
        os.system("pkill -f double_rec.py")
    else:
        print("Recorder Stopped correctly")

    # Wait for Streamer
    print("Waiting for Streamer to stop")
    reply, addr = streamer_socket.recvfrom(1024)    
    answer = reply.decode()
    if answer != "good":
        print("Error while stopping streamer")
        os.system("pkill -f aestream")
    else:
        print("Streamer Stopped correctly")


    # Wait for Simulator
    print("Waiting for Simulator to stop")
    try:
        reply, addr = simulator_socket.recvfrom(1024)    
        answer = reply.decode()
        if answer != "good":
            print("Error while stopping simulator")
        else:
            print("Simulator Stopped correctly")
    except socket.timeout:
        print("Something went wrong with SpiNNaker")
        break
    
    print(f"End of test with tau_m:{t_m}, w_scaler: {w_s}, thickness {th}")
    
    elapsed_time = int((time.time() - start_time)/60)
    t_remain = (len(parameters)-run_counter)*elapsed_time/(run_counter)
    print(f"Time remaining: {t_remain} minutes")
    print("\n\n\n")


# Close the socket when done
simulator_socket.close()
recorder_socket.close()
streamer_socket.close()
