import socket
import time
import pdb
import os

def get_param_label(input_string):

    # Split the string by commas and remove leading/trailing spaces
    msg_list = [x.strip() for x in input_string.split(',')]

    return msg_list[0], msg_list[1]


IP = '172.16.222.199'  
PORT = 2224  

# Create a UDP socket
synchronizer_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the IP and port
synchronizer_socket.bind((IP, PORT))

while True:
    print("RECORDER: Waiting from signal from synchronizer")
    data, addr = synchronizer_socket.recvfrom(1024)
    message, param_label = get_param_label(data.decode())
    
    if message == "record":
        print("Recording a video for N seconds ...")
        os.system(f"python3 ~/tabletop/visualization/double_rec.py -at 2 -d 10 -pl {param_label}")




    synchronizer_socket.sendto(b'good', addr)
