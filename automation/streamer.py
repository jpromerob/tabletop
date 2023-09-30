import socket
import time
import os

IP = '172.16.222.199'  
PORT = 2223  

# Create a UDP socket
synchronizer_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the IP and port
synchronizer_socket.bind((IP, PORT))

while True:
    print("STREAMER: Waiting from signal from synchronizer")
    data, addr = synchronizer_socket.recvfrom(1024)
    message = data.decode()
    if message == "stream":

        print("Streaming a video for N seconds ...")
        cmd = ""
        cmd += f"/opt/aestream/build/src/aestream "
        cmd += f"input file ~/tabletop/recordings/fast_full_game.aedat4 "
        cmd += f"output udp 172.16.222.199 3330 172.16.223.2 3333 "
        cmd += f"resolution 1280 720 "
        cmd += f"undistortion ~/tabletop/calibration/cam_lut_homography_prophesee.csv"

        os.system(cmd)
        print("Streaming ended ...")
        synchronizer_socket.sendto(b'good', addr)


