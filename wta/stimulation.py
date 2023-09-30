import multiprocessing
import socket
import pdb
import math
import sys
import datetime
import time
import numpy as np
import argparse
import random
from struct import pack
import os
import ctypes
from time import sleep


def event_generator(shape, w, h, et):  

    if shape == "circle":
        cx, cy = int(w/2), int(h/2)  # Center coordinates of the circle
        r = int(min(w,h)/4)  # Radius of the circle

        step = 0
        while True: 
            
            angle = 2 * math.pi * (step%360) / 360
            x = int(cx + r * math.cos(angle))
            y = int(cy + r * math.sin(angle))
            step += 1

            for i in range(et):                
                time.sleep(0.002)
                yield ((x,y))  

            # print(f"Sending ({x},{y})")
    elif shape == "square":
        
        while(True):
            for x in range(w):
                for y in range(h):
                    for i in range(et):
                        
                        time.sleep(0.001)
                        yield ((x,y))  

                    # print(f"Sending ({x},{y})")
                    

def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-p', '--port', type= int, help="Port Out", default=3333)
    parser.add_argument('-ip', '--ip', type= str, help="IP out", default="172.16.223.10")
    parser.add_argument('-s', '--shape', type= str, help="Figure", default="circle")
    parser.add_argument('-sw', '--width', type= int, help="Shape Width", default=32)
    parser.add_argument('-sh', '--height', type= int, help="Shape Height", default=32)
    parser.add_argument('-et', '--ev-tstep', type= int, help="Events per timestep", default=10000)

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    p_shift = 15
    y_shift = 0
    x_shift = 16
    no_timestamp = 0x80000000
    sock_data = b""
    ip_addr = args.ip
    spif_port = args.port

    ev_gen = event_generator(args.shape, args.width, args.height, args.ev_tstep)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"Using SPIF ({ip_addr}) on port {spif_port}")

    while True:

        e = next(ev_gen)
        x = e[0]
        y = e[1]
        polarity = 1

        packed = (no_timestamp + (polarity << p_shift) + (y << y_shift) + (x << x_shift))
        sock_data += pack("<I", packed)
        
            
        sock.sendto(sock_data, (ip_addr, spif_port))
        sock.sendto(sock_data, ("172.16.222.199", 1987))
        sock_data = b""
        
    sock.close()
            