import torch
import aestream
import time
import cv2
import pdb
import numpy as np
import math
import argparse
import csv
import os

# pdb.set_trace()


parser = argparse.ArgumentParser(description='Visualizer')
parser.add_argument('-p1', '--port1', type= int, help="Port for events", default=1987)
parser.add_argument('-p2', '--port2', type= int, help="Port for events", default=1988)
parser.add_argument('-s', '--scale', type= float, help="Image scale", default=1)
parser.add_argument('-x', '--width', type= int, help="X res = width", default=64)
parser.add_argument('-y', '--height', type= int, help="Y res = height", default=64)

args = parser.parse_args()

e_port_1 = args.port1
e_port_2 = args.port2
width = args.width
height = args.height


cv2.namedWindow('Durin On Board')

# Stream events from UDP port 3333 (default)
black = np.zeros((width+1,height+1,3))
frame = np.zeros((width+1,height+1,3))


with aestream.UDPInput((width, height), device = 'cpu', port=e_port_1) as stream1:
    with aestream.UDPInput((width+height,1), device = 'cpu', port=e_port_2) as stream2:
            
        while True:
                
                aux = stream1.read().numpy()
                xy_aux = stream2.read().numpy()

                frame[0:width+1,0:height+1,:] = black

                frame[0:width,0:height,1] = aux # Provides a (width, height) tensor
                
                frame[0:width,height,2] = xy_aux[0:width,0]
                frame[width,0:height,2] = np.transpose(xy_aux[width:,0])

                nx = args.scale*args.width/width
                ny = args.scale*args.height/height

                image = cv2.resize(frame.transpose(1,0,2), (math.ceil(width*nx),math.ceil(height*ny)), interpolation = cv2.INTER_AREA)
                
                
                cv2.imshow('Durin On Board', image)
                cv2.waitKey(1)

    


        
