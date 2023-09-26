import torch
import aestream
import time
import cv2
import pdb
import numpy as np
import math
import sys
sys.path.append('../common')
from tools import get_dimensions, get_shapes
import argparse
import csv
import os


def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-p', '--port', type= int, help="Port for events", default=5050)
    parser.add_argument('-s', '--scale', type=int, help="Image scale", default=1)
    parser.add_argument('-l', '--length', type=int, help="Image length", default=640)
    parser.add_argument('-w', '--width', type=int, help="Image width", default=480)

    return parser.parse_args()

if __name__ == '__main__':


    args = parse_args()
    new_l = math.ceil(args.length*args.scale)
    new_w = math.ceil(args.width*args.scale)
    window_name = 'Airhockey Display'
    cv2.namedWindow(window_name)

    # Stream events from UDP port 3333 (default)
    frame = np.zeros((args.length,args.width,3))

    with aestream.UDPInput((args.length, args.width), device = 'cpu', port=args.port) as stream1:
                
        while True:


            frame[0:args.length,0:args.width,1] =  stream1.read().numpy() 
            image = cv2.resize(frame.transpose(1,0,2), (new_l, new_w), interpolation = cv2.INTER_AREA)
            cv2.imshow(window_name, image)
            cv2.waitKey(1)


        