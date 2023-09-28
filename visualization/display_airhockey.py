import torch
import aestream
import time
import cv2
import pdb
import numpy as np
import math
import sys
sys.path.append('../common')
from tools import Dimensions, get_shapes
import argparse
import csv
import os


def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-p', '--port', type= int, help="Port for events", default=5050)
    parser.add_argument('-s', '--scale', type=int, help="Image scale", default=1)
    parser.add_argument('-l', '--length', type=int, help="Image length", default=0)
    parser.add_argument('-w', '--width', type=int, help="Image width", default=0)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    dim = Dimensions.load_from_file('../common/homdim.pkl')
    if args.length == 0 or args.width == 0:    
        res_x = dim.fl
        res_y = dim.fw
    else:
        res_x = args.length
        res_y = args.width

    new_l = math.ceil(res_x*args.scale)
    new_w = math.ceil(res_y*args.scale)
    window_name = 'Airhockey Display'
    cv2.namedWindow(window_name)

    # Stream events from UDP port 3333 (default)
    frame = np.zeros((res_x,res_y,3))

    with aestream.UDPInput((res_x, res_y), device = 'cpu', port=args.port) as stream1:
                
        while True:


            frame[0:res_x,0:res_y,1] =  stream1.read().numpy() 
            image = cv2.resize(frame.transpose(1,0,2), (new_l, new_w), interpolation = cv2.INTER_AREA)
            cv2.imshow(window_name, image)
            cv2.waitKey(1)


        