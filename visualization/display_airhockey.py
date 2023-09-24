import torch
import aestream
import time
import cv2
import pdb
import numpy as np
import math
import sys
sys.path.append('../common')
from tools import add_markers, get_dimensions, get_shapes
import argparse
import csv
import os


def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-p', '--port', type= int, help="Port for events", default=5151)
    parser.add_argument('-s', '--scale', type=int, help="Image scale", default=1)

    return parser.parse_args()

if __name__ == '__main__':


    args = parse_args()
    cv2.namedWindow('Airhocket Display')

    # Stream events from UDP port 3333 (default)
    black = np.zeros((640,480,3))
    frame = black


    l, w, ml, mw, dlx, dly = get_dimensions()
    field, line, goals, circles, radius = get_shapes(l, w, ml, mw, dlx, dly, args.scale)
    red = (0, 0, 255)

    print(line)
    with aestream.UDPInput((640, 480), device = 'cpu', port=args.port) as stream1:
                
        while True:


            frame[0:640,0:480,1] =  stream1.read().numpy() # Provides a (640, 480) tensor           



            image = cv2.resize(frame.transpose(1,0,2), (math.ceil(640*args.scale),math.ceil(480*args.scale)), interpolation = cv2.INTER_AREA)
            
            # Define the four corners of the field
            corners = np.array(field, np.int32)
            cv2.polylines(image, [corners], isClosed=True, color=red, thickness=1)

            for goal in goals:
                corners = np.array(goal, np.int32)
                cv2.polylines(image, [corners], isClosed=True, color=red, thickness=1)

            for cx, cy in circles:
                # print(f"Center: x={cx}, y={cy}")
                cv2.circle(image, (cx, cy), radius, color=red, thickness=1)
            cv2.line(image, line[0], line[1], color=red, thickness=1)

            cv2.imshow('Airhocket Display', image)
            cv2.waitKey(1)


        