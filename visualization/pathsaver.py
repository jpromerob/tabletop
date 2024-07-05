import aestream
import cv2
import os
import numpy as np
import math
import sys
import h5py
import time
sys.path.append('../common')
from tools import Dimensions
import argparse
import multiprocessing
import pdb

import os
import re

def find_next_available_path_number(directory):
    # Regular expression to match files named path_<number>.h5
    pattern = re.compile(r'^path_(\d+)\.h5$')

    # Get a list of all files in the directory
    files = os.listdir(directory)

    # Extract the numbers from the filenames
    numbers = []
    for file in files:
        match = pattern.match(file)
        if match:
            numbers.append(int(match.group(1)))

    # Find the smallest missing integer
    M = 1
    while M in numbers:
        M += 1

    return M

import open3d as o3d


def parse_args():

    parser = argparse.ArgumentParser(description='Frame Saver')

    parser.add_argument('-p', '--port', type= int, help="Port for events", default=3330)
    parser.add_argument('-s', '--scale', type=int, help="Image scale", default=2)
    parser.add_argument('-l', '--length', type=int, help="Image length", default=256)
    parser.add_argument('-w', '--width', type=int, help="Image width", default=165)
    parser.add_argument('-v','--visualize', action='store_true', help='Visualize Events')

    return parser.parse_args()

def capture_process(args):
    vis_flag = args.visualize

    dim = Dimensions.load_from_file('../common/homdim.pkl')
    if args.length == 0 or args.width == 0:    
        res_x = dim.fl
        res_y = dim.fw
    else:
        res_x = args.length
        res_y = args.width

    new_l = math.ceil(res_x*args.scale)
    new_w = math.ceil(res_y*args.scale)

    if vis_flag:
        window_name = f'Display From AEstream (port {args.port})'
        cv2.namedWindow(window_name)

    # Stream events from UDP port 3333 (default)
    frame = np.zeros((res_x,res_y,3))

    interval = 0.001
    nb_frames = int(2.5/interval)
    cloud = np.zeros((res_x,res_y, nb_frames))

    ix_cloud = 0
    with aestream.UDPInput((res_x, res_y), device = 'cpu', port=args.port) as stream1:
        
        next_time = time.time() + interval
        while True:

            current_time = time.time()
            if current_time >= next_time:
                ix_cloud += 1 
                if ix_cloud == nb_frames:
                    print(f"End of capture after {ix_cloud} frames")
                    break
                next_time += interval


            cloud[:,:,ix_cloud] += stream1.read().numpy()

            if vis_flag:
                frame[0:res_x,0:res_y,1] =  stream1.read().numpy() 
                image = cv2.resize(frame.transpose(1,0,2), (new_l, new_w), interpolation = cv2.INTER_AREA)
                
                center_x = int(image.shape[1] // 2)
                center_y = int(image.shape[0] // 2)
                
                cv2.imshow(window_name, image)            
                cv2.waitKey(1)

    if vis_flag:
        cv2.destroyAllWindows()
    
    next_M = find_next_available_path_number("paths")

    # Save the array to an HDF5 file with gzip compression
    with h5py.File(f'paths/path_{next_M}.h5', 'w') as f:
        f.create_dataset(f'cloud', data=cloud, compression='gzip')

    os.system("pkill -f puck_generator")
    
def trigger_process(args):
    os.system("python3 ~/tabletop/generation/puck_generator.py -s 0.3 -d 0.2 -m zigzag")

if __name__ == '__main__':

    args = parse_args()

    capture_proc = multiprocessing.Process(target=capture_process, args=(args,))
    capture_proc.start()

    trigger_proc = multiprocessing.Process(target=trigger_process, args=(args,))
    trigger_proc.start()


    # gt_proc.join()
    capture_proc.join()  
    trigger_proc.join()  
 
        