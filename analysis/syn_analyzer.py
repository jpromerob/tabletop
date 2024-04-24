import aestream
import numpy as np
import time
import torch
import torch.nn as nn
import sys
import multiprocessing
import argparse
import os


import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
import matplotlib.pyplot as plt

from com_analyzer import *


def gen_ev_process(shared_data):

    cmd =""
    cmd += f"python3 ~/tabletop/generation/puck_generator.py "
    cmd += f"-s {shared_data['sparsity']} -d {shared_data['delta']} "
    cmd += f"-ox {shared_data['offx']} -oy {shared_data['offy']} "
    cmd += f" -m {shared_data['gmode']}"
    if shared_data['gpu']:
        print("Launch event generation for GPU")
        cmd += f" -g"
    else:
        print("Launch event generation for SpiNNaker")
        cmd += f""
    
    os.system(cmd)



def initialize_shared_data(args):

    shared_data = multiprocessing.Manager().dict()

    dim = Dimensions.load_from_file('../common/homdim.pkl')
    shared_data['res_x'] = dim.fl
    shared_data['res_y'] = dim.fw
    shared_data['hs'] = dim.hs

    shared_data['board'] = args.board

    shared_data['fname'] = args.fname 

    shared_data['offx'] = args.offx
    shared_data['offy'] = args.offy

    shared_data['delta'] = args.delta
    shared_data['sparsity'] = args.sparsity
    shared_data['gpu'] = args.gpu
    shared_data['data_origin'] = args.dorigin
    shared_data['nb_frames'] = args.nb_frames
    shared_data['gmode'] = args.gmode

    shared_data['done_storing_data'] = False
    shared_data['intime_pose'] = (0,0)
    shared_data['delayed_pose'] = (0,0)

    shared_data['port'] = args.port

    return shared_data


def parse_args():

    parser = argparse.ArgumentParser(description='Display From AEstream')

    parser.add_argument('-n', '--nb-frames', type= int, help="Max number of frames", default=2000)
    parser.add_argument('-f', '--fname', type= str, help="File Name", default="synthetic")
    parser.add_argument('-g','--gpu', action='store_true', help='Run on GPU!')
    parser.add_argument('-b', '--board', type= int, help="Board sending events", default=43)
    parser.add_argument('-p', '--port', type= int, help="Port for events coming from GPU|SpiNNaker", default=5050)

    parser.add_argument('-do', '--dorigin', type= str, help="Data Origin", default="syn")

    parser.add_argument('-m', '--gmode', type= str, help="Generation Mode", default="circle")
    parser.add_argument('-s', '--sparsity', type= float, help="Sparsity", default=0.6)
    parser.add_argument('-d', '--delta', type= float, help="Delta (puck speed)", default=3.0)
    parser.add_argument('-ox', '--offx', type=float, help="Offset X (percentage)", default=0)
    parser.add_argument('-oy', '--offy', type=float, help="Offset Y (percentage)", default=0)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    shared_data = initialize_shared_data(args)

    delayed_proc = multiprocessing.Process(target=delayed_process, args=(shared_data,))
    delayed_proc.start()

    intime_proc = multiprocessing.Process(target=intime_process, args=(shared_data,))
    intime_proc.start()

    genev_proc = multiprocessing.Process(target=gen_ev_process, args=(shared_data,))
    genev_proc.start()


    # gt_proc = multiprocessing.Process(target=ground_truth_process, args=(shared_data,))
    # gt_proc.start()

    ground_truth_process(shared_data)

    # gt_proc.join()
    delayed_proc.join()  
    intime_proc.join()   
    genev_proc.join()  
 

