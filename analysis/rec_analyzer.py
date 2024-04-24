import aestream
import numpy as np
import time
import sys
import multiprocessing
import argparse
import os


import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
import matplotlib.pyplot as plt

sys.path.append('../common')
from tools import Dimensions

from com_analyzer import *




    




def ground_truth_process(shared_data):

    time.sleep(0.5)

    if shared_data['gpu']:
        pipeline = 'gpu'
    else:
        pipeline = 'spinnaker'

    fname = clean_fname(shared_data['fname'])
    iname = f"{fname}_{pipeline}"


    delta_t = 9 # CNN bin size in [ms]
    window_size = 40
    nb_shifts = 100


    delayed_coordinates, intime_coordinates = get_coordinates(shared_data, delta_t)


    delayed_reps = count_repeated_elements(delayed_coordinates)
    max_speed, mean_speed, mode_value = analyze_speed(iname, intime_coordinates, window_size)
    latency, error, t_shift = find_latency_and_error(delayed_coordinates, intime_coordinates, nb_shifts, iname)

    real_error = round(error[0],3)
    best_error = round(error[latency],3)

    write_to_csv(shared_data['data_origin'], fname, pipeline, latency, real_error, best_error, max_speed, mean_speed, delayed_reps)


def initialize_shared_data(args):

    shared_data = multiprocessing.Manager().dict()

    dim = Dimensions.load_from_file('../common/homdim.pkl')
    shared_data['res_x'] = dim.fl
    shared_data['res_y'] = dim.fw
    shared_data['hs'] = dim.hs

    shared_data['fname']=args.fname
    shared_data['gpu'] = args.gpu
    shared_data['data_origin'] = args.dorigin
    shared_data['port'] = args.port
    shared_data['nb_frames'] = args.nb_frames
    shared_data['board'] = args.board

    shared_data['done_storing_data'] = False
    shared_data['delayed_pose'] = (0,0)

    


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

    ae_proc = multiprocessing.Process(target=aestream_process, args=(shared_data,))
    ae_proc.start()

    fl_proc = multiprocessing.Process(target=delayed_process, args=(shared_data,))
    fl_proc.start()

    # gt_proc = multiprocessing.Process(target=ground_truth_process, args=(shared_data,))
    # gt_proc.start()

    ground_truth_process(shared_data)

    # gt_proc.join()
    fl_proc.join()    
    ae_proc.join()

