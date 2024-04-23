import os
from datetime import datetime
import time
import pdb
import random


'''
This Script calls gen_analizer.py (SpiNNaker and GPU) using different combinations of stimulus speed/sparsity
'''

def generate_random_floats(offset_max):
    off_x = round(random.uniform(-offset_max, offset_max),2)
    off_y = round(random.uniform(-offset_max, offset_max),2)
    return off_x, off_y

# min_sparsity, max_sparsity, min_delta, max_delta
speed_mapper = {
    "low": (0.04, 0.04, 0.02, 0.02),
    "medium": (0.12,0.12, 0.08,0.08),
    "high": (0.8, 0.8, 4, 4)
}


nb_runs = 1
nb_pts = 10000


for speed in speed_mapper:
    print("\n\n\n*****************************************")
    print(f"Starting simulation for {speed} speed")
    print("*****************************************\n\n\n")
    min_sparsity = speed_mapper[speed][0]
    max_sparsity = speed_mapper[speed][1]
    min_delta = speed_mapper[speed][2]
    max_delta = speed_mapper[speed][3]

    for i in range(nb_runs):

        off_x, off_y = generate_random_floats(0.2)

        print(f"\n\n{speed} run #{i+1}/{nb_runs}")

        fname = f"Synthetic_{datetime.now().strftime('%y%m%d_%H%M%S')}"

        sparsity = round(min_sparsity+i*(max_sparsity-min_sparsity)/nb_runs,3)

        delta = round(min_delta+i*(max_delta-min_delta)/nb_runs,3)
        print(f"Fname: {fname} | Sparsity: {sparsity} | Delta: {delta}")

        # os.system(f"python3 syn_analyzer.py -n {nb_pts} -s {sparsity} -d {delta} -f {fname} -ox {off_x} -oy {off_y} -m line_x -g")
        # time.sleep(5)

        os.system(f"python3 syn_analyzer.py -n {nb_pts} -s {sparsity} -d {delta} -f {fname} -ox {off_x} -oy {off_y} -m line_x ")
        time.sleep(5)
    




