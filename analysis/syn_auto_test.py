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
    "high": (0.4, 0.8, 4, 12),
    "medium": (0.28, 0.36, 1.8, 3),
    "low": (0.16, 0.24, 0.4, 1)
}


nb_runs = 3
nb_steps = 10
nb_pts = 2000


for speed in speed_mapper:
    print("\n\n\n*****************************************")
    print(f"Starting simulation for {speed} speed")
    print("*****************************************\n\n\n")
    min_sparsity = speed_mapper[speed][0]
    max_sparsity = speed_mapper[speed][1]
    min_delta = speed_mapper[speed][2]
    max_delta = speed_mapper[speed][3]

    for i in range(nb_steps):

        off_x, off_y = generate_random_floats(0.2)

        dname = f"Synthetic_{datetime.now().strftime('%y%m%d_%H%M%S')}"

        sparsity = round(min_sparsity+i*(max_sparsity-min_sparsity)/nb_steps,3)

        delta = round(min_delta+i*(max_delta-min_delta)/nb_steps,3)
        print(f"Fname: {dname} | Sparsity: {sparsity} | Delta: {delta}")

        for j in range(nb_runs):

            print(f"\n\n{speed} run #{i+1}.{j+1}/{nb_steps}")

            fname = dname + f"_v{j+1}"

            # print(fname)
            # # RUN ON GPU
            # os.system(f"python3 analyzer.py -do syn -n {nb_pts} -s {sparsity} -d {delta} -f {fname} -ox {off_x} -oy {off_y} -m circle -g")
            # time.sleep(5)

            # # RUN ON SPINNAKER
            # os.system(f"python3 analyzer.py -do syn -n {nb_pts} -s {sparsity} -d {delta} -f {fname} -ox {off_x} -oy {off_y} -m circle ")
            # time.sleep(5)

            print(fname)
            # RUN ON GPU
            os.system(f"python3 analyzer.py -do syn -n {nb_pts} -s {sparsity} -d {delta/10} -f {fname} -ox {off_x} -oy {off_y} -m zigzag -g")
            time.sleep(5)

            # RUN ON SPINNAKER
            os.system(f"python3 analyzer.py -do syn -n {nb_pts} -s {sparsity} -d {delta/10} -f {fname} -ox {off_x} -oy {off_y} -m zigzag ")
            time.sleep(5)
            
        




