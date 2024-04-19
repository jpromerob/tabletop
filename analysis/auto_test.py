import os
from datetime import datetime
import time
import pdb

# min_sparsity, max_sparsity, min_delta, max_delta
speed_mapper = {
    "low": (0.08, 0.12, 0.4, 1),
    "medium": (0.14, 0.36, 1.8, 3),
    "high": (0.4, 0.8, 4, 12)
}


nb_runs = 10
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

        print(f"\n\n{speed} run #{i+1}/{nb_runs}")

        fname = f"Synthetic_{datetime.now().strftime('%y%m%d_%H%M%S')}"

        sparsity = round(min_sparsity+i*(max_sparsity-min_sparsity)/nb_runs,3)

        delta = round(min_delta+i*(max_delta-min_delta)/nb_runs,3)
        print(f"Fname: {fname} | Sparsity: {sparsity} | Delta: {delta}")

        os.system(f"python3 gen_analyzer.py -n {nb_pts} -s {sparsity} -d {delta} -f {fname} -g")
        time.sleep(5)

        os.system(f"python3 gen_analyzer.py -n {nb_pts} -s {sparsity} -d {delta} -f {fname}")
        time.sleep(5)
    




