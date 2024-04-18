import os
from datetime import datetime
import time

nb_runs = 10
max_sparsity = 0.8
min_sparsity = 0.2
max_delta = 5
min_delta = 2

nb_pts = 10000

for i in range(nb_runs):

    print(f"Run #{i+1}/{nb_runs}")

    fname = f"Synthetic_{datetime.now().strftime('%y%m%d_%H%M%S')}"

    sparsity = round(min_sparsity+i*(max_sparsity-min_sparsity)/nb_runs,3)

    delta = round(min_delta+i*(max_delta-min_delta)/nb_runs,3)
    print(f"Fname: {fname} | Sparsity: {sparsity} | Delta: {delta}")

    os.system(f"python3 gen_analyzer.py -n {nb_pts} -s {sparsity} -d {delta} -f {fname} -g")
    time.sleep(5)

    os.system(f"python3 gen_analyzer.py -n {nb_pts} -s {sparsity} -d {delta} -f {fname}")
    time.sleep(5)
