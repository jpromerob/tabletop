import os
import time

'''
This script runs rec_analyzer.py for the same recording N times (on spiNNaker) 
The idea is to see how 'deterministic' it is ... 
'''

# Define the range for the loop
start = 1
end = 10

fname = "manual_240417_131828"

# Loop through the range and print the loop counter
for i in range(20, 30):
    os.system(f"python3 rec_analyzer.py -n 5000  -f ~/tabletop/recordings/{fname}.aedat4")
    time.sleep(2)
    os.system(f"cp images/{fname}_spinnaker_x_y_on_off_original.png some_runs/run_{i}.png; rm images/{fname}* ")

