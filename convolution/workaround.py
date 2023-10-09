import os

try_counter = 0
while(True):
    try_counter += 1
    os.system(f"rm -rf reports/ ; clear ; echo 'Try #{try_counter}'; python3 rc_scnn_tracker.py -md cnn -tm 3")