import os
import time

counter = 0
max_count = 16
while counter < max_count:
    os.system("python3 pathsaver.py")
    time.sleep(2)
    counter += 1

os.system("python3 pathplotter.py")