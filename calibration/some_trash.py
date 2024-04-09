

import subprocess
import time

# Command to measure

cmd = ""
cmd += "/opt/aestream/build/src/aestream "
cmd += "resolution 1280 720 undistortion ~/tabletop/calibration/luts/cam_lut_homography_prophesee.csv "
cmd += "output udp 172.16.222.30 5050 "
cmd += "input file ~/tabletop/recordings/all.aedat4 "

# Get current time in milliseconds
start_time = time.time()

# Execute the command using subprocess
process = subprocess.Popen(cmd, shell=True)
process.communicate()

# Get end time
end_time = time.time()

# Calculate elapsed time in milliseconds
elapsed_time_ms = (end_time - start_time) * 1000

print("Time taken:", elapsed_time_ms, "milliseconds")
