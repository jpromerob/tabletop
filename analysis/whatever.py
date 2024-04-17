import numpy as np
import matplotlib.pyplot as plt
from rec_analyzer import analyze_speed, find_latency_and_error

online_coordinates = np.load("online.npy")
offline_coordinates = np.load("offline.npy")

plt.plot(online_coordinates[:,0])
plt.plot(offline_coordinates[:,1])

plt.show()


delta_t = 9 # bin size in [ms]
window_size = 40
nb_shifts = 100

fname = "fname"
iname = "iname"

max_speed, mean_speed, mode_value = analyze_speed(fname, offline_coordinates, window_size)
latency, error, min_error = find_latency_and_error(online_coordinates, offline_coordinates, nb_shifts, iname)


print(f"Latency: {latency} ms | Error[t=0]: {error} [mm]")
