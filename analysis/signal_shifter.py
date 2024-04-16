import numpy as np
import pdb
import matplotlib.pyplot as plt
import math

PLOT_FLAG = False

# Load the saved numpy arrays
online_x = np.load('online_x.npy')
offline_x = np.load('offline_x.npy')
online_y = np.load('online_y.npy')
offline_y = np.load('offline_y.npy')

error = np.zeros(20)
for t in range(20):

    if t > 0:
        new_offline_x = offline_x[0:-t]
        new_offline_y = offline_y[0:-t]
    else:
        new_offline_x = offline_x
        new_offline_y = offline_y

    new_online_x = online_x[t:]
    new_online_y = online_y[t:]

    print(new_online_x.shape)
    print(new_online_y.shape)
    print(new_offline_x.shape)
    print(new_offline_y.shape)

    e_x_array = (new_online_x-new_offline_x)**2
    e_y_array = (new_online_y-new_offline_y)**2

    e_x = np.mean(e_x_array)
    e_y = np.mean(e_y_array)

    error[t] = int(math.sqrt(e_x**2 + e_y**2))

    print(f"Error: {e_x}, {e_y}")

    if PLOT_FLAG:

        # Create figure and subplots
        fig, axs = plt.subplots(2,  figsize=(20, 16))

        # Subplot 1
        axs[0].plot(new_online_x, label='Online X')
        axs[0].plot(new_offline_x, label='Offline X')
        axs[0].set_ylim(0, 165)  # Set y-axis limit
        axs[0].legend()

        # Subplot 2
        axs[1].plot(new_online_y, label='Online Y')
        axs[1].plot(new_offline_y, label='Offline Y')
        axs[1].set_ylim(0, 256)  # Set y-axis limit
        axs[1].legend()

        plt.suptitle(f'Delta t = {t} --> Error = {error[t]}')  # Add super title

        # Adjust layout
        plt.tight_layout()

        # Show plot
        plt.show()

latency = np.argmin(error)
print(latency)