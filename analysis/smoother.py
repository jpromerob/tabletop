import numpy as np
import pdb
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(data, weights, 'valid')


if __name__ == "__main__":

    fname = "trash"

    # Example data

    online_coordinates = np.load('online_coordinates.npy')
    offline_coordinates = np.load('offline_coordinates.npy')

   

    # Smoothing window size
    window_size = 40


    # Smoothing the signal using a moving average filter
    on_x = online_coordinates[:,0]
    on_y = online_coordinates[:,1]
    smooth_on_x = moving_average(on_x, window_size)
    smooth_on_y = moving_average(on_y, window_size)

    on_x = on_x[window_size//2:-window_size//2+1]
    on_y = on_y[window_size//2:-window_size//2+1]


    speed_x = smooth_on_x[1:]-smooth_on_x[:-1]
    speed_y = smooth_on_y[1:]-smooth_on_y[:-1]

    # full_speed = moving_average(np.sqrt(speed_x**2+speed_y**2), window_size)
    full_speed = np.sqrt(speed_x**2+speed_y**2)



    # Create a figure and three subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    # Plot speed along X axis
    axs[0].plot(on_x, 'k')
    axs[0].plot(smooth_on_x, 'r')
    axs[0].set_title('X Position')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('X')

    # Plot speed along Y axis
    axs[1].plot(on_y, 'k')
    axs[1].plot(smooth_on_y, 'g')
    axs[1].set_title('Y Position')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Y')

    # Plot overall speed
    axs[2].plot(speed_x, 'r')
    axs[2].plot(speed_y, 'g')
    axs[2].plot(full_speed, 'b')
    axs[2].set_title('Overall Speed')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Speed')

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.savefig(f'images/{fname}_speed_profiles.png')

    plt.clf()



    # Compute histogram
    hist, bins = np.histogram(full_speed, bins=20)

    # Plot histogram
    plt.bar(bins[:-1], hist, width=np.diff(bins), color='blue', alpha=0.7)
    plt.title('Histogram of Speed')
    plt.xlabel('Speed in [m/s]')
    plt.ylabel('# Occurrences')
    plt.xlim(0,5) 
    plt.ylim(0,1000) 
    plt.grid(True)
    plt.savefig(f'images/{fname}_speed_histogram.png')


    max_speed = round(np.max(full_speed),3)
    mean_speed = round(np.mean(full_speed),3)
    mode_value = round(bins[np.argmax(hist)],3)

    print(f"Max: {max_speed} | Mean: {mean_speed} | Mode: {mode_value}")