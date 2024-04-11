import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pdb


def interpolate(signal):
    if signal[-1]==signal[-2]:
        signal[-1] += 0.00001
    i = 0
    fail_count = 0
    while i < len(signal):

        if signal[i] == signal[i-1]:
            j = i
            while j < len(signal)-1 and signal[j] == signal[i]:
                j+=1
            steps = j-i
            # print(f"Interpolating between {signal[i-1]} and {signal[j]}  with {steps} steps")
            # print(signal[i:j])
            delta = (signal[j]-signal[i-1])/steps
            try:
                signal[i:j] = np.arange(signal[i-1], signal[j], delta)
            except:
                fail_count += steps
                pass
            i = j-1
        i += 1
    # print(fail_count)
    return signal

def parse_args():

    parser = argparse.ArgumentParser(description='Dynamixel Plotter')
    parser.add_argument('-f', '--filename', type=str, help="Filename", default="trash.csv")


    return parser.parse_args()

if __name__ == '__main__':


    args = parse_args()

    # Load the CSV file
    df = pd.read_csv(args.filename, header=None)

    # Rename the columns for better readability
    df.columns = ['time', 'puck_x', 'puck_y', 'paddle_x', 'paddle_y']

    # Create subplots

    min_lim = 0
    max_lim = -1
    time = np.array(df['time'][min_lim:max_lim])
    # puck_x = interpolate(np.array(df['puck_x'])[min_lim:max_lim])
    puck_x = np.array(df['puck_x'])[min_lim:max_lim]
    paddle_x = np.array(df['paddle_x'])[min_lim:max_lim]
    # puck_y = interpolate(np.array(df['puck_y'])[min_lim:max_lim])
    puck_y = np.array(df['puck_y'])[min_lim:max_lim]
    paddle_y = np.array(df['paddle_y'])[min_lim:max_lim]

    # pdb.set_trace()
    vel_puck_x = puck_x[1:-1]-puck_x[0:-2]
    vel_puck_y = puck_y[1:-1]-puck_y[0:-2]
    t_vel = time[1:-1]

    idx_vel_x = np.where(vel_puck_x != 0)[0]
    idx_vel_y = np.where(vel_puck_y != 0)[0]


    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Top subplot (puck_x and paddle_x vs time)

    marker_size = 1
    time = np.array(df['time'][min_lim:max_lim])
    axs[0].scatter(time, puck_x, color='red', label='Puck x', s=1)
    axs[0].scatter(time, paddle_x, color='orange', label='Paddle x', s=1)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('X')
    axs[0].set_title('X vs Time')
    axs[0].legend()

    # Bottom subplot (puck_y and paddle_y vs time)
    axs[1].scatter(time, puck_y, color='red', label='Puck y', s=1)
    axs[1].scatter(time, paddle_y, color='orange', label='Paddle y', s=1)
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Y')
    axs[1].set_title('Y vs Time')
    axs[1].legend()


    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.savefig('images/Position_And_Speed.png')
    # plt.show()

    plt.clf()  # Clear the previous plot

    speed = np.sqrt(vel_puck_x**2 + vel_puck_y**2)/100*1000 # /100 as in cm ... *1000 as in ms
    # pdb.set_trace()
    speed = speed[speed!=0]



    max_speed = 5
    cut_speed = 3

    # First plot
    nb_bins = 1000
    hist, bins = np.histogram(speed, bins=nb_bins)
    bin_width = bins[1] - bins[0]
    bins_showed = int(max_speed / bin_width)

    plt.bar(bins[:-1], hist, width=np.diff(bins), align='edge', color='#006666', edgecolor='black')
    plt.title('Puck Speed')
    plt.xlabel('m/s')
    plt.xlim(0, max_speed)
    plt.ylabel(f'Frequency (speed < {cut_speed}m/s)', color='#006666')

    # Second plot
    half_bins = bins[int(bins_showed*cut_speed/max_speed):int(bins_showed)+1]
    half_hist = hist[int(bins_showed*cut_speed/max_speed):int(bins_showed)]

    # Create twin axes
    ax2 = plt.gca().twinx()
    ax2.bar(half_bins[:-1], half_hist, width=np.diff(half_bins), align='edge', color='#330066', edgecolor='black')
    ax2.set_ylabel(f'Frequency (speed > {cut_speed}m/s)', color='#330066')

    # Adjust figure size to ensure proper margin
    plt.gcf().set_size_inches(10, 6)  # Adjust width and height as needed

    plt.savefig('images/Histogram_Speed.png', bbox_inches='tight')  # Save the plot with tight bounding box

    print(f"{len(speed)} samples in {bins_showed} bins")