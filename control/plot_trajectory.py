import pandas as pd
import matplotlib.pyplot as plt
import argparse
import math
import pdb
import numpy as np


def parse_args():

    parser = argparse.ArgumentParser(description='Trajectory Plotter')
    parser.add_argument('-f', '--filename', type=str, help="Filename", default="none")

    return parser.parse_args()


if __name__ == '__main__':


    args = parse_args()

    if args.filename == "none":
        print("Please give a valid file name")
        quit()

    # Load data from CSV file
    data = pd.read_csv(args.filename, header=None, names=['cur_X', 'cur_Y', 'old_X', 'old_Y'])

    # Extract variables
    cur_x = data['cur_X']
    cur_y = data['cur_Y']
    old_x = data['old_X']
    old_y = data['old_Y']

    error_x = np.array(cur_x - old_x)
    error_y = np.array(cur_y - old_y)
    error_d = np.sqrt((error_x)**2+(error_y)**2)


    # Calculate mean and standard deviation
    mean_error_x = error_x.mean()
    std_error_x = error_x.std()

    print(f"Mean of error_x: {round(mean_error_x, 1)} [cm]")
    print(f"Standard deviation of error_x: {round(std_error_x, 1)} [cm]")


    # Calculate mean and standard deviation
    mean_error_y = error_y.mean()
    std_error_y = error_y.std()

    print(f"Mean of error_y: {round(mean_error_y, 1)} [cm]")
    print(f"Standard deviation of error_y: {round(std_error_y, 1)} [cm]")


    # Calculate mean and standard deviation
    mean_error_d = error_d.mean()
    std_error_d = error_d.std()

    print(f"Mean of error_d: {round(mean_error_d, 1)} [cm]")
    print(f"Standard deviation of error_d: {round(std_error_d, 1)} [cm]")

    max_error_d = math.ceil(mean_error_d + 3*std_error_d)

    # Plot variables against time
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Plot original data
    axs[0].plot(cur_x, label='cur_X')
    axs[0].plot(cur_y, label='cur_Y')
    axs[0].plot(old_x, label='old_X')
    axs[0].plot(old_y, label='old_Y')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('X-Y')
    axs[0].set_title('X-Y vs Time')
    axs[0].legend()
    axs[0].grid(True)

    # Plot error data
    axs[1].plot(error_d, label='Error')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Euclidian Error')
    axs[1].set_title('Error vs Time')
    axs[1].legend()
    axs[1].set_ylim(0, max_error_d)
    axs[1].grid(True)

    axs[0].set_xticks([])
    axs[1].set_xticks([])


    plt.tight_layout()
    plt.show()
