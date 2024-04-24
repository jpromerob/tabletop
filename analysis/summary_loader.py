import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import pdb
 

def find_outliers(arr, val):
    q1 = np.percentile(arr, 100-val)
    q3 = np.percentile(arr, val)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = np.where((arr < lower_bound) | (arr > upper_bound))[0]
    return outliers


def parse_args():

    parser = argparse.ArgumentParser(description='Summary Loader')

    parser.add_argument('-f', '--fname', type= str, help="File Name", default="syn")

    return parser.parse_args()

def indices_above_threshold(arr, threshold):
    indices = []
    for i, value in enumerate(arr):
        if value > threshold:
            indices.append(i)
    return indices

def consolidate(df, pipeline):

    data = df[df['Pipeline'] == pipeline]

    threshold = 85
    outliers_idx = find_outliers(data['Latency'], threshold)

    # Remove Latency Outliers
    clean_latency = np.delete(np.array(data['Latency']), outliers_idx)
    clean_error = np.delete(np.array(data['Error']), outliers_idx)
    clean_min_error = np.delete(np.array(data['MinError']), outliers_idx)
    clean_max_speed = np.delete(np.array(data['MaxSpeed']), outliers_idx)
    clean_dlyd_reps = np.delete(np.array(data['DlydReps']), outliers_idx)*100

    bad_tracking_percentage = 100
    bad_idx = indices_above_threshold(clean_dlyd_reps, bad_tracking_percentage)
    # print(len(bad_idx))

    clean_latency = np.delete(clean_latency, bad_idx)
    clean_error = np.delete(clean_error, bad_idx)
    clean_min_error = np.delete(clean_min_error, bad_idx)
    clean_max_speed = np.delete(clean_max_speed, bad_idx)
    clean_dlyd_reps = np.delete(clean_dlyd_reps, bad_idx)

    conso = [
        clean_latency,
        clean_error,
        clean_min_error,
        clean_max_speed,
        clean_dlyd_reps
    ]


    print(f"\nConsolidated Mean values for {pipeline} pipeline:")
    print(f"Latency: {round(clean_latency.mean(), 3)} [ms]")
    print(f"Error: {round(clean_error.mean(), 3)} [mm]")
    print(f"MinError: {round(clean_min_error.mean(), 3)} [mm]")
    print(f"MaxSpeed: {round(clean_max_speed.max(), 3)} [m/s]")
    print(f"Over {len(clean_latency)} samples")

    # pdb.set_trace()

    return conso

if __name__ == '__main__':

    args = parse_args()

    # Read the CSV file into a DataFrame
    df = pd.read_csv(f"{args.fname}_summary_ok.csv")

    # Filter rows for 'gpu' and 'spinnaker' pipelines
    plot_names = ["Latency", "Error", "Min Error", "Max Speed", "Delayed Reps"]   
    unit = ["ms", "mm", "mm", "m/s", "..."]     

    gpu = consolidate(df, "gpu")
    spinnaker = consolidate(df, "spinnaker")

    for i in range(len(gpu)):
        plt.figure(figsize=(8, 6))
        plt.boxplot([gpu[i], spinnaker[i]])
        plt.xticks([1, 2], ["GPU", "SpiNNaker"])
        plt.title(f"Comparison of {plot_names[i]} between GPU and SpiNNaker")
        plt.ylabel(plot_names[i])
        plt.ylim(0, math.ceil(gpu[i].max()*1.5))  # Setting y-limits

        if unit[i] != "m/s":
            stat = "Mean"
            gpu_val = round(gpu[i].mean(),1)
            spinnaker_val = round(spinnaker[i].mean(),1)
        else:
            stat = "Max"
            gpu_val = round(gpu[i].max(),1)
            spinnaker_val = round(spinnaker[i].max(),1)

        textbox_text = f"{stat} GPU: {gpu_val}[{unit[i]}]\n{stat} SpiNNaker: {spinnaker_val} [{unit[i]}]"
        plt.text(0.35, 0.85, textbox_text, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))

        plt.grid(True)
        plt.savefig(f"{args.fname}_BoxPlot{plot_names[i]}")

