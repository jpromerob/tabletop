import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
 

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

if __name__ == '__main__':

    args = parse_args()

    # Read the CSV file into a DataFrame
    df = pd.read_csv(f"{args.fname}_summary.csv")

    # Filter rows for 'gpu' and 'spinnaker' pipelines
    gpu_data = df[df['Pipeline'] == 'gpu']
    spinnaker_data = df[df['Pipeline'] == 'spinnaker']

    threshold = 99
    outliers_idx_gpu = find_outliers(gpu_data['Latency'], threshold)
    outliers_idx_spinnaker = find_outliers(spinnaker_data['Latency'], threshold)


    clean_gpu_latency = np.delete(np.array(gpu_data['Latency']), outliers_idx_gpu)
    clean_gpu_error = np.delete(np.array(gpu_data['Error']), outliers_idx_gpu)
    clean_gpu_min_error = np.delete(np.array(gpu_data['MinError']), outliers_idx_gpu)
    clean_gpu_max_speed = np.delete(np.array(gpu_data['MaxSpeed']), outliers_idx_gpu)

    clean_spinnaker_latency = np.delete(np.array(spinnaker_data['Latency']), outliers_idx_spinnaker)
    clean_spinnaker_error = np.delete(np.array(spinnaker_data['Error']), outliers_idx_spinnaker)
    clean_spinnaker_min_error = np.delete(np.array(spinnaker_data['MinError']), outliers_idx_spinnaker)
    clean_spinnaker_max_speed = np.delete(np.array(spinnaker_data['MaxSpeed']), outliers_idx_spinnaker)

    plot_names = ["Latency", "Error", "Min Error", "Max Speed"]   
    unit = ["ms", "mm", "mm", "m/s"]     

    gpu = [
        clean_gpu_latency,
        clean_gpu_error,
        clean_gpu_min_error,
        clean_gpu_max_speed
    ]

    spinnaker = [
        clean_spinnaker_latency,
        clean_spinnaker_error,
        clean_spinnaker_min_error,
        clean_spinnaker_max_speed
    ]

    for i in range(len(gpu)):
        plt.figure(figsize=(8, 6))
        plt.boxplot([gpu[i], spinnaker[i]])
        plt.xticks([1, 2], ["GPU", "SpiNNaker"])
        plt.title(f"Comparison of {plot_names[i]} between GPU and SpiNNaker")
        plt.ylabel(plot_names[i])
        plt.ylim(0, int(gpu[i].max()*1.5))  # Setting y-limits

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

    # Print the consolidated mean values for each column
    print("Consolidated Mean values for GPU pipeline:")
    print(f"Latency: {round(clean_gpu_latency.mean(), 3)} [ms]")
    print(f"Error: {round(clean_gpu_error.mean(), 3)} [mm]")
    print(f"MinError: {round(clean_gpu_min_error.mean(), 3)} [mm]")
    print(f"MaxSpeed: {round(clean_gpu_max_speed.max(), 3)} [m/s]")

    print("\nConsolidated Mean values for Spinnaker pipeline:")
    print(f"Latency: {round(clean_spinnaker_latency.mean(), 3)} [ms]")
    print(f"Error: {round(clean_spinnaker_error.mean(), 3)} [mm]")
    print(f"MinError: {round(clean_spinnaker_min_error.mean(), 3)} [mm]")
    print(f"MaxSpeed: {round(clean_spinnaker_max_speed.max(), 3)} [m/s]")
