import pandas as pd
import pdb

# Read the CSV file into a DataFrame
df = pd.read_csv("summary.csv")

# Filter rows for 'gpu' and 'spinnaker' pipelines
gpu_data = df[df['Pipeline'] == 'gpu']
spinnaker_data = df[df['Pipeline'] == 'spinnaker']

# Calculate mean for each column separately
gpu_latency_mean = round(gpu_data['Latency'].mean(),1)
gpu_error_mean = round(gpu_data['Error'].mean(),1)
gpu_min_error_mean = round(gpu_data['MinError'].mean(),1)
gpu_max_speed_max = round(gpu_data['MaxSpeed'].max(),1)
spinnaker_latency_mean = round(spinnaker_data['Latency'].mean(),1)
spinnaker_error_mean = round(spinnaker_data['Error'].mean(),1)
spinnaker_min_error_mean = round(spinnaker_data['MinError'].mean(),1)
spinnaker_max_speed_max = round(spinnaker_data['MaxSpeed'].max(),1)

# Print the consolidated mean values for each column
print("Consolidated Mean values for GPU pipeline:")
print(f"Latency: {gpu_latency_mean} [ms]")
print(f"Error: {gpu_error_mean} [mm]")
print(f"MinError: {gpu_min_error_mean} [mm]")
print(f"MaxSpeed: {gpu_max_speed_max} [m/s]")

print("\nConsolidated Mean values for Spinnaker pipeline:")
print(f"Latency: {spinnaker_latency_mean} [ms]")
print(f"Error: {spinnaker_error_mean} [mm]")
print(f"MinError: {spinnaker_min_error_mean} [mm]")
print(f"MaxSpeed: {spinnaker_max_speed_max} [m/s]")

