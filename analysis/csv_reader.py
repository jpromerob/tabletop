import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_csv(filename):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Get the numeric columns
    numeric_columns = df.select_dtypes(include='number')

    # Initialize dictionaries to store column data
    column_arrays = {}

    # Iterate through each numeric column and store its data in arrays
    for col in numeric_columns.columns:
        column_arrays[col] = df[col].to_numpy()

    return column_arrays

def plot_arrays(arrays):
    # Create subplots for each numeric column
    num_plots = len(arrays)
    fig, axs = plt.subplots(num_plots, 1, figsize=(8, 6 * num_plots))

    # Iterate through each numeric column and plot it
    for i, (col, data) in enumerate(arrays.items()):
        ax = axs[i] if num_plots > 1 else axs
        ax.plot(data)
        ax.set_title(col)
        ax.set_xlabel('Index')
        ax.set_ylabel(col)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    arrays = load_csv('data.csv')
    plot_arrays(arrays)
