
import numpy as np

def split_and_sum(original_array, block_size):
    n, m = original_array.shape
    result_array = original_array.reshape(n // block_size, block_size, m // block_size, block_size).sum(axis=(1, 3))
    return result_array


import numpy as np

# Create a random 4x4 integer NumPy array for demonstration
original_array = np.random.randint(1, 5, (12, 7))  # Generates random integers between 1 and 4
print(original_array)
result_array = split_and_sum(original_array, block_size=4)
print(result_array)