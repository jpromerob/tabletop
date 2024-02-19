import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Create some data
data = np.random.rand(10, 10) * 2 - 1  # Random data between -1 and 1

# Define the colors for the colormap
colors = [(0, 'red'), 
          (0.5, 'black'), 
          (1, 'green')]

# Create the custom colormap
cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)

# Define a custom normalization that maps negative values to 0, positive values to 1, and zero to 0.5
norm = Normalize(vmin=-1, vmax=1)

# Plot with custom colormap and normalization
plt.imshow(data, cmap=cmap, norm=norm)
plt.colorbar()
plt.show()
