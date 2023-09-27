import matplotlib.pyplot as plt
from PIL import Image

# Load the PNG image using PIL (Pillow)
image = Image.open('Accumulated.png')

# Convert the image to a NumPy array (optional but useful for further processing)
image_array = plt.imread('Accumulated.png')

# Create a Matplotlib figure and axis
fig, ax = plt.subplots()

# Display the image on the axis
ax.imshow(image_array)

# Optional: Set axis labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Image Plot')

# Show the image
plt.show()