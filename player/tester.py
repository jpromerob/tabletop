import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pdb

# Load the PNG image
image_path = 'Pat7.png'

input_image = np.array(Image.open(image_path).convert('L'))  # Convert to grayscale if necessary
green_image = np.stack((input_image,) * 3, axis=-1)
green = np.array([0, 255, 0], dtype=np.uint8)
white_pixels = input_image == 255
green_image[white_pixels] = green


mirrored_image = np.fliplr(input_image)
blue_image = np.stack((mirrored_image,) * 3, axis=-1)
blue = np.array([0, 0, 255], dtype=np.uint8)
white_pixels = mirrored_image == 255
blue_image[white_pixels] = blue

image_shape = green_image.shape

black_image = np.zeros(image_shape, dtype=np.uint8)  # Adjust the size as needed
extracted_normal = np.zeros(image_shape, dtype=np.uint8)  # Adjust the size as needed
extracted_mirrored = np.zeros(image_shape, dtype=np.uint8)  # Adjust the size as needed
together = np.zeros(image_shape, dtype=np.uint8)  # Adjust the size as needed

p_base_line = 0.03 # %
p_mirror_line = 0.5 # %
p_gap = (p_mirror_line-p_base_line)/2


gap = int(input_image.shape[1]*(p_gap))

mirror = int(input_image.shape[1]*(p_mirror_line))
middle_left = mirror-gap
base_left = mirror-2*gap


mirror = mirror
middle_right = mirror+gap
base_right = mirror+2*gap


print(f"{middle_right-mirror} vs {middle_right-mirror}")
print(f"{mirror-middle_left} vs {middle_left-base_left}")

print(f"{base_left} {middle_left} {mirror}")
print(f"{mirror} {middle_right} {base_right}")


extracted_normal[:,mirror:middle_right,:] = green_image[:,mirror:middle_right,:]

extracted_mirrored[:,middle_left:middle_left+gap,:] = blue_image[:,middle_left:mirror,:]

together[:,mirror:middle_right,:] = green_image[:,mirror:middle_right,:]
together[:,middle_right:middle_right+gap,:] = blue_image[:,middle_left:mirror,:]

# Create a figure with 2x2 subplots
fig, axs = plt.subplots(5,1, figsize=(4,15))


for i in range(5):
    axs[i].axvline(x=base_left, color='r', linestyle='--')
    axs[i].axvline(x=middle_left, color='r', linestyle='--')
    axs[i].axvline(x=mirror, color='r', linestyle='--')
    axs[i].axvline(x=middle_right, color='r', linestyle='--')
    axs[i].axvline(x=base_right, color='r', linestyle='--')

# Plot the original image
axs[0].imshow(green_image, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')

# Plot the mirrored version of the image
axs[1].imshow(blue_image, cmap='gray')
axs[1].set_title('Mirrored Image')
axs[1].axis('off')

# Plot the original image
axs[2].imshow(extracted_normal, cmap='gray')
axs[2].set_title('Extracted Normal Image')
axs[2].axis('off')

# Plot the original image
axs[3].imshow(extracted_mirrored, cmap='gray')
axs[3].set_title('Extracted Mirrored Image')
axs[3].axis('off')

# Plot the original image
axs[4].imshow(together, cmap='gray')
axs[4].set_title('Together Image')
axs[4].axis('off')

# Leave the other two subplots empty for now
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
axs[3].axis('off')
axs[4].axis('off')

plt.show()

# Save the 'together' image as a PNG file
plt.imsave('together.png', together)
