from PIL import Image
import os
import pdb

# Directory path containing the PNG images
images_directory = 'images/'
# Output GIF file path
output_gif_path = 'output.gif'

# Function to extract the number from the file name
def extract_number(filename):
    return int(filename.split('_')[1].split('.')[0])

# List all the PNG image files in the directory
image_files = [f for f in os.listdir(images_directory) if f.endswith('.png')]

# Sort the image files using the custom sorting function
image_files.sort(key=extract_number)

# Create an empty list to store the image frames
frames = []

# Load each image and append it to the frames list
for image_file in image_files:
    image_path = os.path.join(images_directory, image_file)
    image = Image.open(image_path)
    frames.append(image)

# Save the frames as an animated GIF
frames[0].save(output_gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=5, loop=0)