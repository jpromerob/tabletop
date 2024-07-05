from PIL import Image
import os

def crop_image(image_path, output_path):
    # Opens a image in RGB mode
    im = Image.open(image_path)
    
    # Size of the image in pixels (size of original image)
    # (This is not mandatory)
    width, height = im.size
    
    # Setting the points for cropped image
    px_ratio = 15
    left = 0
    top = height*px_ratio/100
    right = width
    bottom = height*(100-px_ratio)/100
    
    # Cropped image of above dimension
    # (It will not change original image)
    im1 = im.crop((left, top, right, bottom))
    
    # Shows the image in image viewer
    im1.save(output_path)



def crop_images_in_directory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith(".png"):
            image_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)
            crop_image(image_path, output_path)


input_directory = 'images'
output_directory = 'images'

crop_images_in_directory(input_directory, output_directory)