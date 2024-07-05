import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import re

def file_exists(directory, filename):
    # Construct the full file path
    file_path = os.path.join(directory, filename)
    
    # Check if the file exists
    return os.path.isfile(file_path)

def list_path_numbers(directory):
    # Regular expression to match files named path_<number>.h5
    pattern = re.compile(r'^path_(\d+)\.h5$')
    
    # List to hold the extracted numbers
    numbers = []
    
    # Get a list of all files in the directory
    files = os.listdir(directory)
    
    # Iterate over the files and extract the numbers
    for file in files:
        match = pattern.match(file)
        if match:
            numbers.append(int(match.group(1)))
    
    return numbers

def create_point_cloud_from_frames(cloud):
    width, height, depth = cloud.shape

    # Create an empty list to store points
    points = []

    for z in range(depth):
        frame = cloud[:,:,z]
        for y in range(height):
            for x in range(width):
                intensity = frame[x,y]
                if intensity > 0:  # Consider only non-zero intensity pixels
                    points.append([x, y, z])

    points = np.array(points)
    return points

def display_point_cloud(points, nb_frames, fname):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    

    # pdb.set_trace()

    keep_looking = True
    frames_missing = 5
    pt_ix = points.shape[0]-1
    frame_ix = points[pt_ix,2]
    while keep_looking:
        if frame_ix > points[pt_ix,2]:
            frames_missing -= 1
            frame_ix = points[pt_ix,2]
            if frames_missing == 0:
                keep_looking = False            
        pt_ix -= 1

    # pdb.set_trace()

    # ax.scatter(points[:, 2], points[:, 0], points[:, 1], c=colors, cmap=cmap, marker='.', s=0.5)
    
    cmap = plt.get_cmap('brg')  # Choose a colormap (you can use other colormaps as well)
    colors = points[0:pt_ix, 2]  # Use the z-coordinate as the color values
    ax.scatter(points[0:pt_ix, 2], points[0:pt_ix, 0], points[0:pt_ix, 1],  c=colors, cmap=cmap, marker='.', s=0.1, alpha=0.1)
    ax.scatter(points[pt_ix:-1, 2], points[pt_ix:-1, 0], points[pt_ix:-1, 1], color='g', marker='.', s=0.2)
    
    ax.set_xlim(0, nb_frames) # Z
    ax.set_ylim(0, 256) # X
    ax.set_zlim(0, 165) # Y

    # Hide ticks on X and Z axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    r1 = round(1,2)
    r2 = round(256/165,2)
    r3 = round(165/165,2)

    # Set equal aspect ratio
    ax.set_box_aspect([r1, r2, r3])

    ax.set_xlabel('Time')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    
    # Add a colorbar to show the mapping between z-coordinate values and colors
    # cbar = plt.colorbar(ax.scatter([], [], [], c=[], cmap=cmap, marker='.'), ax=ax)
    # cbar.set_label('Z-coordinate')

    ax.view_init(elev=30, azim=-45)
    
    # plt.savefig(f"images/{fname}_perspective.png", dpi=1200)
    
    ax.view_init(elev=0, azim=0)    


    # Adjust subplot parameters to reduce margins
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    plt.savefig(f"images/{fname}.png", dpi=1200, bbox_inches='tight')




if __name__ == '__main__':


    nb_list = list_path_numbers("paths")

    for suffix in nb_list:
        print(suffix)

        fname = f"path_{suffix}"

        if file_exists("images", f"{fname}_front.png"):
            print(f"Image for {fname} already exists")
            continue
        else:
            print(f"Creating image for {fname}")
            # Load the array from the HDF5 file
            with h5py.File(f'paths/{fname}.h5', 'r') as f:
                cloud = f['cloud'][:]


            nb_frames = cloud.shape[2]

            # Create point cloud
            points = create_point_cloud_from_frames(cloud)

            # Display point cloud
            display_point_cloud(points, nb_frames, fname)
