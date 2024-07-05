import aestream
import cv2
import numpy as np
import math
import sys
import time
sys.path.append('../common')
from tools import Dimensions
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb

import open3d as o3d


def parse_args():

    parser = argparse.ArgumentParser(description='Frame Saver')

    parser.add_argument('-p', '--port', type= int, help="Port for events", default=3330)
    parser.add_argument('-s', '--scale', type=int, help="Image scale", default=2)
    parser.add_argument('-l', '--length', type=int, help="Image length", default=256)
    parser.add_argument('-w', '--width', type=int, help="Image width", default=165)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    vis_flag = False

    dim = Dimensions.load_from_file('../common/homdim.pkl')
    if args.length == 0 or args.width == 0:    
        res_x = dim.fl
        res_y = dim.fw
    else:
        res_x = args.length
        res_y = args.width

    new_l = math.ceil(res_x*args.scale)
    new_w = math.ceil(res_y*args.scale)

    if vis_flag:
        window_name = f'Display From AEstream (port {args.port})'
        cv2.namedWindow(window_name)

    # Stream events from UDP port 3333 (default)
    frame = np.zeros((res_x,res_y,3))

    interval = 0.001
    nb_frames = int(1.5/interval)
    cloud = np.zeros((res_x,res_y, nb_frames))

    ix_cloud = 0
    with aestream.UDPInput((res_x, res_y), device = 'cpu', port=args.port) as stream1:
        
        next_time = time.time() + interval
        while True:

            current_time = time.time()
            if current_time >= next_time:
                ix_cloud += 1 
                if ix_cloud == nb_frames:
                    print(f"End of capture after {ix_cloud} frames")
                    break
                next_time += interval


            cloud[:,:,ix_cloud] += stream1.read().numpy()

            if vis_flag:
                frame[0:res_x,0:res_y,1] =  stream1.read().numpy() 
                image = cv2.resize(frame.transpose(1,0,2), (new_l, new_w), interpolation = cv2.INTER_AREA)
                
                center_x = int(image.shape[1] // 2)
                center_y = int(image.shape[0] // 2)
                
                cv2.imshow(window_name, image)            
                cv2.waitKey(1)

    if vis_flag:
        cv2.destroyAllWindows()
    
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

def display_point_cloud(points):
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
    
    plt.savefig(f"my_3d_cloud_perspective.png", dpi=1200)
    
    ax.view_init(elev=0, azim=0)    
    plt.savefig(f"my_3d_cloud_front.png", dpi=1200)


    plt.show()


# Create point cloud
points = create_point_cloud_from_frames(cloud)

# Display point cloud
display_point_cloud(points)



# pdb.set_trace()
        