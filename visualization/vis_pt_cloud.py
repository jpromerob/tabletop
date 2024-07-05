import numpy as np
import cv2
import open3d as o3d

def create_point_cloud_from_frames(frames):
    depth = len(frames)
    height, width = frames[0].shape
    
    # Create an empty list to store points
    points = []
    
    for z in range(depth):
        frame = frames[z]
        for y in range(height):
            for x in range(width):
                intensity = frame[y, x]
                if intensity > 0:  # Consider only non-zero intensity pixels
                    points.append([x, y, z])
    
    points = np.array(points)
    return points

def display_point_cloud(points):
    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd])

# Load your frames here (this is just a placeholder)
N = 10  # Number of frames
frames = [cv2.imread(f'frame_{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(N)]

# Create point cloud
points = create_point_cloud_from_frames(frames)

# Display point cloud
display_point_cloud(points)
