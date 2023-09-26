import torch
import aestream
import time
import cv2
import os
import numpy as np
import math
import argparse
import csv
import os
import sys
sys.path.append('../common')
from tools import get_dimensions
import numpy as np
from scipy.signal import convolve2d
import paramiko
import socket


NATIVE_RES_X = 640
NATIVE_RES_Y = 480

def send_dimensions():
    # Define the connection parameters
    hostname = '172.16.223.5'  # IP address or hostname of Computer B
    port = 22  # Default SSH port is 22
    username = 'juan'
    password = '@Q9ep427x'  # Replace with your actual password

    # Define the local add remote file paths
    local_file_path = '../common/homdim.pkl'  # Path to the file on Computer A
    remote_file_path = '/home/juan/tabletop/common/homdim.pkl'  # Path to the destination on Computer B
    

    # Create an SSH client
    ssh_client = paramiko.SSHClient()

    # Automatically add the server's host key (this is insecure, see note below)
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Connect to the remote server
    ssh_client.connect(hostname, port, username, password)

    # Use SFTP to copy the file from Computer A to Computer B
    with ssh_client.open_sftp() as sftp:
        sftp.put(local_file_path, remote_file_path)

    # Close the SSH connection
    ssh_client.close()

    print(f"File '{local_file_path}' copied to '{remote_file_path}' on {hostname}")


def find_clusters(compressed_array, radius):
    margin = 2*radius
    cluster_list = []
    for x in range(NATIVE_RES_X-margin):
        for y in range(NATIVE_RES_X-margin):
            cluster = compressed_array[x:x+margin, y:y+margin]
            if np.sum(cluster) >= 4:            
                cluster_list.append((x+radius, y+radius))

    cluster_list.append((-1,-1))

    return cluster_list

def find_marker_coordinates(cluster_list, radius):

    margin = 2 *radius
    last_x = -1
    last_y = -1
    sum_x = 0
    sum_y = 0
    marker_list = []
    for x, y in cluster_list:
        # print(f"x={x}, y={y}")
        if abs(last_x-x) + abs(last_y-y) > math.sqrt((2*margin)**2):
            # we have a new cluster

            if last_x > 0 and last_y > 0:
                marker_list.append([round(sum_x/pt_counter), round(sum_y/pt_counter)])

            last_x = x
            last_y = y
            sum_x = 0
            sum_y = 0
            pt_counter = 0
        sum_x += x
        sum_y += y
        pt_counter+= 1

    
    max_sum = 0
    min_sum = NATIVE_RES_X+NATIVE_RES_Y
    avg_x = 0
    for index, point in enumerate(marker_list):
        x, y = point
        avg_x += x/4
        if x+y > max_sum:
            max_sum = x+y
            idx_4th = index
        if x+y < min_sum:
            min_sum = x+y
            idx_2nd = index
    

    for index, point in enumerate(marker_list):
        x, y = point
        if index != idx_2nd and index != idx_4th:
            if x < avg_x:
                idx_1st = index
            else:
                idx_3rd = index

    final_list = []
    final_list.append(marker_list[idx_1st])
    final_list.append(marker_list[idx_2nd])
    final_list.append(marker_list[idx_3rd])
    final_list.append(marker_list[idx_4th])

    print("Messy:")
    print(marker_list)
    print("Sorted:")
    print(final_list)


    return final_list

def get_dst_pts(dim):

    
    pts_dst = np.array([[dim.d2ex, dim.d2ey+dim.iw], #bottom_left
                        [dim.d2ex, dim.d2ey], # top_left
                        [dim.d2ex+dim.il, dim.d2ey], # top_right
                        [dim.d2ex+dim.il, dim.d2ey+dim.iw]], # bottom_right
                       dtype=int)

    return pts_dst

def get_new_coord(x, y, h, dim, marker_list, radius):

    idx_x = int((x*h[0][0]+y*h[0][1]+h[0][2])/(x*h[2][0]+y*h[2][1]+h[2][2]))
    idx_y = int((x*h[1][0]+y*h[1][1]+h[1][2])/(x*h[2][0]+y*h[2][1]+h[2][2]))
    if not(idx_x >= 0 and idx_x < dim.d2ex+dim.il+dim.d2ex and idx_y >= 0 and idx_y < dim.d2ey+dim.iw+dim.d2ey):
        idx_x = -1
        idx_y = -1        

    # Ignore pixels where LEDs are
    for ml_x, ml_y in marker_list:
        if math.sqrt((x-ml_x)**2+(y-ml_y)**2) <= radius :
            idx_x = -1
            idx_y = -1     

    return idx_x, idx_y

def modify_lut(homgra, dim, marker_list, radius):
    
    h = homgra #np.loadtxt('homgra.txt')
    

    # Replace 'your_file.csv' with the path to your CSV file
    csv_file_path = 'cam_lut_undistortion.csv'

    # Open the CSV file in read mode
    with open(csv_file_path, newline='') as csvfile:
        # Create a CSV reader object
        csv_reader = csv.reader(csvfile)

        # Initialize an empty list to store the data
        old_data_list = []

        # Loop through each line in the CSV file
        for row in csv_reader:
            # Convert the row data to a NumPy array and append to the data list
            old_data_list.append(np.array(row))


    new_data_list = []    
    for line in old_data_list:
        valid = True

        element = [line[0],-1,-1,-1,-1]
        if int(line[1]) >= 0 and int(line[2]) >=0:
            x = int(line[1])
            y = int(line[2])
            new_x, new_y = get_new_coord(x, y, h, dim, marker_list, radius)
            element[1] = new_x
            element[2] = new_y
        if int(line[3]) >= 0 and int(line [4]) >=0:
            x = int(line[3])
            y = int(line[4])
            new_x, new_y = get_new_coord(x, y, h, dim, marker_list, radius)
            element[3] = new_x
            element[4] = new_y
        
        if element[1] < 0 or element [2] < 0:
            element = [line[0],-1,-1,-1,-1]

        new_data_list.append(element)


    csv_file_path = 'cam_lut_homography.csv'

    # Open the CSV file in write mode
    with open(csv_file_path, 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file,delimiter=',')

        # Write each element of my_list as a row in the CSV file
        for idx, row in enumerate(new_data_list):
            if idx< 1000000 :
                csv_writer.writerow(new_data_list[idx])
            else:
                csv_writer.writerow(old_data_list[idx])

def shall_calibrate():

    
    receiver_ip = "172.16.222.199"
    receiver_port = 5252

    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((receiver_ip, receiver_port))

    # Set a timeout for the recvfrom operation (e.g., 5 seconds)
    sock.settimeout(1)

    try:
        data, sender_address = sock.recvfrom(1024)
        received_number = int(data.decode())
        print(f"Received: {received_number}")
    except socket.timeout:
        received_number = 0
    finally:
        sock.close()

    if received_number == 1:
        return True
    else:
        return False

def add_markers(im_acc, radius, points, color):

    for idx in range(len(points)):
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                im_acc[points[idx][1]+i, points[idx][0]+j] = color
    
    return im_acc 


def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-p', '--port', type= int, help="Port for events", default=5151)
    parser.add_argument('-t', '--threshold', type= int, help="Threshold for noise filtering", default=5)
    parser.add_argument('-r', '--radius', type= int, help="Cluster radius", default=2)
    parser.add_argument('-e', '--events', type= float, help="Number of events", default=50000)
    parser.add_argument('-vs', '--vis-scale', type=int, help="Visualization scale", default=1)
    parser.add_argument('-hs', '--hom-scale', type=float, help="Homography scale", default=0.8)

    return parser.parse_args()

if __name__ == '__main__':


    args = parse_args()
    dim = get_dimensions(args.hom_scale)
    dim.save_to_file('../common/homdim.pkl')

    # Step 1: Record the start time
    start_time = time.time()

    os.system("rm cam_lut_homography.csv")
    # os.system("rm *.png")
    cv2.namedWindow('TableTopTracker')

    # Stream events from UDP port 3333 (default)
    frame = np.zeros((NATIVE_RES_X,NATIVE_RES_Y,3))
    accumulator = np.zeros((NATIVE_RES_X,NATIVE_RES_Y,3))


    print("Waiting for Calibration signal")
    while(True):
        
        if shall_calibrate():            

            print("Starting new calibration")

            frame_counter = 0
            with aestream.UDPInput((NATIVE_RES_X, NATIVE_RES_Y), device = 'cpu', port=args.port) as stream1:
                        
                while True:

                    frame[0:NATIVE_RES_X,0:NATIVE_RES_Y,1] =  stream1.read().numpy() # Provides a (640, 480) tensor
                    
                    if frame_counter == 0:
                        start_time = time.time()

                    frame_counter += 1

                    total_sum = np.sum(accumulator)

                    if total_sum < args.events:
                        accumulator += frame
                        image = cv2.resize(accumulator.transpose(1,0,2), (math.ceil(NATIVE_RES_X*args.vis_scale),math.ceil(NATIVE_RES_Y*args.vis_scale)), interpolation = cv2.INTER_AREA)
                        # cv2.imshow('TableTopTracker', image)
                        # cv2.waitKey(1)
                    else:
                        planar_acc = np.sum(accumulator, axis=2)
                        compressed_array = np.where(planar_acc > args.threshold, 1, 0)           
                        
                        break

            print("All necessary events collected :)")

            radius = args.radius
            cluster_list = find_clusters(compressed_array, radius)
            marker_list = find_marker_coordinates(cluster_list, radius)

            # Save Accumulated Image with Markers
            im_acc = cv2.resize(10*accumulator.transpose(1,0,2), (math.ceil(NATIVE_RES_X*args.vis_scale),math.ceil(NATIVE_RES_Y*args.vis_scale)), interpolation = cv2.INTER_AREA)
            for x, y in marker_list:
                image[y-radius:y+radius, x-radius:x+radius, :] = [0,0,255]
            cv2.imwrite('Accumulated.png', im_acc)





            pts_src = np.array(marker_list)
            pts_dst = get_dst_pts(dim)

            im_mkd = add_markers(im_acc, radius, pts_src, [0,0,255])
            cv2.imwrite("Marked.png", im_mkd)

            
            # Calculate Homography
            homgra, status = cv2.findHomography(pts_src, pts_dst)
            np.savetxt("homgra.txt", homgra)

            print("Warping Perspective")
            # Warp source image to destination based on homography
            im_fix = cv2.warpPerspective(im_mkd, homgra, (2*dim.d2ex+dim.il, 2*dim.d2ey+dim.iw), flags=cv2.INTER_LINEAR)



            im_mkd = add_markers(im_fix, radius, pts_dst, [0,0,255])
            cv2.imwrite("Corrected.png", im_mkd)

            cv2.destroyAllWindows()

            modify_lut(homgra, dim, marker_list, radius)

            
            # Record the end time
            end_time = time.time()

            # Calculate and print the elapsed time
            elapsed_time = int(end_time - start_time)
            print(f"Elapsed time: {elapsed_time} seconds")

            send_dimensions()

            print("Waiting for NEW Calibration signal")


            
    
