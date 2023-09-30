
import os
import argparse
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upscale an AVI video by a factor of 2.3.')
    parser.add_argument('-p', '--path', type=str, help='Path', default="")
    parser.add_argument('-sf', '--scaling_factor', type=float, help='Scaling factor', default=3.0)
    
    args = parser.parse_args()

    # Use glob to list all AVI files in the folder
    avi_files = glob.glob(os.path.join(args.path, '*.avi'))

    # Print the list of AVI files
    for file_path in avi_files:
        os.system(f"python3 ~/tabletop/visualization/replay.py -fn {file_path}")
