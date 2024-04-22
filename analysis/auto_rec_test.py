import os
import time

def list_files_with_suffix(directory, suffix):
    # Initialize an empty list to store files with the specified suffix
    files_with_suffix = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file ends with the specified suffix
        if filename.endswith(suffix):
            # If it does, append the file name to the list
            files_with_suffix.append(filename)

    return files_with_suffix

if __name__ == "__main__":
    # Directory path
    directory_path = "/home/juan/tabletop/recordings"

    # Suffix to filter files
    file_suffix = ".aedat4"  # Change this to the desired suffix

    # List files with the specified suffix in the directory
    files = list_files_with_suffix(directory_path, file_suffix)

    # Print the list of files
    print("Files with suffix '{}' in directory '{}':".format(file_suffix, directory_path))
    for file in files:
        print(file)
        os.system(f"python3 rec_analyzer.py -n 5000  -f ~/tabletop/recordings/{file}")
        print("\n\n\n")
        time.sleep(2)

        os.system(f"python3 rec_analyzer.py -n 5000  -f ~/tabletop/recordings/{file} -g")
        print("\n\n\n")
        time.sleep(2)

    
