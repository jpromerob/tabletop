import cv2
import pdb
import argparse

def upscale_video(filename, scaling_factor):


    input_filename = filename
    output_filename = f"{filename[21:-4]}_x4.avi"

    # pdb.set_trace()

    print(f"{input_filename} --> {output_filename}")

    # Open the video file
    video_capture = cv2.VideoCapture(input_filename)

    # Check if the video file was opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        exit()

    # Get the original video's frame width and height
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))

    # Calculate the new frame dimensions after upscaling
    new_width = int(frame_width * scaling_factor)
    new_height = int(frame_height * scaling_factor)

    # Define the codec and create a VideoWriter object to save the upscaled video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec as needed
    out = cv2.VideoWriter(output_filename, fourcc, 30.0, (new_width, new_height))

    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()

        if not ret:
            break

        # Upscale the frame
        upscaled_frame = cv2.resize(frame, (new_width, new_height))

        # Write the upscaled frame to the output video
        out.write(upscaled_frame)

    # Release the video objects
    video_capture.release()
    out.release()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upscale an AVI video by a factor of 2.3.')
    parser.add_argument('-fn', '--filename', type=str, help='Video filename')
    parser.add_argument('-sf', '--scaling_factor', type=float, help='Scaling factor', default=4.0)
    
    args = parser.parse_args()
    
    upscale_video(args.filename, args.scaling_factor)
