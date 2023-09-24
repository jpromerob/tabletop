import socket
import subprocess
import time
import os
import sys
import argparse
import select

def get_input_with_timeout(prompt, timeout_seconds):
    print(prompt, end=' ', flush=True)
    
    rlist, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    
    if rlist:
        return input()
    else:
        return None
    
def send_signal(receiver_ip, receiver_port):

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Define the number you want to send (1 or 0)
    number_to_send = 1

    # Send the number as a string
    message = str(number_to_send)
    sock.sendto(message.encode(), (receiver_ip, receiver_port))

    # Close the socket
    sock.close()

def build_cmd(ip_address, port, filename, dvx, lut):
    cmd_base = f"/opt/aestream/build/src/aestream"
    cmd_out = f"output udp {ip_address} {port}"
    # output udp 172.16.223.2 3333 172.16.222.199 3330
    if filename == "":
        cmd_in = f"input inivation {dvx[0]} {dvx[1]} dvx"
    else:
        cmd_in = f"input file {filename}"
    cmd_lut = f"resolution 640 480 undistortion {lut}.csv"
    cmd = f"{cmd_base} {cmd_out} {cmd_in} {cmd_lut}"
    return cmd


def allow_manual_setup(args):

    dvx = [args.device_bus, args.device_id]
    command = build_cmd(args.calibrator_ip, args.port_streaming, args.filename, dvx, "cam_lut_undistortion")
    process = subprocess.Popen(command, shell=True)
    cam_location_in_progress = True
    time.sleep(1)
    while(cam_location_in_progress):
        # If the process exceeds the specified duration, terminate it
        timeout_seconds = 2  # Adjust the timeout duration as needed
        user_input = get_input_with_timeout("\rHappy?", timeout_seconds)
        if user_input is not None:
            print("Super Happy")
            cam_location_in_progress = False
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.terminate()
    os.system("pkill -f cam_lut_undistortion.csv")

def enable_calibration(args):

    dvx = [args.device_bus, args.device_id]
    command = build_cmd(args.calibrator_ip, args.port_calibration, args.filename, dvx, "cam_lut_undistortion")
    process = subprocess.Popen(command, shell=True)
    send_signal(args.calibrator_ip, args.port_intercom)
    try:
        process.wait(timeout=args.duration)
    except subprocess.TimeoutExpired:
        process.terminate()
    os.system("pkill -f cam_lut_undistortion.csv")
    time.sleep(3*args.duration)


def make_sure_dvs_ready():
    os.system("pkill -f cam_lut_homography.csv")
    os.system("pkill -f cam_lut_undistortion.csv")



def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-ip', '--calibrator-ip', type=str, help="IP address of calibrator", default="172.16.222.199")
    parser.add_argument('-ps', '--port-streaming', type= int, help="Port for live streaming", default=5050)
    parser.add_argument('-pc', '--port-calibration', type= int, help="Port for calibration", default=5151)
    parser.add_argument('-pi', '--port-intercom', type= int, help="Port for intercom", default=5252)
    parser.add_argument('-db', '--device-bus', type= int, help="Device Bus", default=2)
    parser.add_argument('-di', '--device-id', type= int, help="Device Id", default=2)
    parser.add_argument('-cd', '--duration', type= int, help="Nb of streamed seconds", default=3)
    parser.add_argument('-fn', '--filename', type=str, help="Recording file name", default="")

    return parser.parse_args()

def stream_warped_data(args):
    dvx = [args.device_bus, args.device_id]
    os.system(build_cmd(args.calibrator_ip, args.port_streaming, dvx, "cam_lut_homography"))


if __name__ == '__main__':

    args = parse_args()

    os.system("rm cam_lut_homography.csv")
    make_sure_dvs_ready()
    allow_manual_setup(args)
    enable_calibration(args)
    stream_warped_data(args)