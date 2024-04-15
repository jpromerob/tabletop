import argparse
import sched
import time
import socket

# Add necessary imports
import numpy as np
from struct import pack


P_SHIFT = 15
Y_SHIFT = 0
X_SHIFT = 16
NO_TIMESTAMP = 0x80000000

global sock 
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Define your function here
def forward_data(spikes, ip, port, width):
    global sock
    data = b""
    np_spikes = np.array(spikes)
    for i in range(np_spikes.shape[0]):      
        x = int(np_spikes[i] % width)
        y = int(np_spikes[i] / width)
        polarity = 1
        packed = (NO_TIMESTAMP + (polarity << P_SHIFT) + (y << Y_SHIFT) + (x << X_SHIFT))
        data += pack("<I", packed)
    sock.sendto(data, (ip, port))
    # sock.sendto(data, ("172.16.222.30", 1387))

# Create a scheduler
scheduler = sched.scheduler(time.time, time.sleep)

def coordinate_generator(length, width):
    while(True):
        for y in range(width):
            for x in range(length):
                for i in range(10):
                    yield x, y

def main():
    parser = argparse.ArgumentParser(description="Script to call forward_data function every 100 microseconds.")
    parser.add_argument("-ip", type=str, default="172.16.222.30", help="Destination IP address (default: 127.0.0.1)")
    parser.add_argument("-port", type=int, default=3331, help="Destination port number (default: 8080)")
    parser.add_argument("-x", type=int, default=256, help="Size X axis")
    parser.add_argument("-y", type=int, default=165, help="Size Y axis")

    args = parser.parse_args()

    ip = args.ip
    port = args.port
    length = args.x
    width = args.y

    gen = coordinate_generator(length, width)

    while(True):

        x, y = next(gen)
        spikes = [y*length+x]  # Replace with your data
        forward_data(spikes, ip, port, length)
        time.sleep(0.001)
    # # Replace 'spikes' with appropriate data
    # gen = coordinate_generator(N)
    # x, y = nex(gen)
    # spikes = [args.y*WIDTH+args.x]  # Replace with your data
    
    # # Start the scheduler
    # scheduler.enter(0, 1, call_forward_data, (scheduler, spikes, args.ip, args.port))
    # scheduler.run()

if __name__ == "__main__":
    main()
