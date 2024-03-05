import socket
import time
import random

def generate_coordinates():
    while True:
        # Generate random x and y coordinates
        x = random.uniform(-180, 180)
        y = random.uniform(-90, 90)
        
        yield x, y

def send_coordinates():
    # IP and port of the receiver
    receiver_ip = "172.16.222.30"
    receiver_port = 5151
    
    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Generate and send coordinates indefinitely
    for x, y in generate_coordinates():
        # Encode coordinates as bytes
        data = f"{x},{y}".encode()
        
        # Send data to the receiver
        sock.sendto(data, (receiver_ip, receiver_port))
        
        print(f"Sent coordinates: ({x}, {y})")
        
        # Sleep for some time before sending the next coordinates
        time.sleep(1)

if __name__ == "__main__":
    send_coordinates()
