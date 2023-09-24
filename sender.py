import socket
import time
import random

# IP and port of the receiver (Computer B)
receiver_ip = "172.16.222.30"
receiver_port = 5151

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    # Generate random x and y coordinates
    x = random.randint(-10, 10)
    y = random.randint(26, 42)

    # Construct the message as a string, e.g., "10,20"
    message = f"{x},{y}"
    
    # Send the message to the receiver
    sock.sendto(message.encode(), (receiver_ip, receiver_port))
    
    # Print what was sent
    print(f"Sent: {message}")
    
    # Wait for 1 second before sending the next message
    time.sleep(2)