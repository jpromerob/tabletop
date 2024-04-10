import socket
from struct import unpack

P_SHIFT = 15
Y_SHIFT = 0
X_SHIFT = 16
NO_TIMESTAMP = 0x80000000

def decode_data(data, width):
    num_bytes = len(data)
    for i in range(0, num_bytes, 4):
        packed = unpack("<I", data[i:i+4])[0]
        x = (packed >> X_SHIFT) & ((1 << (P_SHIFT - X_SHIFT)) - 1)
        y = (packed >> Y_SHIFT) & ((1 << (X_SHIFT - Y_SHIFT)) - 1)
        print("x:", x, "y:", y)

def receive_data(ip, port, width):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    print("Listening on {}:{}".format(ip, port))
    while True:
        data, _ = sock.recvfrom(1024)
        decode_data(data, width)

if __name__ == "__main__":
    IP_ADDRESS = "127.0.0.1"  # Change to your desired IP address
    PORT = 1387  # Change to your desired port number
    WIDTH = 256  # Change to your desired width value
    receive_data(IP_ADDRESS, PORT, WIDTH)
