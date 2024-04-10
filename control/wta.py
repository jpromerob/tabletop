
import aestream
import numpy as np
import socket
import sys
sys.path.append('../common')
from tools import Dimensions, get_shapes
import argparse

# Define constants for UDP IP and port
UDP_IP = '172.16.222.30'
PORT_UDP_PADDLE_DESIRED_XY = 6262  # Choose an appropriate port number


def parse_args():

    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')

    parser.add_argument('-b', '--board', type= int, help="Board sending events", default=43)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    in_port = args.board*100+87

    dim = Dimensions.load_from_file('../common/homdim.pkl')

    x_coord = int(dim.fl/2)
    y_coord = int(dim.fw)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    with aestream.UDPInput((dim.fl, dim.fw), device = 'cpu', port=in_port) as stream1:
                
        while True:

            reading = stream1.read().numpy() 
            idx_x = np.where(reading[:,0]>0.5)
            idx_y = np.where(reading[:,1]>0.5)

            try:
                if len(idx_x[0])>0 and len(idx_y[0]) > 0:
                    x_coord = int(np.mean(idx_x))
                    y_coord = int(np.mean(idx_y))
            except:
                pass       
            x_norm = x_coord/dim.fl*100
            y_norm = y_coord/dim.fw*100

            data = "{},{}".format(x_norm, y_norm).encode()
            sock.sendto(data, (UDP_IP, PORT_UDP_PADDLE_DESIRED_XY))






        