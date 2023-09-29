import socket
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import multiprocessing
from multiprocessing import Value  # Import Value for shared variable

# Receiver IP and port
receiver_ip = "172.16.222.30"
receiver_port = 4000

# Maximum number of data points to display on the X-axis
max_data_points = 100

# Define shared variables for new_robot_x and new_robot_y
new_robot_x = Value('d', 0.0)  # 'd' indicates a double (floating-point) value
new_robot_y = Value('d', 0.0)

# Global lists for x_data and y_data
x_data = []
y_data = []

# Global axes for subplots
ax1 = None
ax2 = None

# Define a function to update the plots
def animate(i):
    global x_data, y_data, ax1, ax2

    # Access shared variables
    current_x = new_robot_x.value
    current_y = new_robot_y.value

    print(f"({current_x},{current_y})")
    x_data.append(current_x)
    y_data.append(current_y)

    # Keep only the latest 'max_data_points' data points
    if len(x_data) > max_data_points:
        x_data.pop(0)
        y_data.pop(0)

    # Update the x and y subplots
    ax1.clear()
    ax1.plot(x_data, color = 'g')
    ax1.set_title('Robot X')
    ax1.set_ylim(-20,20)  # Set Y-axis limits for the first subplot
    ax2.clear()
    ax2.plot(y_data, color = 'b')
    ax2.set_title('Robot Y')
    ax2.set_ylim(20,80)  # Set Y-axis limits for the second subplot

def receive_data():
    global new_robot_x, new_robot_y

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to the receiver's IP and port
    sock.bind((receiver_ip, receiver_port))

    while True:
        data, _ = sock.recvfrom(1024)
        decoded_data = data.decode()
        current_x, current_y = map(float, decoded_data.split(','))

        # Update shared variables
        new_robot_x.value = current_x
        new_robot_y.value = current_y

def parse_args():
    parser = argparse.ArgumentParser(description='SpiNNaker-SPIF Simulation with Artificial Data')
    parser.add_argument('-t', '--trajectory', type=str, help="Trajectory: linear, circular, spinnaker", default="linear")
    parser.add_argument('-s', '--speed', type=int, help="Speed: 1 to 100", default=10)
    parser.add_argument('-l', '--line', type=int, help="Line: 0 is back, 18 is front", default=0)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Create a multiprocessing Process for data reception
    data_receive_process = multiprocessing.Process(target=receive_data)
    data_receive_process.daemon = True  # Set as a daemon process to exit when the main process exits

    # Start the data reception process
    data_receive_process.start()

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # Set up the animation with cache_frame_data=False to suppress the warning
    ani = animation.FuncAnimation(fig, animate, interval=1, cache_frame_data=False)

    # Show the animated plot
    plt.tight_layout()
    plt.show()
