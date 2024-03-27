import torch
import aestream
import time
import cv2
import math
import argparse
import numpy as np
import pdb
from PIL import Image
from torch.quantization import quantize_dynamic


class ControlNet(torch.nn.Module):
    def __init__(self, N, M):
        super(ControlNet, self).__init__()
        self.N = N
        self.M = M
        self.input_size = N * M
        self.output_size = N * M

        # Define the parameters of the network
        self.output_layer = torch.nn.Linear(self.input_size, self.output_size)

        # Initialize weights
        self.output_layer.weight.data.zero_()  # Fill weights with zeros

        # List to store tuples of connections
        self.inputs = []
        self.outputs = []
        self.map_far = []

        # Connectivity calculation
        p_base_line = 0.1  # %
        p_mirror_line = 0.5  # %
        p_gap = (p_mirror_line - p_base_line) / 2

        mirror = int(M / 2)
        gap = int(M * p_gap)
        middle_left = mirror - gap
        base_left = mirror - 2 * gap
        middle_right = mirror + gap
        base_right = mirror + 2 * gap

        print(f"Connectivity")
        pts = 0
        for i in range(N):  # height
            for j in range(M):  # width

                # Far pitch
                if j <= mirror:
                    self.map_far.append((j, i, M - 1 - int(j * base_left / mirror), int((i - N / 2) * j / mirror + N / 2)))
                    pts += 1
                elif j < middle_right:
                    self.map_far.append((j, i, M - 1 - (j - mirror + base_left), i))
                    pts += 1
                else:
                    self.map_far.append((j, i, j, i))
                    pts += 1

        print(f"We have {pts} connections")

        for i in range(pts):
            try:
                input_index = self.map_far[i][1] * M + self.map_far[i][0]
                output_index = self.map_far[i][3] * M + self.map_far[i][2]
                self.output_layer.weight.data[:, input_index] = 0
                self.output_layer.weight.data[output_index, input_index] = 4
            except:
                pdb.set_trace()

        print(f"Done with weights")

    def forward(self, x):
        x = x.reshape(-1, self.input_size)
        x = self.output_layer(x)
        x = x.view(-1, self.N, self.M)
        return x

def parse_args():
    parser = argparse.ArgumentParser(description='Automatic Coordinate Location')
    parser.add_argument('-p', '--port', type=int, help="Port for events", default=5050)
    parser.add_argument('-s', '--scale', type=int, help="Image scale", default=1)
    parser.add_argument('-l', '--length', type=int, help="Image length", default=255)
    parser.add_argument('-w', '--width', type=int, help="Image width", default=164)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    res_x = args.length
    res_y = args.width
    new_l = math.ceil(res_x*args.scale)
    new_w = math.ceil(res_y*args.scale)
    window_name = 'Airhockey Display'
    cv2.namedWindow(window_name)

    # Stream events from UDP port 3333 (default)
    frame = np.zeros((res_x,res_y,3))

    # Load the PNG image
    image_path = 'Pat7.png'
    input_image = Image.open(image_path).convert('L')  # Convert to grayscale if necessary

    # Convert the image to a tensor
    input_data = torch.tensor(np.array(input_image), dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to CPU initially for quantization
    model = ControlNet(164, 255).cpu()

    # Quantize the model's weights to low precision
    quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    # Move quantized model to the device
    quantized_model.to(device)

    # Get the size of the quantized model
    model_size = get_model_size(quantized_model)


    # Set model to evaluation mode (no gradients)
    quantized_model.eval()

    k_sz = 16

    with torch.no_grad():
        with aestream.UDPInput((res_x, res_y), device='cpu', port=args.port) as stream1:
            while True:
                start_time = time.time()
                reading = stream1.read()
                reshaped_data = torch.tensor(np.transpose(reading), dtype=torch.float32).unsqueeze(0)
                device_input_data = reshaped_data.to(device)
                output_data_device = quantized_model(device_input_data)
                output_data = output_data_device.cpu().squeeze().numpy()
                frame[0:res_x, 0:res_y, 1] = np.transpose(output_data, (1, 0))
                end_time = time.time()
                elapsed_time = end_time - start_time
                image = cv2.resize(frame.transpose(1,0,2), (new_l, new_w), interpolation=cv2.INTER_AREA)
                cv2.imshow(window_name, image)
                cv2.waitKey(1)
