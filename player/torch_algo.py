import torch
from PIL import Image
import numpy as np
import pdb
import time

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


        p_base_line = 0.03 # %
        p_mirror_line = 0.5 # %
        p_gap = (p_mirror_line-p_base_line)/2

        mirror = int(M/2)
        gap = int(M*p_gap)
        middle_left = mirror-gap
        base_left = mirror-2*gap
        middle_right = mirror+gap
        base_right = mirror+2*gap


        print(f"Connectivity")
        pts = 0
        for i in range(N): # height
            for j in range(M): # width

                # Far pitch
                if j <= mirror:
                    self.map_far.append((j, i, M-1-int(j*base_left/mirror), int((i-N/2)*j/mirror+N/2)))
                    # self.map_far.append((j, i, int(j), int((i-N/2)*j/mirror+N/2)))
                    pts += 1
                elif j < middle_right:
                    self.map_far.append((j, i, M-1-(j-mirror+base_left), i))
                    pts += 1
                else:
                    # self.map_far.append((j, i, M-1-0, int(N/2)))
                    self.map_far.append((j, i, j, i))
                    pts += 1

        print(f"We have {pts} connections")


        for i in range(pts):
            try:
                input_index = self.map_far[i][1]*M+self.map_far[i][0]
                output_index = self.map_far[i][3]*M+self.map_far[i][2]
                self.output_layer.weight.data[:, input_index] = 0
                self.output_layer.weight.data[output_index, input_index] = 4
            except:
                pdb.set_trace()

        print(f"Done with weights")

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.output_layer(x)
        x = x.view(-1, self.N, self.M)
        return x

# Load the PNG image
image_path = 'Pat7.png'
input_image = Image.open(image_path).convert('L')  # Convert to grayscale if necessary



# Convert the image to a tensor
input_data = torch.tensor(np.array(input_image), dtype=torch.float32).unsqueeze(0)  # Add batch dimension
input_image = Image.fromarray(input_data.squeeze().detach().numpy().astype('uint8'))
input_image.save('Input.png')
input_size = input_data.size()
print(f"Input size: {input_image.size}")



# Resize the image to fit your model's input size (N x M)
model = ControlNet(164, 255)

# Forward pass through the model
output_data = model(input_data)

# Convert the output tensor to a PIL image
output_image = Image.fromarray(output_data.squeeze().detach().numpy().astype('uint8'))

# Save the output image as PNG
output_image.save('Output.png')

