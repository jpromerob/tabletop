import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import pdb

class ControlNet(nn.Module):
    def __init__(self, N, M):
        super(ControlNet, self).__init__()
        self.N = N
        self.M = M
        self.input_size = N * M
        self.output_size = N * M

        # Define the parameters of the network
        self.output_layer = nn.Linear(self.input_size, self.output_size)

        # Initialize weights
        self.output_layer.weight.data.zero_()  # Fill weights with zeros
        for i in range(N):
            for j in range(M):
                self.output_layer.weight.data[M*i+j, M*i+(M-1-j)] = 10

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.output_layer(x)
        x = x.view(-1, self.N, self.M)
        return x

# Load the PNG image
image_path = 'Pat3.png'
input_image = Image.open(image_path).convert('L')  # Convert to grayscale if necessary


# Convert the image to a tensor
input_data = torch.tensor(np.array(input_image), dtype=torch.float32).unsqueeze(0)  # Add batch dimension
input_image = Image.fromarray(input_data.squeeze().detach().numpy().astype('uint8'))
input_image.save('Input.png')
input_size = input_data.size()
print(f"Input size: {input_image.size}")

# pdb.set_trace()

# Resize the image to fit your model's input size (N x M)
model = ControlNet(input_size[1], input_size[2])

# Forward pass through the model
output_data = model(input_data)
print(f"Input size: {input_image.size}")

# Convert the output tensor to a PIL image
output_image = Image.fromarray(output_data.squeeze().detach().numpy().astype('uint8'))

# Save the output image as PNG
output_image.save('Output.png')
