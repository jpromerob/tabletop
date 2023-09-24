import torch
import torch.nn as nn
import pdb
import random
import numpy as np
import math
import time

import matplotlib.pyplot as plt


def make_kernel_circle(r, k_sz,weight, kernel):
    # pdb.set_trace()
    var = int((k_sz+1)/2-1)
    a = np.arange(0, 2 * math.pi, 0.01)
    dx = np.round(r * np.sin(a)).astype("uint32")
    dy = np.round(r * np.cos(a)).astype("uint32")
    kernel[var + dx, var + dy] = weight


# The one in the video (using original recordings)
scaler = 0.08
k_sz = 41
pos_w = 0.8
neg_w = -1.0
print(k_sz)
kernel = np.zeros((k_sz, k_sz), dtype=np.float32)
make_kernel_circle(int(38/2), k_sz, pos_w*scaler, kernel) # 38px
make_kernel_circle(int(36/2), k_sz, pos_w*scaler, kernel) # 36px
make_kernel_circle(int(29/2), k_sz, neg_w*scaler, kernel) # 29px
make_kernel_circle(int(27/2), k_sz, neg_w*scaler, kernel) # 27px
make_kernel_circle(int(20/2), k_sz, pos_w*scaler, kernel) # 20 px
make_kernel_circle(int(19/2), k_sz, pos_w*scaler, kernel) # 19 px
make_kernel_circle(int(15/2), k_sz, neg_w*scaler, kernel) # 15 px
make_kernel_circle(int(13/2), k_sz, neg_w*scaler, kernel) # 13 px
make_kernel_circle(int(9/2), k_sz, pos_w*scaler, kernel) # 9 px
make_kernel_circle(int(7/2), k_sz, pos_w*scaler, kernel) # 7 px
make_kernel_circle(int(4/2), k_sz, neg_w*scaler, kernel) # 4 px
make_kernel_circle(int(2/2), k_sz, neg_w*scaler, kernel) # 2 px

custom_kernel = torch.from_numpy(kernel)

class CustomCNN(nn.Module):
    def __init__(self, custom_kernel):
        super(CustomCNN, self).__init__()
        self.custom_kernel = nn.Parameter(custom_kernel, requires_grad=False)  # Make the kernel a learnable parameter

    def forward(self, x):
        x = nn.functional.conv2d(x, self.custom_kernel.unsqueeze(0).unsqueeze(0))  # Apply convolution
        return x

custom_cnn = CustomCNN(custom_kernel)
                              
input_image = torch.zeros(1, 1, 480, 640)  # Batch size of 1, 1 channel, height 480, width 640
rnd_x = [random.randint(0, 640-k_sz-1) for _ in range(3)]
rnd_y = [random.randint(0, 480-k_sz-1) for _ in range(3)]

for i in range(3):
    input_image[0,0,rnd_y[i]:rnd_y[i]+k_sz,rnd_x[i]:rnd_x[i]+k_sz] = custom_kernel



# Convert the PyTorch tensor to a NumPy array
image_in = input_image.numpy()
image_in = image_in.squeeze() 
plt.imshow(image_in, cmap='gray')
plt.axis('off')
plt.savefig('input.png', bbox_inches='tight', pad_inches=0, format='png')
plt.clf()



# Record the start time
start_time = time.time()

output_image = custom_cnn(input_image)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = int(1000*(end_time - start_time))

print(f"Time elapsed: {elapsed_time:.4f} ms")


# Convert the PyTorch tensor to a NumPy array
image_out = output_image.numpy()
image_out = image_out.squeeze()
plt.imshow(image_out, cmap='gray') 
plt.axis('off') 
plt.savefig('output.png', bbox_inches='tight', pad_inches=0, format='png')
plt.clf()