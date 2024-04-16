import numpy as np
import pdb
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
sys.path.append('../common')

PLOT_FLAG = False

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        self.kernel = np.load(f"../common/fast_kernel.npy")
        self.ksz = self.kernel.shape[0]
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=self.ksz, bias=False)
        self.conv.weight.data = torch.FloatTensor(self.kernel).unsqueeze(0).unsqueeze(0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x) 
        return x

# Example usage
if __name__ == "__main__":

    time_array = np.load("time_array.npy")
    online_coordinates = np.load("coordinate_array.npy")
    frame_array = np.load("frame_array.npy")


    delta_t = 3 # bin size in [ms]
    threshold = 100
    nb_pts = time_array.shape[0]-delta_t

    model = CustomCNN()
    offline_coordinates = np.zeros((nb_pts,2), dtype=int)

    for i in np.linspace(int(delta_t/2), nb_pts, nb_pts - int(delta_t/2) + 1).astype(int):
        
        if i%100 == 0:
            print(f"Analyzing frame #{i}")

        sub_frame = frame_array[:,:,i-int(delta_t/2):i+int(delta_t/2)+1]
        in_frame = np.sum(sub_frame, axis=-1)

        # pdb.set_trace()

        if PLOT_FLAG:
            plt.imsave(f'images/frame_test_input.png', in_frame)
        in_max_index = np.argmax(in_frame)
        in_max_index_2d = np.unravel_index(in_max_index, in_frame.shape)

        out_frame = np.zeros(in_frame.shape)

        in_tensor = torch.tensor(in_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        out_tensor = model(in_tensor)
        out_x = out_tensor.shape[2]
        out_y = out_tensor.shape[3]
        offset = int(model.ksz/2)
        out_frame[offset:offset+out_x, offset:offset+out_y] = out_tensor.squeeze().detach().numpy()

        if PLOT_FLAG:
            plt.imsave(f'images/frame_test_output.png', out_frame)


        # pdb.set_trace()

        # Find the index of the maximum value
        out_max_index = np.argmax(out_frame)
        out_max_index_2d = np.unravel_index(out_max_index, out_frame.shape)
        
        if out_frame[out_max_index_2d] > threshold:
            offline_coordinates[i,0] = int(out_max_index_2d[0])
            offline_coordinates[i,1] = int(out_max_index_2d[1])
        else:
            offline_coordinates[i,0] = int(offline_coordinates[i-1,0])
            offline_coordinates[i,1] = int(offline_coordinates[i-1,1])


        target_frame = in_frame
        target_frame[offline_coordinates[i,0], offline_coordinates[i,1]] = 3*in_frame[in_max_index_2d]
    
        if PLOT_FLAG:
            plt.imsave(f'images/frame_test_target.png', target_frame)
        
        if i >= nb_pts*0.9:
            break
    

    last_element = int(len(online_coordinates)*0.75)

    online_x = online_coordinates[1:last_element, 0]
    offline_x = offline_coordinates[1:last_element, 1]
    online_y = online_coordinates[1:last_element, 1]
    offline_y = offline_coordinates[1:last_element, 0]


    np.save('online_x.npy', online_x)
    np.save('offline_x.npy', offline_x)
    np.save('online_y.npy', online_y)
    np.save('offline_y.npy', offline_y)

    

