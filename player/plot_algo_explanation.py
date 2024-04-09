import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d

# Read data from CSV file
df = pd.read_csv('datapoints.csv')

# Extracting data from DataFrame
original_t = df.iloc[:, 0].values
original_x = df.iloc[:, 1].values
original_y = df.iloc[:, -1].values


# sin_waveform = np.sin(0.7*original_t)

# original_x[0:int(len(original_x)/2)] = sin_waveform[0:int(len(original_x)/2)]*original_x[0:int(len(original_x)/2)]

# Interpolation
t = np.linspace(original_t.min(), original_t.max(), 1000)
f_x = interp1d(original_t, original_x, kind='cubic')
f_y = interp1d(original_t, original_y, kind='cubic')
puck_x = f_x(t)
puck_y = f_y(t)


max_x = 10
min_x = - max_x
max_y = 64
half_y = 20
middle_line = max_y-half_y
min_y = max_y-2*half_y
base_right = 28
mirror_line = int((middle_line+base_right)/2)
y_ticks = np.arange(min_y, max_y+1, 4)

paddle_x = np.zeros(puck_x.shape)
paddle_y = np.zeros(puck_y.shape)

for i in range(len(paddle_x)):
    if puck_y[i] > middle_line:
        paddle_x[i] = (max_y-puck_y[i])/half_y*puck_x[i]
    else:
        paddle_x[i] = puck_x[i]


for i in range(len(paddle_y)):
    if puck_y[i] >= middle_line:
        paddle_y[i] = (min_y)+(base_right-min_y)*(max_y-puck_y[i])/half_y
    elif puck_y[i] < middle_line and puck_y[i] > mirror_line:
        paddle_y[i] = mirror_line-abs(puck_y[i]-mirror_line)
    else:
        paddle_y[i] = puck_y[i]

index_middle = np.abs(puck_y - middle_line).argmin()


# Create subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 10), gridspec_kw={'width_ratios': [1, 1, 1.8]})

# x vs t (for puck and paddle)
axs[0,0].plot(t, puck_x, marker='o', color='red')
axs[0,0].axvline(x=t[index_middle], color='black', linestyle='--')
axs[0,0].set_title('Puck: x vs t')
axs[0,0].set_xlabel('t')
axs[0,0].set_ylabel('x')
axs[0,0].set_ylim(min_x-2, max_x+2) 
axs[0,0].text(0.5, 0.95, 't[puck_y=middle]', transform=axs[0,0].transAxes)



axs[1,0].plot(t, paddle_x, marker='o', color='orange')
axs[1,0].axvline(x=t[index_middle], color='black', linestyle='--')
axs[1,0].set_title('Paddle: x vs t')
axs[1,0].set_xlabel('t')
axs[1,0].set_ylabel('x')
axs[1,0].set_ylim(min_x-2, max_x+2) 
axs[1,0].text(0.5, 0.95, 't[puck_y=middle]', transform=axs[1,0].transAxes)


# y vs t (for puck and paddle)
axs[0,1].plot(t, puck_y, marker='o', color='red')
axs[0,1].axvline(x=t[index_middle], color='black', linestyle='--')
axs[0,1].axhline(y=middle_line, color='black', linestyle='--')
axs[0,1].set_title('Puck: y vs t')
axs[0,1].set_xlabel('t')
axs[0,1].set_ylabel('y')
axs[0,1].set_ylim((min_y-2),(max_y+2)) 
axs[0,1].set_yticks(y_ticks)
axs[0,1].text(0.5, 0.95, 't[puck_y=middle]', transform=axs[0,1].transAxes)

axs[1,1].plot(t, paddle_y, marker='o', color='orange')
axs[1,1].axvline(x=t[index_middle], color='black', linestyle='--')
axs[1,1].set_title('Paddle: y vs t')
axs[1,1].set_xlabel('t')
axs[1,1].set_ylabel('y')
axs[1,1].set_ylim((min_y-2),(max_y+2)) 
axs[1,1].set_yticks(y_ticks)
axs[1,1].text(0.5, 0.95, 't[puck_y=middle]', transform=axs[1,1].transAxes)


# x vs y (for puck and paddle) with t encoded in color
sc_puck = axs[0,2].scatter(-puck_y, puck_x, c=t, cmap='Greys', marker='o')
axs[0,2].set_title('Puck: x vs y (color: t)')
axs[0,2].set_xlabel('y')
axs[0,2].set_ylabel('x')
axs[0,2].set_xticks([])
axs[0,2].set_yticks([])
axs[0,2].set_ylim(min_x-2, max_x+2) 
axs[0,2].set_xlim(-(max_y),-(min_y)) 
axs[0,2].axvline(x=-middle_line, color='black', linestyle='-')
axs[0,2].axvline(x=-mirror_line, color='black', linestyle='--')
axs[0,2].axvline(x=-base_right, color='black', linestyle='-')
cbar = fig.colorbar(sc_puck, ax=axs[0,2])
cbar.set_label('t')

sc_paddle = axs[1,2].scatter(-paddle_y, paddle_x, c=t, cmap='Greys', marker='o')
axs[1,2].set_title('Paddle: x vs y (color: t)')
axs[1,2].set_xlabel('y')
axs[1,2].set_ylabel('x')
axs[1,2].set_xticks([])
axs[1,2].set_yticks([])
axs[1,2].set_ylim(min_x-2, max_x+2) 
axs[1,2].set_xlim(-(max_y),-(min_y)) 
axs[1,2].axvline(x=-middle_line, color='black', linestyle='-')
axs[1,2].axvline(x=-mirror_line, color='black', linestyle='--')
axs[1,2].axvline(x=-base_right, color='black', linestyle='-')
axs[1,2].text(0.75, 0.03, 'Mirroring', transform=axs[1,2].transAxes)
cbar = fig.colorbar(sc_paddle, ax=axs[1,2])
cbar.set_label('t')

plt.tight_layout()
plt.savefig('control_summary.png')
# plt.show()
